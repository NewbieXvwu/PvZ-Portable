"""Read-only task69 measurements around either frozen teacher source/build.

Usage: python THIS.py /absolute/source/root [unchanged search_play arguments]
The observer consumes existing observations/events. It never queries or changes game state,
search parameters, candidate scores, futures, RNG, or the action returned by the teacher.
"""
from collections import Counter
from pathlib import Path
import sys

search_play = None
_ORIGINAL_PLAY = _ORIGINAL_VERIFY = None
_ACTIVE = None


class Measurements:
    def __init__(self):
        self.sun = Counter()
        self.verification = {}
        self.selected = Counter()
        self.replacements = []
        self.attempts = self.rejected = 0
        self.previous = None
        self.final_sun = None

    def before(self, env, obs, fast_state, decisions):
        self.previous = obs

    def after(self, obs, action, policy_choice, rejected, info):
        self.final_sun = obs['sun']
        for name in ('sun_produced', 'sun_spent'):
            self.sun[name] += info.get('events', {}).get(name, 0)
        if action['type'] != 'replace':
            return
        self.attempts += 1
        self.rejected += int(rejected)
        if rejected:
            return
        def types_at(board):
            return Counter(p['imitater_type'] if p['type'] == 48 else p['type']
                           for p in board.get('plants', [])
                           if (p['row'], p['col']) == (action['row'], action['col']))
        old, new = types_at(self.previous), types_at(obs)
        self.replacements.append(dict(tick=self.previous['tick'], **action,
            removed=list((old - new).elements()), added=list((new - old).elements()),
            sun_before=self.previous['sun'], sun_after=obs['sun']))

    def result(self):
        return dict(verification_by_type=self.verification, search_selected_by_type=dict(self.selected),
                    replace_attempts=self.attempts, replace_rejected=self.rejected,
                    replace_events=self.replacements, **self.sun,
                    sun_remaining=self.final_sun)


def measured_verification(*args, **kwargs):
    stats = args[11] if len(args) > 11 else kwargs['stats']
    candidates, best = args[3], args[6]
    result = _ORIGINAL_VERIFY(*args, **kwargs)
    if _ACTIVE is not None:
        _ACTIVE.selected[candidates[best]['type']] += 1
        verification = stats.get('last_verification')
        if verification:
            kind = verification['challenger']['type']
            bucket = _ACTIVE.verification.setdefault(kind, dict(attempts=0, revoked=0))
            bucket['attempts'] += 1
            bucket['revoked'] += int(not verification['after_override'])
    return result


def measured_play(job):
    global _ACTIVE
    _ACTIVE = Measurements()
    try:
        result = _ORIGINAL_PLAY(job, observer=_ACTIVE)
        result['study'] = _ACTIVE.result()
        return result
    finally:
        _ACTIVE = None


def main():
    global search_play, _ORIGINAL_PLAY, _ORIGINAL_VERIFY
    source = Path(sys.argv.pop(1)).resolve()
    sys.path[:0] = [str(source / 'python'), str(source / 'scripts')]
    import search_play as teacher
    search_play = teacher
    if Path(teacher.__file__).resolve() != source / 'scripts/search_play.py':
        raise RuntimeError('measurement imported the wrong frozen teacher')
    _ORIGINAL_PLAY, _ORIGINAL_VERIFY = teacher._play, teacher._verified_choice
    teacher._verified_choice, teacher._play = measured_verification, measured_play
    teacher.main()


if __name__ == '__main__':
    main()
