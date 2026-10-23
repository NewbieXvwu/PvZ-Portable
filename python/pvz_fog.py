"""Read-only fog episode diagnostics, isolated from all policy inputs."""
from collections import Counter


class FogEpisodeDiagnostics:
    def __init__(self):
        self.first_seen = {}
        self.seen_hidden = set()
        self.plantings = Counter()
        self.loss = None
        self.hidden_rows = set()
        self.agreement = Counter()

    def sample(self, env, obs):
        raw = env.privileged_state()
        if 'fog_loss' not in raw:
            raise RuntimeError('fog diagnostics require the task45 engine')
        public_ids = {z['id'] for z in obs['zombies']}
        self.hidden_rows = set()
        for zombie in raw['zombies']:
            identity = zombie['id']
            row = zombie['row']
            col = max(0, min(8, int((zombie['x'] - 40) // 80)))
            hidden = 0 <= row < 6 and obs['fog_amount'][row][col] > 0
            self.first_seen.setdefault(identity, dict(tick=obs['tick'], in_fog=hidden,
                                                       public=identity in public_ids))
            if hidden:
                self.seen_hidden.add(identity)
                self.hidden_rows.add(row)
        if raw['fog_loss'] is not None:
            winner = dict(raw['fog_loss'])
            winner['first_seen'] = self.first_seen.get(winner['id'])
            winner['ever_in_fog'] = winner['id'] in self.seen_hidden
            winner['wave'] = obs['wave']
            winner['tick'] = obs['tick']
            self.loss = winner

    def compare(self, obs, student, teacher, searched):
        # The student's predicted heads are type / packet / cell; wait duration has no head.
        key = lambda a: tuple(a.get(k) for k in ('type', 'packet', 'row', 'col'))
        match = key(student) == key(teacher)
        self.agreement['decisions'] += 1
        self.agreement['matches'] += match
        self.agreement['searched'] += searched
        self.agreement['searched_matches'] += bool(searched and match)
        if not match:
            self.agreement['overrides'] += 1
            rows = {a['row'] for a in (student, teacher) if 'row' in a}
            self.agreement['overrides_fog_row'] += any(any(obs['fog_amount'][r]) for r in rows)
            self.agreement['overrides_hidden_enemy_row'] += bool(rows & self.hidden_rows)
            self.agreement['overrides_fog_target_cell'] += any(
                obs['fog_amount'][a['row']][a['col']] > 0
                for a in (student, teacher) if 'row' in a and 'col' in a)

    def planted(self, kind, action):
        if kind in (22, 25, 27):
            self.plantings[kind, action['row'], action['col']] += 1

    def result(self):
        return dict(loss=self.loss, teacher_agreement=dict(self.agreement), opening_plant_positions=[
            dict(type=kind, row=row, col=col, count=count)
            for (kind, row, col), count in sorted(self.plantings.items())])
