"""Count actual student forwards in one search game using search_play's arguments."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
import search_play as search
from render_episode import load_training_task


def main():
    args = search.parser().parse_args()
    task = load_training_task(args.manifest, args.task_id)
    seed = int(args.seed_list.split(',')[0]) if args.seed_list else task['seeds'][args.seed_offset]
    exports, values = search._export_values(args.value, args.static_value)
    search._load_nets(values[0], args.rollout_policy, values[1])
    search._W.update(lean_rollouts=True, rollout_sim_skip=args.rollout_sim_skip, batch_waits=True)
    counts = {'root_total': 0, 'root_action': 0, 'continuation': 0}
    encode = search._W['fast'].encode
    fast_action = search._fast_action
    rollout = search._rollout_policy

    def counted_encode(*positional, **keywords):
        counts['root_total'] += 1
        return encode(*positional, **keywords)

    def counted_action(*positional, **keywords):
        before = counts['root_total']
        result = fast_action(*positional, **keywords)
        counts['root_action'] += counts['root_total'] - before
        return result

    def counted_rollout(env, *positional, **keywords):
        result = rollout(env, *positional, **keywords)
        counts['continuation'] += env._last_response['net_calls']
        return result

    search._W['fast'].encode = counted_encode
    search._fast_action = counted_action
    search._rollout_policy = counted_rollout
    try:
        result = search._play({'seed': seed, 'future_seed': None, 'arm': 'fast_search', 'args': args})
    finally:
        search._close_worker_env()
        exports.cleanup()
    counts['root_candidates'] = counts['root_total'] - counts['root_action']
    result.update(network_calls=counts, rollout_metrics=search._W.get('rollout_metrics'),
                  root_fraction=counts['root_total'] / (counts['root_total'] + counts['continuation']),
                  config={key: str(value) for key, value in vars(args).items()})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'network_calls.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
