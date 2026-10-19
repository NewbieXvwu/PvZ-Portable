"""Experimental CPU verification budget; unused unless a caller explicitly opts in.

Three new futures per leg are inspected first.  A gap far from the override margin
stops verification; ambiguous pairs use the original full budget and value batches.
This is a heuristic, not a statistical confidence bound.  Adoption requires the
same-point noise gate and paired teacher-strength gate.
"""


def cpu_pair(env, obs, root, candidates, indexes, totals, counts, args, decision_seed,
             samples, *, rollout, terminal_score, frame, values):
    initial = min(3, samples)
    leaves = {i: [] for i in indexes}

    def collect(start, end):
        for i in indexes:
            for future in range(start, end):
                leaves[i].append(rollout(env, obs, candidates[i], obs['tick'] + args.horizon,
                                        root, (decision_seed * 1000 + future) % 2**32))

    def means(n):
        result = []
        for i in indexes:
            fixed, frames = [], []
            for leaf in leaves[i]:
                score = terminal_score(leaf)
                if score is None:
                    frames.append(frame(leaf))
                else:
                    fixed.append(score)
            # If the budget expands, score all nonterminal leaves together again.
            # Splitting that batch and adding partial sums would change the full-budget
            # reference's numerical order (including terminal-first accumulation).
            added = fixed + (values(frames) if frames else [])
            result.append((totals[i] * counts[i] + sum(added)) / (counts[i] + n))
        return result

    collect(args.samples, args.samples + initial)
    first = means(initial)
    gap = first[1] - first[0]
    early = initial < samples and abs(gap - args.margin) >= args.adaptive_verify_gap
    if early or initial == samples:
        return {'means': first, 'samples': initial, 'checkpoint_gap': gap, 'early_stop': early}
    collect(args.samples + initial, args.samples + samples)
    return {'means': means(samples), 'samples': samples, 'checkpoint_gap': gap, 'early_stop': False}
