"""Audit the search teacher's value model and candidate generator on development seeds.

Two risks the teacher's design cannot rule out on its own, both of which are about
*where the search looks* rather than how cleverly it looks there:

* ``SearchValueModel`` is trained on the states the teacher actually visited, but it
  is queried on the counterfactual one-step children of every root candidate -- the
  states a strong teacher rarely walks into.  Mean squared error on its own training
  set cannot detect that mismatch, because every state in a won episode carries the
  same label sign.  What matters instead is whether the model *orders* siblings the
  way a much deeper search does, which is what :func:`sibling_ranking` measures.
* ``CandidateGenerator`` prunes before the search ever sees an action, and no amount
  of extra simulation budget recovers an action it never proposed.
  :func:`candidate_recall` measures the loss at both stages separately -- generation
  and screening -- against every legal action in the state.

Both are development-set only: ``read_seed_set`` is called with the ``development``
role, so pointing this at the frozen final-test set fails rather than quietly
burning it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from pvz_agent_model import WAIT_TICKS, configure_torch_threads, resolve_device
from pvz_common import ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION, sha256_file
from pvz_env import PvZEnv, training_task
from pvz_search import SearchAdvice, SearchTeacher
from pvz_search_candidates import action_key
from pvz_search_diagnostics import visible_state_key
from pvz_search_value import SearchValueModel, load_search_value
from pvz_seed_sets import DEFAULT_DEV_SEEDS, read_seed_set
from pvz_training_artifacts import task_signature

LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_ACTIONS = 2000
RECALL_AT = (1, 3, 5)


def leaf_score(observation: dict[str, Any], value_model: SearchValueModel | None) -> float:
    """Score a leaf exactly the way ``SearchTeacher._leaf_value`` would.

    Sharing the definition matters: if the audit scored leaves differently from the
    search, the ranking it measures would not be the ranking the search acts on.
    """
    if observation["terminal"]:
        return 1.0 if observation["result"] == 1 else -1.0
    if value_model is not None:
        return max(-1.0, min(1.0, float(value_model.predict(observation))))
    return SearchTeacher._bootstrap_leaf_value(observation)


def pairwise_agreement(reference: list[float], candidate: list[float]) -> tuple[int, int]:
    """``(agreements, comparable_pairs)`` over strictly ordered pairs.

    A pair where either side is exactly tied carries no ordering information, so it
    is excluded rather than scored as agreement or disagreement.  Counting ties as
    agreement would flatter a value model that simply outputs a constant.
    """
    if len(reference) != len(candidate):
        raise ValueError("pairwise agreement needs two equally long score lists")
    agree = 0
    pairs = 0
    for left in range(len(reference)):
        for right in range(left + 1, len(reference)):
            expected = reference[left] - reference[right]
            actual = candidate[left] - candidate[right]
            if expected == 0.0 or actual == 0.0:
                continue
            pairs += 1
            agree += (expected > 0.0) == (actual > 0.0)
    return agree, pairs


def _scored_children(
    teacher: SearchTeacher,
    value_model: SearchValueModel | None,
    observation: dict[str, Any],
    actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[float], int]:
    """One-step-score *actions* from *observation*.

    Returns ``(actions, scores, terminal_actions)``.  An action that ends the level
    has no child to score -- the search itself scores those by the terminal outcome,
    not by the value model -- so it is dropped from the ranking and counted instead.
    """
    scored_actions: list[dict[str, Any]] = []
    scores: list[float] = []
    terminal = 0
    for action, child in teacher.one_step_children(observation, actions):
        if child is None:
            terminal += 1
            continue
        scored_actions.append(action)
        scores.append(leaf_score(child, value_model))
    return scored_actions, scores, terminal


def sibling_ranking(
    teacher: SearchTeacher,
    deep_teacher: SearchTeacher,
    value_model: SearchValueModel | None,
    observation: dict[str, Any],
    deep_advice: SearchAdvice | None = None,
) -> dict[str, Any] | None:
    """Order the screened candidates by the value model and by a deep search.

    The deep search's own per-candidate values are the reference.  They come from the
    same state and the same candidate set, so the comparison isolates one question:
    does scoring the one-step child with ``SearchValueModel`` produce the same
    ordering the search reaches after extending every line?
    """
    deep_advice = deep_teacher.advice(observation) if deep_advice is None else deep_advice
    if len(deep_advice.candidates) < 2:
        return None
    deep_by_key = {action_key(action): value for action, value in deep_advice.candidates}
    actions = [action for action, _ in deep_advice.candidates]
    scored_actions, shallow, terminal = _scored_children(teacher, value_model, observation, actions)
    reference = [deep_by_key[action_key(action)] for action in scored_actions]
    if len(reference) < 2:
        return None
    agree, pairs = pairwise_agreement(reference, shallow)
    shallow_best = max(range(len(shallow)), key=lambda index: shallow[index])
    deep_best = max(range(len(reference)), key=lambda index: reference[index])
    return {
        "candidates": len(actions),
        "scored": len(scored_actions),
        "terminal_actions": terminal,
        "pairwise_agreements": agree,
        "pairwise_pairs": pairs,
        "pairwise_accuracy": agree / pairs if pairs else None,
        "top1_agrees": shallow_best == deep_best,
        "shallow_best": scored_actions[shallow_best],
        "deep_best": scored_actions[deep_best],
        "deep_best_value": reference[deep_best],
        "shallow_best_value": shallow[shallow_best],
        "deep_spread": max(reference) - min(reference),
    }


def budget_consistency(shallow: SearchAdvice, deep: SearchAdvice) -> dict[str, Any]:
    """Compare a default-budget decision with a much deeper search of the same state.

    This is the saturation test the configuration sweep needs: if the cheap
    configuration ranks the candidates the way an expensive one does, the extra
    budget buys nothing, and the budget knob is settled.
    """
    shallow_by_key = {action_key(action): value for action, value in shallow.candidates}
    deep_by_key = {action_key(action): value for action, value in deep.candidates}
    shared = [key for key in deep_by_key if key in shallow_by_key]
    agree, pairs = pairwise_agreement([deep_by_key[key] for key in shared],
                                      [shallow_by_key[key] for key in shared])
    return {
        "shared_candidates": len(shared),
        "pairwise_agreements": agree,
        "pairwise_pairs": pairs,
        "pairwise_accuracy": agree / pairs if pairs else None,
        "top1_agrees": action_key(shallow.action) == action_key(deep.action),
        "shallow_top1": shallow.action,
        "deep_top1": deep.action,
    }


def candidate_recall(
    teacher: SearchTeacher,
    value_model: SearchValueModel | None,
    observation: dict[str, Any],
    advice: SearchAdvice,
    recall_at: tuple[int, ...] = RECALL_AT,
) -> dict[str, Any]:
    """Measure how much of the legal action set survives generation and screening.

    Three nested sets, scored by the same one-step evaluator so the comparison is
    apples to apples:

    * ``everything`` -- every legal plant placement, every legal shovel and every
      wait duration;
    * ``generated`` -- what ``CandidateGenerator`` hands back for the request the
      search itself would make;
    * ``screened`` -- what ``_screen_root_states`` keeps for the deep search.

    The gap between ``everything`` and ``generated`` is generation loss, which no
    simulation budget can fix.  The gap between ``generated`` and ``screened`` is
    screening loss, which more budget could.
    """
    legal = observation["legal_actions"]
    everything = [{"type": "plant", **item} for item in legal["plants"]]
    everything.extend({"type": "shovel", "col": col, "row": row} for col, row in legal["shovels"])
    if legal.get("wait", True):
        everything.extend({"type": "wait", "ticks": ticks} for ticks in WAIT_TICKS)

    _, all_scores, _ = _scored_children(teacher, value_model, observation, everything)
    if not all_scores:
        return {"legal_actions": len(everything), "scored_actions": 0, "recall": {}}

    generated = teacher.candidate_generator.actions(
        observation, teacher.root_request_limit, True, teacher.horizon_ticks)
    screened_actions = [action for action, _ in advice.candidates]

    # The full legal set's ranking, so a candidate set can be scored by which of the
    # best actions it actually contains.
    scored_all = sorted(zip(everything, all_scores), key=lambda item: item[1], reverse=True)
    best = [action_key(action) for action, _ in scored_all]
    top_positions = {key: index for index, key in enumerate(best)}

    report: dict[str, Any] = {
        "legal_actions": len(everything),
        "scored_actions": len(all_scores),
        "generated_actions": len(generated),
        "screened_actions": len(screened_actions),
        "screened_simulations": advice.screening_simulations,
        "best_action": scored_all[0][0],
        "best_action_score": scored_all[0][1],
        "recall": {},
    }
    for name, actions in (("generated", generated), ("screened", screened_actions)):
        positions = [top_positions[action_key(action)] for action in actions
                     if action_key(action) in top_positions]
        keys = {action_key(action) for action in actions}
        entry: dict[str, Any] = {
            "contains_best": best[0] in keys,
            "best_rank_inside": (min(positions) + 1) if positions else None,
        }
        for k in recall_at:
            top = set(best[:k])
            entry[f"recall@{k}"] = len(top & keys) / len(top) if top else None
        report["recall"][name] = entry
    return report


def audit_state(
    teacher: SearchTeacher,
    deep_teacher: SearchTeacher,
    value_model: SearchValueModel | None,
    observation: dict[str, Any],
    advice: SearchAdvice,
    modes: frozenset[str],
) -> dict[str, Any]:
    """Run the requested audits against one already-searched state."""
    record: dict[str, Any] = {
        "visible_state_key": visible_state_key(observation),
        "tick": observation["tick"],
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "sun": observation["sun"],
        "progress": observation["wave"] / max(1, observation["wave_count"]),
        "chosen_action": advice.action,
        "simulation_count": advice.simulation_count,
        "effective_depth_budget": advice.effective_depth_budget,
    }
    if "budget" in modes:
        deep = deep_teacher.advice(observation)
        record["budget"] = budget_consistency(advice, deep)
    else:
        deep = None
    if "sibling" in modes:
        record["sibling"] = sibling_ranking(teacher, deep_teacher, value_model, observation, deep)
    if "recall" in modes:
        record["recall"] = candidate_recall(teacher, value_model, observation, advice)
    return record


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rate(records: list[bool]) -> float | None:
    return sum(records) / len(records) if records else None


def summarize(records: list[dict[str, Any]], modes: frozenset[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"states": len(records)}
    if "budget" in modes:
        entries = [record["budget"] for record in records if record.get("budget")]
        summary["budget_consistency"] = {
            "states": len(entries),
            "top1_agreement_rate": _rate([entry["top1_agrees"] for entry in entries]),
            "mean_pairwise_accuracy": _mean([entry["pairwise_accuracy"] for entry in entries
                                             if entry["pairwise_accuracy"] is not None]),
            "mean_shared_candidates": _mean([float(entry["shared_candidates"]) for entry in entries]),
        }
    if "sibling" in modes:
        entries = [record["sibling"] for record in records if record.get("sibling")]
        accuracies = [entry["pairwise_accuracy"] for entry in entries
                      if entry["pairwise_accuracy"] is not None]
        summary["sibling_ranking"] = {
            "states": len(entries),
            "mean_pairwise_accuracy": _mean(accuracies),
            "top1_agreement_rate": _rate([entry["top1_agrees"] for entry in entries]),
            "mean_scored_candidates": _mean([float(entry["scored"]) for entry in entries]),
            # A deep search that cannot separate its own candidates has no ordering to
            # reproduce, so the accuracy above is only meaningful where the spread is not ~0.
            "mean_deep_spread": _mean([entry["deep_spread"] for entry in entries]),
        }
    if "recall" in modes:
        entries = [record["recall"] for record in records if record.get("recall")]
        entries = [entry for entry in entries if entry["recall"]]
        summary["candidate_recall"] = {
            "states": len(entries),
            "generation_loss_rate": _rate([not entry["recall"]["generated"]["contains_best"]
                                           for entry in entries]),
            "screening_loss_rate": _rate([not entry["recall"]["screened"]["contains_best"]
                                          for entry in entries]),
        }
        for name in ("generated", "screened"):
            for k in RECALL_AT:
                summary["candidate_recall"][f"{name}_recall@{k}"] = _mean(
                    [entry["recall"][name][f"recall@{k}"] for entry in entries
                     if entry["recall"][name][f"recall@{k}"] is not None])
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--seeds", type=Path, default=DEFAULT_DEV_SEEDS)
    parser.add_argument("--search-value", type=Path,
                        help="SearchValueModel checkpoint; without it the audit scores "
                             "leaves with the bootstrap evaluator")
    parser.add_argument("--modes", default="budget,sibling,recall")
    parser.add_argument("--stride", type=int, default=8, help="audit every Nth decision")
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=lambda value: [int(item) for item in value.split(",") if item],
                        default=list(DECK))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument(
        "--threads", type=int, default=0,
        help="CPU thread count for torch; 0 selects the measured default (4). Thread count "
             "perturbs results only in the low order bits (<=6e-7 relative), far below the "
             "1e-5..1e-4 decision error budget, so it does not move audit verdicts.",
    )
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-horizon-ticks", type=int, default=900)
    parser.add_argument("--search-simulation-budget", type=int, default=256)
    parser.add_argument("--search-max-decisions", type=int, default=64)
    parser.add_argument("--deep-width", type=int, default=3)
    parser.add_argument("--deep-candidates", type=int, default=8)
    parser.add_argument("--deep-horizon-ticks", type=int, default=1800)
    parser.add_argument("--deep-simulation-budget", type=int, default=1024)
    parser.add_argument("--deep-max-decisions", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if args.stride < 1 or args.max_states < 1:
        parser.error("--stride and --max-states must be positive")
    modes = frozenset(item for item in args.modes.split(",") if item)
    unknown = modes - {"budget", "sibling", "recall"}
    if unknown:
        parser.error(f"unknown audit modes: {sorted(unknown)}")
    if not modes:
        parser.error("--modes must select at least one audit")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modes = frozenset(item for item in args.modes.split(",") if item)
    seeds = read_seed_set(args.seeds, args.level, "development")
    device = resolve_device(args.device)
    torch_threads = configure_torch_threads(args.threads)
    current_task_signature = task_signature(args.level, args.deck, args.zombie_count_multiplier,
                                            args.resource_dir)

    value_model = None
    value_metadata: dict[str, Any] = {"source": "bootstrap evaluator"}
    if args.search_value:
        value_model, checkpoint = load_search_value(args.search_value, device, current_task_signature)
        value_metadata = {
            "source": str(args.search_value.resolve()),
            "sha256": sha256_file(args.search_value),
            "bootstrap_seeds": checkpoint.get("bootstrap_seeds"),
            "refinement_seeds": checkpoint.get("refinement_seeds"),
        }

    records: list[dict[str, Any]] = []
    with PvZEnv(args.resource_dir, args.executable) as env:
        teacher = SearchTeacher(
            env,
            value_model=value_model,
            beam_width=args.search_width,
            candidate_limit=args.search_candidates,
            horizon_ticks=args.search_horizon_ticks,
            simulation_budget=args.search_simulation_budget,
            max_decisions=args.search_max_decisions,
        )
        deep_teacher = SearchTeacher(
            env,
            value_model=value_model,
            beam_width=args.deep_width,
            candidate_limit=args.deep_candidates,
            horizon_ticks=args.deep_horizon_ticks,
            simulation_budget=args.deep_simulation_budget,
            max_decisions=args.deep_max_decisions,
        )
        search_settings = {
            "beam_width": args.search_width,
            "candidate_limit": args.search_candidates,
            "horizon_ticks": args.search_horizon_ticks,
            "simulation_budget": args.search_simulation_budget,
            "max_decisions": args.search_max_decisions,
            "root_candidate_limit": teacher.root_candidate_limit,
            "root_request_limit": teacher.root_request_limit,
        }
        deep_settings = {
            "beam_width": args.deep_width,
            "candidate_limit": args.deep_candidates,
            "horizon_ticks": args.deep_horizon_ticks,
            "simulation_budget": args.deep_simulation_budget,
            "max_decisions": args.deep_max_decisions,
            "root_candidate_limit": deep_teacher.root_candidate_limit,
            "root_request_limit": deep_teacher.root_request_limit,
        }
        for seed in seeds:
            if len(records) >= args.max_states:
                break
            observation, _ = env.reset(
                deck=tuple(args.deck), task=training_task(seed, args.level, args.zombie_count_multiplier))
            decision = 0
            while not observation["terminal"] and decision < args.max_actions:
                advice = teacher.advice(observation)
                if decision % args.stride == 0 and len(records) < args.max_states:
                    record = audit_state(teacher, deep_teacher, value_model, observation, advice, modes)
                    record["seed"] = seed
                    record["decision_index"] = decision
                    records.append(record)
                observation, _, done, _, info = env.step(advice.action)
                if not info.get("ok"):
                    raise RuntimeError(f"search selected an illegal action on seed {seed}: {advice.action}")
                decision += 1
                if done:
                    break
            if not observation["terminal"]:
                raise RuntimeError(f"audit episode exceeded {args.max_actions} decisions on seed {seed}")
            print(f"seed {seed} states={len(records)}", flush=True)

    report = {
        "protocol_version": ENV_PROTOCOL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "task_version": TASK_VERSION,
        "evaluation_role": "development",
        "level": args.level,
        "deck": args.deck,
        "zombie_count_multiplier": args.zombie_count_multiplier,
        "resolved_device": str(device),
        "torch_threads": torch_threads,
        "seed_file": str(args.seeds.resolve()),
        "seed_file_sha256": sha256_file(args.seeds),
        "task_signature": current_task_signature,
        "modes": sorted(modes),
        "search_value": value_metadata,
        "search": search_settings,
        "deep_search": deep_settings,
        "summary": summarize(records, modes),
        "states": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
