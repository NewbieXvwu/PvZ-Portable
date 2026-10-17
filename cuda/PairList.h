// The explicit (root, future) workload both rollout drivers accept with --pairs.
//
// A benchmark can derive the pair from the lane index (cuda/RolloutCuda.cu's RolloutOf), which is what
// the throughput gate uses.  Comparing a GPU continuation against a CPU reference continuation is a
// different job: it needs *the same* root and *the same* future on both sides, and the corpus's pairing
// (python/pvz_env.py's `seed * 1000 + future`, written into corpus.jsonl by scripts/cuda_root_corpus.py)
// is not a formula either driver can re-derive.  So the pairs come from the file.
//
// Format: one pair per line, `<rootIndex> <futureSeed>`, whitespace separated; blank lines and lines
// starting with '#' are ignored.  scripts/cuda_leaf_compare.py --pairs-out writes it from corpus.jsonl.
//
// A line may carry the teacher's action after those two numbers: `<root> <future> <kind> <packet> <row>
// <col> [ticks]`, with kind 1 = plant, 2 = wait (ticks defaults to 60), 0 = let the student decide.  That
// is the search's workload: one root continued once per candidate action, each candidate over several
// futures.  A line with only the two numbers keeps meaning "the student decides", so a phase-1 comparison
// file written before this existed reads exactly as it did then.
//
// Host-only code: the device drivers include this, the kernels never do.
#ifndef PVZ_PAIR_LIST_H
#define PVZ_PAIR_LIST_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pvzhost
{

// The per-line action, in the five int32 a lane's RolloutBegin reads (cuda/Rollout.inc).
struct FirstAction
{
	int32_t kind;
	int32_t packet;
	int32_t row;
	int32_t col;
	int32_t ticks;
};

constexpr int32_t kFirstActionWords = 5;

inline void FlattenFirstActions(const std::vector<FirstAction>& actions, std::vector<int32_t>& flat)
{
	flat.clear();
	flat.reserve(actions.size() * kFirstActionWords);
	for (const FirstAction& action : actions)
	{
		flat.push_back(action.kind);
		flat.push_back(action.packet);
		flat.push_back(action.row);
		flat.push_back(action.col);
		flat.push_back(action.ticks);
	}
}

inline bool ReadPairList(const std::string& path, std::vector<int32_t>& rootIndex,
	std::vector<int32_t>& futureSeed, std::vector<FirstAction>& first, std::string& error)
{
	std::FILE* stream = std::fopen(path.c_str(), "r");
	if (!stream)
	{
		error = "cannot open " + path;
		return false;
	}
	char line[256];
	int32_t line_number = 0;
	bool ok = true;
	while (std::fgets(line, sizeof(line), stream))
	{
		++line_number;
		int32_t root = 0, seed = 0;
		FirstAction action{0, -1, 0, 0, 60};
		const int parsed = std::sscanf(line, "%d %d %d %d %d %d %d", &root, &seed, &action.kind,
			&action.packet, &action.row, &action.col, &action.ticks);
		if (parsed == 0)
		{
			// A comment or a blank line.  Anything else that does not start with a digit is a mistake, and
			// saying so is the whole point: a silently dropped line shifts every later lane's pairing.
			char lead = ' ';
			for (const char* cursor = line; *cursor != '\0'; ++cursor)
				if (*cursor != ' ' && *cursor != '\t' && *cursor != '\r' && *cursor != '\n')
				{
					lead = *cursor;
					break;
				}
			if (lead == '#' || lead == ' ') continue;
			error = "line " + std::to_string(line_number) + " of " + path + " is not a pair";
			ok = false;
			break;
		}
		// Two numbers is the phase-1 form (the student decides); six or seven add the teacher's action.
		// Anything in between is a half-written line, and half a line is how a lane ends up planting a
		// candidate that belongs to its neighbour.
		if (parsed != 2 && parsed != 6 && parsed != 7)
		{
			error = "line " + std::to_string(line_number) + " of " + path
				+ " is not 2, 6 or 7 integers";
			ok = false;
			break;
		}
		if (action.kind < 0 || action.kind > 2)
		{
			error = "line " + std::to_string(line_number) + " of " + path
				+ ": first-action kind must be 0 (student), 1 (plant) or 2 (wait)";
			ok = false;
			break;
		}
		if (action.kind == 1 && action.packet < 0)
		{
			error = "line " + std::to_string(line_number) + " of " + path
				+ ": a plant action needs a packet index";
			ok = false;
			break;
		}
		// A wait has no packet: whatever number happens to sit in that column must not reach ActionPlant.
		if (action.kind == 2) action.packet = -1;
		if (root < 0)
		{
			error = "negative root index on line " + std::to_string(line_number) + " of " + path;
			ok = false;
			break;
		}
		rootIndex.push_back(root);
		futureSeed.push_back(seed);
		first.push_back(action);
	}
	std::fclose(stream);
	if (ok && rootIndex.empty()) error = path + " holds no pairs";
	return ok && !rootIndex.empty();
}

}  // namespace pvzhost

#endif
