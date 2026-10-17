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
// Host-only code: the device drivers include this, the kernels never do.
#ifndef PVZ_PAIR_LIST_H
#define PVZ_PAIR_LIST_H

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace pvzhost
{

inline bool ReadPairList(const std::string& path, std::vector<int32_t>& rootIndex,
	std::vector<int32_t>& futureSeed, std::string& error)
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
		const int parsed = std::sscanf(line, "%d %d", &root, &seed);
		if (parsed == 0)
		{
			// A comment or a blank line.  Anything else that does not start with a digit is a mistake, and
			// saying so is the whole point: a silently dropped line shifts every later lane's pairing.
			char first = ' ';
			for (const char* cursor = line; *cursor != '\0'; ++cursor)
				if (*cursor != ' ' && *cursor != '\t' && *cursor != '\r' && *cursor != '\n')
				{
					first = *cursor;
					break;
				}
			if (first == '#' || first == ' ') continue;
			error = "line " + std::to_string(line_number) + " of " + path + " is not a pair";
			ok = false;
			break;
		}
		if (parsed != 2)
		{
			error = "line " + std::to_string(line_number) + " of " + path + " is not two integers";
			ok = false;
			break;
		}
		if (root < 0)
		{
			error = "negative root index on line " + std::to_string(line_number) + " of " + path;
			ok = false;
			break;
		}
		rootIndex.push_back(root);
		futureSeed.push_back(seed);
	}
	std::fclose(stream);
	if (ok && rootIndex.empty()) error = path + " holds no pairs";
	return ok && !rootIndex.empty();
}

}  // namespace pvzhost

#endif
