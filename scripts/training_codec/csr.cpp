#include <cstdint>
#include <cstring>
#include <vector>
#include <zlib.h>

// One call per batch; decode each distinct 32-row block once, preserving float bits.
extern "C" int pvz_csr_gather(
    const uint8_t* bytes, uint64_t size, const int64_t* offsets,
    const uint32_t* raw_sizes, const int64_t* groups, int64_t ngroups,
    const int64_t* inverse, const int64_t* selected, int64_t nselected,
    const uint16_t* counts, int64_t nrows, int64_t cells,
    int index_width, int value_width, int64_t* destinations, uint8_t* values) {
    if ((index_width != 1 && index_width != 2 && index_width != 4)
        || (value_width != 2 && value_width != 4)) return 1;
    std::vector<std::vector<uint8_t>> decoded;
    try {
        decoded.resize(ngroups);
        for (int64_t i = 0; i < ngroups; ++i) {
            const auto group = groups[i];
            if (group < 0 || group >= (nrows + 31) / 32) return 2;
            const auto lo = offsets[group], hi = offsets[group + 1];
            if (lo < 0 || hi < lo || uint64_t(hi) > size) return 2;
            auto& raw = decoded[i];
            raw.resize(raw_sizes[group]);
            uLongf length = raw.size();
            if (uncompress(raw.data(), &length, bytes + lo, hi - lo) != Z_OK
                || length != raw.size()) return 3;
        }
        int64_t cursor = 0;
        for (int64_t i = 0; i < nselected; ++i) {
            const auto row = selected[i], group = inverse[i];
            if (row < 0 || row >= nrows || group < 0 || group >= ngroups) return 2;
            const auto block = row / 32;
            if (groups[group] != block) return 2;
            const int rows = int(nrows - block * 32 < 32 ? nrows - block * 32 : 32);
            const auto& raw = decoded[group];
            const size_t header = (rows + 1) * sizeof(int64_t);
            if (raw.size() < header) return 3;
            int64_t start, stop, nnz;
            std::memcpy(&start, raw.data() + (row % 32) * 8, 8);
            std::memcpy(&stop, raw.data() + (row % 32 + 1) * 8, 8);
            std::memcpy(&nnz, raw.data() + rows * 8, 8);
            if (start < 0 || stop < start || stop > nnz || stop - start != counts[row]
                || header + uint64_t(nnz) * (index_width + value_width) != raw.size()) return 3;
            const auto* indices = raw.data() + header + start * index_width;
            const auto* source = raw.data() + header + nnz * index_width + start * value_width;
            for (int64_t j = 0; j < stop - start; ++j) {
                uint32_t cell = 0;
                std::memcpy(&cell, indices + j * index_width, index_width);
                if (cell >= uint64_t(cells)) return 3;
                destinations[cursor + j] = i * cells + cell;
            }
            std::memcpy(values + cursor * value_width, source, (stop - start) * value_width);
            cursor += stop - start;
        }
    } catch (...) { return 4; }
    return 0;
}
