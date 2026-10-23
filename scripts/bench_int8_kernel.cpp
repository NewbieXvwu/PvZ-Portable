// Isolated task35 experiment: does AVX-VNNI int8 beat the AVX2 float SGEMM that the
// student network uses today, on the real convolution shapes?  Never linked into the
// game.  Same fixture as bench_conv_kernels.cpp (scripts/bench_conv_kernels.py writes it),
// same im2col, same sparse-channel selection, so only the matrix kernel differs.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <immintrin.h>
#include <cblas.h>

struct Case {
    int layer, width, count;
    std::vector<float> matrix, columns, bias;   // matrix[width][count], columns[count][60]
    // int8 form: activations quantised to u8 and stored cell-major (aq[cell][k], k padded to 4),
    // weights quantised to s8 per output channel and packed as w4[target][k/4] (4 bytes).
    std::vector<uint8_t> aq;
    std::vector<int8_t> wq;
    std::vector<float> scale;                  // scale[target] = act_scale * weight_scale[target]
    int padded = 0;
};

template<class T> void Read(std::istream& input, T* data, size_t count) {
    if (!input.read(reinterpret_cast<char*>(data), sizeof(T) * count))
        throw std::runtime_error("truncated kernel fixture");
}

double CpuTime() {
    timespec value{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

__attribute__((noinline)) void Blas(const Case& item, float* output) {
    for (int row = 0; row < item.width; ++row)
        std::fill_n(output + row * 60, 60, item.bias[row]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        item.width, 60, item.count, 1, item.matrix.data(), item.count,
        item.columns.data(), 60, 1, output, 60);
}

// Quantise once, outside the timed loop: activations per case (the whole tensor shares one
// scale, as a per-layer calibration would give), weights per output channel.
void Quantise(Case& item) {
    const int padded = (item.count + 3) / 4 * 4;
    item.padded = padded;
    float actMax = 0;
    for (float value : item.columns) actMax = std::max(actMax, std::fabs(value));
    const float actScale = actMax > 0 ? actMax / 255.0f : 1.0f;
    // Activations are stored cell-major so the four bytes a vpdpbusd needs for one cell are
    // contiguous: aq[cell * padded + k].
    item.aq.assign(static_cast<size_t>(60) * padded, 0);
    for (int k = 0; k < item.count; ++k)
        for (int cell = 0; cell < 60; ++cell) {
            const float value = item.columns[static_cast<size_t>(k) * 60 + cell];
            item.aq[static_cast<size_t>(cell) * padded + k] =
                static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::round(value / actScale))));
        }
    // Weights are packed once for the whole model: the lane axis is the output channel, so a
    // 32-byte tile holds eight output channels x four k values, contiguous.
    const int blocks = padded / 4;
    item.wq.assign(static_cast<size_t>(blocks) * (item.width / 8) * 32, 0);
    item.scale.assign(item.width, 1.0f);
    for (int target = 0; target < item.width; ++target) {
        float weightMax = 0;
        for (int k = 0; k < item.count; ++k)
            weightMax = std::max(weightMax, std::fabs(item.matrix[static_cast<size_t>(target) * item.count + k]));
        const float weightScale = weightMax > 0 ? weightMax / 127.0f : 1.0f;
        item.scale[target] = actScale * weightScale;
        const int lane = target % 8;
        for (int k = 0; k < item.count; ++k) {
            const int8_t value = static_cast<int8_t>(std::min(127.0f,
                std::max(-127.0f, std::round(item.matrix[static_cast<size_t>(target) * item.count + k] / weightScale))));
            item.wq[static_cast<size_t>((k / 4) * (item.width / 8) + target / 8) * 32 + lane * 4 + k % 4] = value;
        }
    }
}

// 8 cells per accumulator; four k values per vpdpbusd.  The activation panel is already
// cell-major, so one 32-byte load covers 8 cells x 4 k values, and the weight bytes for
// those four k values are broadcast to every cell.
// One cell at a time, eight output channels per accumulator and eight accumulators in flight,
// so the five-cycle vpdpbusd latency never serialises the k loop.  The activation operand is a
// broadcast of four k values of one cell; the weight operand is a pre-packed 32-byte tile.
__attribute__((noinline)) void Int8(const Case& item, float* output) {
    const int padded = item.padded;
    const int blocks = padded / 4;
    const int width = item.width;
    alignas(32) int32_t totals[8][8];
    for (int cell = 0; cell < 60; ++cell) {
        const uint8_t* row = item.aq.data() + static_cast<size_t>(cell) * padded;
        __m256i accum[8];
        for (int target = 0; target < 8; ++target) accum[target] = _mm256_setzero_si256();
        for (int block = 0; block < blocks; ++block) {
            int32_t activations;
            std::memcpy(&activations, row + block * 4, 4);
            const __m256i values = _mm256_set1_epi32(activations);
            for (int target = 0; target < width / 8; ++target) {
                const auto* tile = reinterpret_cast<const __m256i*>(item.wq.data() +
                    (static_cast<size_t>(block) * (width / 8) + target) * 32);
                accum[target] = _mm256_dpbusd_epi32(accum[target], values, _mm256_loadu_si256(tile));
            }
        }
        for (int target = 0; target < width / 8; ++target)
            _mm256_store_si256(reinterpret_cast<__m256i*>(totals[target]), accum[target]);
        for (int target = 0; target < width; ++target) {
            const int channel = target;
            output[static_cast<size_t>(channel) * 60 + cell] = std::max(0.0f,
                static_cast<float>(totals[target / 8][target % 8]) * item.scale[channel] + item.bias[channel]);
        }
    }
}

// The exact float result of the *quantised* operands.  If the int8 kernel is right, this and
// the kernel agree to float rounding; the gap to Blas() is then purely quantisation error.
__attribute__((noinline)) void QuantRef(const Case& item, float* output) {
    const int padded = item.padded;
    const int width = item.width;
    for (int target = 0; target < width; ++target)
        for (int cell = 0; cell < 60; ++cell) {
            double total = 0;
            for (int k = 0; k < item.count; ++k) {
                const size_t weight = (static_cast<size_t>(k / 4) * (width / 8) + target / 8) * 32
                    + (target % 8) * 4 + k % 4;
                total += static_cast<double>(item.aq[static_cast<size_t>(cell) * padded + k]) *
                         static_cast<double>(item.wq[weight]);
            }
            output[target * 60 + cell] = std::max(0.0f, static_cast<float>(total) * item.scale[target] + item.bias[target]);
        }
}

int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("usage: bench-int8-kernel /absolute/fixture.bin");
    std::ifstream input(argv[1], std::ios::binary);
    std::array<char, 8> magic{};
    Read(input, magic.data(), magic.size());
    if (std::string(magic.data(), 8) != "PVZKERN1") throw std::runtime_error("bad fixture");
    uint32_t cases = 0;
    Read(input, &cases, 1);
    std::vector<Case> items;
    for (uint32_t index = 0; index < cases; ++index) {
        std::array<int32_t, 5> shape;
        Read(input, shape.data(), shape.size());
        const auto [channels, width, rows, cols, sparse] = shape;
        if (width != 64 || channels < 1 || channels > 512 ||
                !((rows == 3 && cols == 3) || (rows == 1 && cols == 5)))
            throw std::runtime_error("unsupported fixture shape");
        std::vector<float> board(channels * 60), weights(width * channels * rows * cols), bias(width);
        Read(input, board.data(), board.size());
        Read(input, weights.data(), weights.size());
        Read(input, bias.data(), bias.size());
        std::vector<int> active;
        for (int channel = 0; channel < channels; ++channel)
            if (!sparse || std::any_of(board.begin() + channel * 60, board.begin() + (channel + 1) * 60,
                    [](float value) { return value != 0; })) active.push_back(channel);
        Case item{static_cast<int>(index % 3), width, static_cast<int>(active.size()) * rows * cols};
        item.bias = bias;
        item.columns.assign(static_cast<size_t>(item.count) * 60, 0);
        item.matrix.assign(static_cast<size_t>(width) * item.count, 0);
        int feature = 0;
        for (int channel : active)
            for (int kr = 0; kr < rows; ++kr)
                for (int kc = 0; kc < cols; ++kc, ++feature) {
                    for (int row = 0; row < 6; ++row)
                        for (int col = 0; col < 10; ++col) {
                            const int y = row + kr - rows / 2, x = col + kc - cols / 2;
                            if (y >= 0 && y < 6 && x >= 0 && x < 10)
                                item.columns[static_cast<size_t>(feature) * 60 + row * 10 + col] =
                                    board[channel * 60 + y * 10 + x];
                        }
                    for (int target = 0; target < width; ++target)
                        item.matrix[static_cast<size_t>(target) * item.count + feature] =
                            weights[((target * channels + channel) * rows + kr) * cols + kc];
                }
        if (!item.count) throw std::runtime_error("empty fixture");
        Quantise(item);
        items.push_back(std::move(item));
    }
    std::vector<float> blas(64 * 60), int8(64 * 60), reference(64 * 60);
    std::array<double, 3> error{};
    std::array<double, 3> relative{};
    std::array<double, 3> squared{};
    std::array<double, 3> kernelGap{};
    std::array<double, 3> magnitude{};
    std::array<int, 3> counts{};
    for (const Case& item : items) {
        Blas(item, blas.data()); Int8(item, int8.data()); QuantRef(item, reference.data());
        // Convolve applies ReLU after the matrix product, so the float reference gets the same
        // clamp before the comparison; neither timed arm includes it.
        double peak = 0;
        for (int i = 0; i < 60 * 64; ++i) peak = std::max(peak, static_cast<double>(std::max(0.0f, blas[i])));
        for (int i = 0; i < 60 * 64; ++i) {
            const double difference = std::fabs(std::max(0.0f, blas[i]) - int8[i]);
            error[item.layer] = std::max(error[item.layer], difference);
            squared[item.layer] += difference * difference;
            relative[item.layer] = std::max(relative[item.layer], peak > 0 ? difference / peak : 0.0);
            kernelGap[item.layer] = std::max(kernelGap[item.layer],
                static_cast<double>(std::fabs(reference[i] - int8[i])));
            magnitude[item.layer] += std::max(0.0f, blas[i]);
        }
        ++counts[item.layer];
    }
    constexpr int repeats = 30;
    std::cout << "{\"repeats\":" << repeats << ",\"scope\":\"matrix kernel only; shared im2col and quantisation excluded\",\"rounds\":[";
    for (int round = 0; round < 3; ++round) {
        if (round) std::cout << ',';
        std::array<std::array<double, 3>, 2> seconds{};
        for (int order = 0; order < 2; ++order) {
            const int arm = (round + order) % 2;
            for (int layer = 0; layer < 3; ++layer) {
                const double start = CpuTime();
                for (int repeat = 0; repeat < repeats; ++repeat)
                    for (const Case& item : items) if (item.layer == layer) {
                        if (arm == 0) Blas(item, blas.data()); else Int8(item, int8.data());
                    }
                seconds[arm][layer] = CpuTime() - start;
            }
        }
        std::cout << '[';
        for (int layer = 0; layer < 3; ++layer) {
            if (layer) std::cout << ',';
            std::cout << "{\"layer\":" << layer << ",\"cases\":" << counts[layer]
                << ",\"blas_us\":" << seconds[0][layer] * 1e6 / (repeats * counts[layer])
                << ",\"int8_us\":" << seconds[1][layer] * 1e6 / (repeats * counts[layer])
                << ",\"int8_over_blas\":" << seconds[1][layer] / seconds[0][layer]
                << ",\"max_abs_error\":" << error[layer]
                << ",\"max_relative_error\":" << relative[layer] << ",\"rms_error\":" << std::sqrt(squared[layer] / (counts[layer] * 60.0 * 64.0))
                << ",\"kernel_vs_quant_ref\":" << kernelGap[layer]
                << ",\"mean_output\":" << magnitude[layer] / (counts[layer] * 60.0 * 64.0) << '}';
        }
        std::cout << ']';
    }
    std::cout << "],\"checksum\":" << blas[0] + int8[0] << "}\n";
}
