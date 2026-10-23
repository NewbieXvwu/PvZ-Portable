// Isolated task35 experiment, never linked into the game. All paths share the
// same sparse channel selection and im2col input; compare only matrix kernels.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <ctime>
#include <immintrin.h>
#include <cblas.h>

struct Case {
    int layer, width, count;
    std::vector<float> matrix, columns, bias;
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

// Four output channels x four cells. Keep the exact float accumulation order,
// with explicit SSE instructions rather than relying on compiler recognition.
__attribute__((noinline)) void Simd(const Case& item, float* output) {
    for (int target = 0; target < item.width; target += 4)
        for (int cell = 0; cell < 60; cell += 4) {
            __m128 accum[4];
            for (int row = 0; row < 4; ++row)
                accum[row] = _mm_set1_ps(item.bias[target + row]);
            for (int k = 0; k < item.count; ++k) {
                const __m128 values = _mm_loadu_ps(item.columns.data() + k * 60 + cell);
                for (int row = 0; row < 4; ++row)
                    accum[row] = _mm_add_ps(accum[row], _mm_mul_ps(values,
                        _mm_set1_ps(item.matrix[(target + row) * item.count + k])));
            }
            for (int row = 0; row < 4; ++row)
                _mm_storeu_ps(output + (target + row) * 60 + cell, accum[row]);
        }
}

__attribute__((noinline)) void Blas(const Case& item, float* output) {
    for (int row = 0; row < item.width; ++row)
        std::fill_n(output + row * 60, 60, item.bias[row]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        item.width, 60, item.count, 1, item.matrix.data(), item.count,
        item.columns.data(), 60, 1, output, 60);
}

int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("usage: bench-conv-kernels /absolute/fixture.bin");
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
        item.columns.resize(item.count * 60, 0);
        item.matrix.resize(width * item.count);
        int feature = 0;
        for (int channel : active)
            for (int kr = 0; kr < rows; ++kr)
                for (int kc = 0; kc < cols; ++kc, ++feature) {
                    for (int row = 0; row < 6; ++row)
                        for (int col = 0; col < 10; ++col) {
                            const int y = row + kr - rows / 2, x = col + kc - cols / 2;
                            if (y >= 0 && y < 6 && x >= 0 && x < 10)
                                item.columns[feature * 60 + row * 10 + col] = board[channel * 60 + y * 10 + x];
                        }
                    for (int target = 0; target < width; ++target)
                        item.matrix[target * item.count + feature] =
                            weights[((target * channels + channel) * rows + kr) * cols + kc];
                }
        if (!item.count) throw std::runtime_error("empty fixture");
        items.push_back(std::move(item));
    }
    std::vector<float> blas(64 * 60), simd(64 * 60);
    std::array<double, 3> error{};
    std::array<int, 3> counts{};
    for (const Case& item : items) {
        Blas(item, blas.data()); Simd(item, simd.data());
        for (size_t i = 0; i < blas.size(); ++i)
            error[item.layer] = std::max(error[item.layer], static_cast<double>(std::abs(blas[i] - simd[i])));
        ++counts[item.layer];
    }
    constexpr int repeats = 30;
    std::cout << "{\"repeats\":" << repeats << ",\"scope\":\"matrix kernel only; shared im2col excluded\",\"rounds\":[";
    for (int round = 0; round < 3; ++round) {
        if (round) std::cout << ',';
        std::array<std::array<double, 3>, 2> seconds{};
        for (int order = 0; order < 2; ++order) {
            const int arm = (round + order) % 2;
            for (int layer = 0; layer < 3; ++layer) {
                const double start = CpuTime();
                for (int repeat = 0; repeat < repeats; ++repeat)
                    for (const Case& item : items) if (item.layer == layer) {
                        if (arm == 0) Blas(item, blas.data()); else Simd(item, simd.data());
                    }
                seconds[arm][layer] = CpuTime() - start;
            }
        }
        std::cout << '[';
        for (int layer = 0; layer < 3; ++layer) {
            if (layer) std::cout << ',';
            std::cout << "{\"layer\":" << layer << ",\"cases\":" << counts[layer]
                << ",\"blas_us\":" << seconds[0][layer] * 1e6 / (repeats * counts[layer])
                << ",\"simd_us\":" << seconds[1][layer] * 1e6 / (repeats * counts[layer])
                << ",\"simd_over_blas\":" << seconds[1][layer] / seconds[0][layer]
                << ",\"max_abs_error\":" << error[layer] << '}';
        }
        std::cout << ']';
    }
    std::cout << "],\"checksum\":" << blas[0] + simd[0] << "}\n";
}
