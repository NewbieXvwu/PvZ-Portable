// task35 isolated first-layer experiment. No game process or production dispatch.
// Compare complete im2col+BLAS with direct nonzero-input accumulation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <immintrin.h>
#include <cblas.h>

template<class T> void Read(std::istream& in, T* out, size_t count) {
    if (!in.read(reinterpret_cast<char*>(out), sizeof(T) * count)) throw std::runtime_error("truncated fixture");
}
double Cpu() { timespec t{}; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

struct Case {
    int channels, width, count;
    std::vector<int> active;
    std::vector<float> board, weights, bias, packed, transposed;
};

void Blas(const Case& item, float* columns, float* out) {
    std::fill_n(columns, item.count * 60, 0);
    int feature = 0;
    for (int ch : item.active)
        for (int kr = 0; kr < 3; ++kr)
            for (int kc = 0; kc < 3; ++kc, ++feature) {
                const int first = std::max(0, 1 - kc), last = std::min(10, 11 - kc);
                for (int y = std::max(0, 1 - kr); y < std::min(6, 7 - kr); ++y) {
                    const float* src = item.board.data() + ch * 60 + (y + kr - 1) * 10 + first + kc - 1;
                    float* dst = columns + feature * 60 + y * 10 + first;
                    const int run = last - first;
                    if (run == 10) std::copy_n(src, 10, dst); else std::copy_n(src, 9, dst);
                }
            }
    for (int ch = 0; ch < item.width; ++ch) std::fill_n(out + ch * 60, 60, item.bias[ch]);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, item.width, 60, item.count,
                1, item.packed.data(), item.count, columns, 60, 1, out, 60);
}

__attribute__((target("avx2"), noinline))
void Sparse(const Case& item, float* spatial, float* out) {
    for (int cell = 0; cell < 60; ++cell) std::copy_n(item.bias.data(), item.width, spatial + cell * item.width);
    for (int ch : item.active)
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 10; ++x) {
                const float value = item.board[ch * 60 + y * 10 + x];
                if (value == 0) continue;
                const __m256 input = _mm256_set1_ps(value);
                for (int kr = std::max(0, y - 4); kr < std::min(3, y + 2); ++kr)
                    for (int kc = std::max(0, x - 8); kc < std::min(3, x + 2); ++kc) {
                        const int cell = (y + 1 - kr) * 10 + x + 1 - kc;
                        const float* weight = item.transposed.data() + ((ch * 3 + kr) * 3 + kc) * item.width;
                        float* dest = spatial + cell * item.width;
                        for (int target = 0; target < item.width; target += 8)
                            _mm256_storeu_ps(dest + target, _mm256_add_ps(_mm256_loadu_ps(dest + target),
                                _mm256_mul_ps(input, _mm256_loadu_ps(weight + target))));
                    }
            }
    for (int cell = 0; cell < 60; ++cell)
        for (int ch = 0; ch < item.width; ++ch) out[ch * 60 + cell] = spatial[cell * item.width + ch];
}

static inline void Accumulate(float* __restrict out, const float* __restrict weight, float value, int width) {
    for (int target = 0; target < width; ++target) out[target] += value * weight[target];
}

__attribute__((noinline))
void Portable(const Case& item, float* spatial, float* out) {
    for (int cell = 0; cell < 60; ++cell) std::copy_n(item.bias.data(), item.width, spatial + cell * item.width);
    for (int ch : item.active)
        for (int y = 0; y < 6; ++y)
            for (int x = 0; x < 10; ++x) {
                const float value = item.board[ch * 60 + y * 10 + x];
                if (value == 0) continue;
                for (int kr = std::max(0, y - 4); kr < std::min(3, y + 2); ++kr)
                    for (int kc = std::max(0, x - 8); kc < std::min(3, x + 2); ++kc) {
                        const int cell = (y + 1 - kr) * 10 + x + 1 - kc;
                        Accumulate(spatial + cell * item.width,
                            item.transposed.data() + ((ch * 3 + kr) * 3 + kc) * item.width, value, item.width);
                    }
            }
    for (int cell = 0; cell < 60; ++cell)
        for (int ch = 0; ch < item.width; ++ch) out[ch * 60 + cell] = spatial[cell * item.width + ch];
}

int main(int argc, char** argv) {
    if (argc != 2) throw std::runtime_error("usage: bench-sparse-conv /absolute/PVZKERN1.bin");
    std::ifstream in(argv[1], std::ios::binary);
    std::array<char, 8> magic; Read(in, magic.data(), 8);
    if (std::string(magic.data(), 8) != "PVZKERN1") throw std::runtime_error("bad fixture");
    uint32_t count; Read(in, &count, 1);
    std::vector<Case> items;
    size_t nonzero = 0, cells = 0;
    for (uint32_t i = 0; i < count; ++i) {
        std::array<int32_t, 5> shape; Read(in, shape.data(), 5);
        const auto [channels, width, rows, cols, first] = shape;
        std::vector<float> board(channels * 60), weights(width * channels * rows * cols), bias(width);
        Read(in, board.data(), board.size()); Read(in, weights.data(), weights.size()); Read(in, bias.data(), bias.size());
        if (!first) continue;
        if (rows != 3 || cols != 3 || width != 64) throw std::runtime_error("unsupported first layer");
        Case item{channels, width, 0}; item.board = std::move(board); item.weights = std::move(weights); item.bias = bias;
        for (int ch = 0; ch < channels; ++ch)
            if (std::any_of(item.board.begin() + ch * 60, item.board.begin() + (ch + 1) * 60,
                            [](float v) { return v != 0; })) item.active.push_back(ch);
        item.count = item.active.size() * 9;
        item.packed.resize(width * item.count); item.transposed.resize(item.weights.size());
        for (int target = 0; target < width; ++target) {
            int feature = 0;
            for (int ch : item.active) for (int k = 0; k < 9; ++k, ++feature)
                item.packed[target * item.count + feature] = item.weights[(target * channels + ch) * 9 + k];
            for (int k = 0; k < channels * 9; ++k)
                item.transposed[k * width + target] = item.weights[target * channels * 9 + k];
        }
        for (float v : item.board) nonzero += v != 0;
        cells += item.board.size(); items.push_back(std::move(item));
    }
    if (items.size() != 64) throw std::runtime_error("expected 64 real roots");
    std::vector<float> columns(512 * 9 * 60), spatial(64 * 60), a(64 * 60), b(64 * 60), c(64 * 60);
    double maxError = 0, error2 = 0, reference2 = 0, portableError = 0;
    for (const Case& item : items) {
        Blas(item, columns.data(), a.data()); Sparse(item, spatial.data(), b.data()); Portable(item, spatial.data(), c.data());
        for (size_t j = 0; j < a.size(); ++j) {
            maxError = std::max(maxError, static_cast<double>(std::abs(a[j] - b[j])));
            portableError = std::max(portableError, static_cast<double>(std::abs(c[j] - b[j])));
            error2 += std::pow(static_cast<double>(a[j]) - b[j], 2); reference2 += std::pow(a[j], 2);
        }
    }
    constexpr int repeats = 100;
    std::cout << "{\"roots\":" << items.size() << ",\"board_nonzero_fraction\":" << static_cast<double>(nonzero) / cells
              << ",\"max_abs_error\":" << maxError << ",\"relative_rms\":" << std::sqrt(error2 / reference2)
              << ",\"portable_vs_avx2_max_abs_error\":" << portableError
              << ",\"scope\":\"first-layer convolution including im2col and output layout; fixed weight preparation excluded\",\"rounds\":[";
    for (int round = 0; round < 9; ++round) {
        std::array<double, 3> seconds{};
        for (size_t index = 0; index < items.size(); ++index)
            for (int order = 0; order < 3; ++order) {
                const int arm = (round + index + order) % 3; const double start = Cpu();
                for (int repeat = 0; repeat < repeats; ++repeat)
                    if (arm == 0) Blas(items[index], columns.data(), a.data());
                    else if (arm == 1) Sparse(items[index], spatial.data(), b.data());
                    else Portable(items[index], spatial.data(), c.data());
                seconds[arm] += Cpu() - start;
            }
        std::cout << (round ? ",{\"blas_us\":" : "{\"blas_us\":") << seconds[0] * 1e6 / (repeats * items.size())
                  << ",\"sparse_us\":" << seconds[1] * 1e6 / (repeats * items.size())
                  << ",\"sparse_over_blas\":" << seconds[1] / seconds[0]
                  << ",\"portable_us\":" << seconds[2] * 1e6 / (repeats * items.size())
                  << ",\"portable_over_blas\":" << seconds[2] / seconds[0] << '}';
    }
    std::cout << "],\"checksum\":" << a[0] + b[0] << "}\n";
}
