#ifndef PVZ_GENERIC_FORWARD_H
#define PVZ_GENERIC_FORWARD_H

#include <string>
#include <vector>
#include <memory>

// Capacity-bounded float32 graph. Each worker owns its scratch and weight packs.
// No allocation in Run(); rich-token graphs can change logical length within loaded capacity.
namespace pvz::net {
struct NativeTensor;
struct NativeConvolution;
struct ChannelTensor;
struct SparseConvolution;
struct PackedLinear;
struct Tensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
    int role = 0; // 0 activation, 1 input, 2 weight
    size_t elements = 0; // logical size; data retains capacity when token length changes
    unsigned tokenAxes = 0;
    std::shared_ptr<PackedLinear> linear;
    std::shared_ptr<NativeTensor> native;
    std::shared_ptr<ChannelTensor> channelsLast;
};
struct Node {
    struct RowCopy { int source, destination, count; };
    std::string op;
    std::vector<int> inputs, ints;
    std::vector<float> floats;
    int output = 0, stage = 0;
    int code = -1;
    std::vector<float> scratch;
    std::vector<float> packed;
    std::vector<int> indices;
    std::vector<int> routes;
    std::vector<float> routeWeights;
    std::vector<RowCopy> copies;
    std::shared_ptr<std::vector<int>> relationIndices;
    bool buildRelationIndices = false;
    int activeCount = -1;
    std::shared_ptr<NativeConvolution> native;
    std::shared_ptr<SparseConvolution> sparse;
};
class Graph {
public:
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;
    void Load(const std::string& path);
    void Run(int firstStage = 0, int lastStage = 2);
    // Rich-token graphs only: compact logical shapes, retaining all load-time allocations.
    void SetTokenCount(int count);
    // Explicit development-tool profiling only; never called by native rollout.
    std::vector<double> Profile(int iterations);
    void Materialize(); // Debug/export: make all native-layout intermediates available as NCHW.
    Tensor& Get(const std::string& name);
    const Tensor& Get(const std::string& name) const;
    bool Has(const std::string& name) const;
    // Metadata: [observation id, task context, identity packet head]. Empty for standalone graphs.
    std::vector<int> metadata;
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;
private:
    int tokenCapacity = 0, tokenCount = 0;
    void Prepare();
    void Execute(Node& node);
};
}
#endif
