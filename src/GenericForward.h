#ifndef PVZ_GENERIC_FORWARD_H
#define PVZ_GENERIC_FORWARD_H

#include <string>
#include <vector>

// Fixed-shape float32 inference graph. A Graph owns its scratch space; use one per worker.
// No interpreter, allocation or shape inference in Run(). Sparse kernels are cached by channel set.
namespace pvz::net {
struct Tensor {
    std::string name;
    std::vector<int> shape;
    std::vector<float> data;
    int role = 0; // 0 activation, 1 input, 2 weight
};
struct Node {
    std::string op;
    std::vector<int> inputs, ints;
    std::vector<float> floats;
    int output = 0, stage = 0;
    int code = -1;
    std::vector<float> scratch;
    std::vector<float> packed;
    std::vector<int> indices;
    int activeCount = -1;
};
class Graph {
public:
    void Load(const std::string& path);
    void Run(int firstStage = 0, int lastStage = 2);
    Tensor& Get(const std::string& name);
    const Tensor& Get(const std::string& name) const;
    bool Has(const std::string& name) const;
    // Metadata: [observation id, task context, identity packet head]. Empty for standalone graphs.
    std::vector<int> metadata;
    std::vector<Tensor> tensors;
    std::vector<Node> nodes;
private:
    void Prepare();
    void Execute(Node& node);
};
}
#endif
