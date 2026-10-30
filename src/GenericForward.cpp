#include "GenericForward.h"
#include "GenericForwardDnnl.h"
#include "GenericForwardSparse.h"
#include "GenericForwardToken.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <utility>
#ifdef __APPLE__
#include <vecLib/cblas.h>
#else
#include <cblas.h>
#endif

namespace pvz::net {
namespace {
const std::vector<std::string> ops = {"linear", "conv", "relu", "add", "concat", "reshape",
    "permute", "pool", "norm", "attention", "ffn", "moe", "scatter", "pointer",
    "embedding", "broadcast", "select", "slice", "mask", "relation", "silu", "gather", "masked_pool", "conv_relu", "row_context"};
void Require(bool ok, std::string_view message) {
    if (!ok) throw std::runtime_error("generic forward: " + std::string(message));
}
template<class T> T Read(std::istream& in) {
    T value{}; in.read(reinterpret_cast<char*>(&value), sizeof(T));
    Require(bool(in), "truncated file"); return value;
}
std::string ReadString(std::istream& in) {
    auto size = Read<uint32_t>(in); Require(size <= 65536, "string length");
    std::string result(size, '\0'); in.read(result.data(), size);
    Require(bool(in), "truncated string"); return result;
}
template<class T> std::vector<T> ReadVector(std::istream& in, size_t limit) {
    auto size = Read<uint32_t>(in); Require(size <= limit, "vector length");
    std::vector<T> result(size); in.read(reinterpret_cast<char*>(result.data()), size * sizeof(T));
    Require(bool(in), "truncated vector"); return result;
}
size_t Count(const std::vector<int>& shape) {
    size_t count = 1;
    for (int d : shape) { Require(d > 0 && count <= 64000000 / size_t(d), "tensor shape"); count *= d; }
    return count;
}
// Linear weights retain PyTorch's [out,in] layout; BLAS consumes them transposed.
void Linear(const float* x, const Tensor& w, const Tensor* bias, float* y, int rows) {
    const int out = w.shape[0], in = w.shape[1];
    if(PackedMatrix(w,x,bias ? bias->data.data() : nullptr,y,rows)) return;
    if(out == 1) {
        cblas_sgemv(CblasRowMajor,CblasNoTrans,rows,in,1.f,x,in,w.data.data(),1,0.f,y,1);
        if(bias) for(int r=0;r<rows;++r) y[r]+=bias->data[0];
        return;
    }
    if (rows == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, out, in, 1.f, w.data.data(), in, x, 1, 0.f, y, 1);
        if (bias) for (int c=0;c<out;++c) y[c]+=bias->data[c];
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, out, in,
        1.f, x, in, w.data.data(), in, 0.f, y, out);
    if (bias) for (int r = 0; r < rows; ++r)
        for (int c = 0; c < out; ++c) y[r * out + c] += bias->data[c];
}
void Activate(float* x, int n, bool swiglu, int width) {
    if (!swiglu) { for (int i = 0; i < n; ++i) x[i] = std::max(0.f, x[i]); return; }
    TokenSilu(x,x,n/(2*width),width,true);
}
void FFN(const float* x, int rows, const Tensor& up, const Tensor& ub,
         const Tensor& down, const Tensor& db, bool swiglu, float* tmp, float* y) {
    Linear(x, up, &ub, tmp, rows);
    Activate(tmp, rows * up.shape[0], swiglu, down.shape[1]);
    Linear(tmp, down, &db, y, rows);
}
}

bool Graph::Has(const std::string& name) const {
    return std::any_of(tensors.begin(), tensors.end(), [&](const Tensor& t) { return t.name == name; });
}
const Tensor& Graph::Get(const std::string& name) const {
    for (const auto& t : tensors) if (t.name == name) {
        auto& cache=const_cast<Tensor&>(t); SparseToHost(cache); NativeToHost(cache); return t;
    }
    throw std::runtime_error("generic forward: missing tensor " + name);
}
Tensor& Graph::Get(const std::string& name) { return const_cast<Tensor&>(std::as_const(*this).Get(name)); }
void Graph::Materialize() { for(auto& t:tensors) { SparseToHost(t); NativeToHost(t); } }
void Graph::Load(const std::string& path) {
#ifdef OPENBLAS_VERSION
    openblas_set_num_threads(1);
#elif defined(__APPLE__)
    setenv("VECLIB_MAXIMUM_THREADS", "1", 1);
#endif
    std::ifstream in(path, std::ios::binary);
    char magic[8]{}; in.read(magic, 8);
    Require(std::string(magic, 8) == "PVZGN001" && std::endian::native == std::endian::little, "format/endian");
    Graph next;
    next.metadata = ReadVector<int>(in, 32);
    auto nt = Read<uint32_t>(in); Require(nt > 0 && nt <= 10000, "tensor count");
    size_t total = 0;
    for (uint32_t i = 0; i < nt; ++i) {
        Tensor t;
        t.name = ReadString(in); t.role = Read<int>(in); t.shape = ReadVector<int>(in, 8);
        Require(!next.Has(t.name) && t.role >= 0 && t.role <= 2, "tensor name/role");
        size_t count = Count(t.shape); total += count;
        Require(total <= 128000000, "graph memory limit");
        t.data.resize(count); t.elements = count;
        if (t.role == 2) { in.read(reinterpret_cast<char*>(t.data.data()), count * 4); Require(bool(in), "truncated weight"); }
        next.tensors.push_back(std::move(t));
    }
    auto nn = Read<uint32_t>(in); Require(nn <= 10000, "node count");
    for (uint32_t i = 0; i < nn; ++i) {
        Node n; n.op = ReadString(in); n.stage = Read<int>(in); n.output = Read<int>(in);
        n.inputs = ReadVector<int>(in, 1024); n.ints = ReadVector<int>(in, 64); n.floats = ReadVector<float>(in, 16);
        next.nodes.push_back(std::move(n));
    }
    Require(in.peek() == std::char_traits<char>::eof(), "trailing data");
    next.Prepare(); *this = std::move(next);
}

void Graph::Prepare() {
    if (!metadata.empty() && metadata[0] == 10 && Has("tokens")) {
        tokenCapacity = tokenCount = Get("tokens").shape[1];
        for (auto& t : tensors) if (t.role == 1 && (t.name == "tokens" || t.name == "token_meta" ||
            t.name == "valid" || t.name == "pool_valid" || t.name == "coords")) t.tokenAxes = 2;
    }
    std::vector<bool> ready;
    for (const auto& t : tensors) ready.push_back(t.role != 0);
    int stage = 0;
    for (auto& n : nodes) {
        auto found = std::find(ops.begin(), ops.end(), n.op);
        Require(found != ops.end(), "unknown op " + n.op); n.code = int(found - ops.begin());
        Require(n.stage >= stage && n.stage <= 2, "stage order"); stage = n.stage;
        Require(n.output >= 0 && size_t(n.output) < tensors.size() && !ready[n.output], "output index");
        for (int i : n.inputs) Require(i >= 0 && size_t(i) < tensors.size() && ready[i], "input dependency");
        Require(!n.inputs.empty(), "empty input list");
        auto& a = tensors[n.inputs[0]]; auto& y = tensors[n.output];
        // Bound all operation-specific shape/index checks to model loading, never per frame.
        auto param = [&](size_t count) { Require(n.ints.size() == count, n.op + " parameters"); };
        auto inputs = [&](size_t count) { Require(n.inputs.size() == count, n.op + " inputs"); };
        auto at = [&](int i) -> Tensor& { return tensors[n.inputs.at(i)]; };
        const int rank = a.shape.size();
        if (n.code == 0) {
            Require(n.inputs.size() == 2 || n.inputs.size() == 3, "linear inputs");
            const auto& w = at(1); Require(w.shape.size() == 2 && a.shape.back() == w.shape[1], "linear shape");
            auto expected = a.shape; expected.back() = w.shape[0]; Require(y.shape == expected, "linear output");
            if (n.inputs.size() == 3) Require(at(2).data.size() == size_t(w.shape[0]), "linear bias");
        } else if (n.code == 1 || n.code == 23) {
            Require(n.inputs.size() == 2 || n.inputs.size() == 3, "conv inputs");
            Require(n.ints.size() == 4 || (n.ints.size() == 5 && n.ints[4] == 1), "conv parameters");
            auto& w = at(1);
            Require(rank == 4 && w.shape.size() == 4 && y.shape.size() == 4, "conv rank");
            Require(n.ints[0] >= 0 && n.ints[1] >= 0 && n.ints[2] > 0 && n.ints[3] > 0, "conv stride/padding");
            Require(a.shape[1] == w.shape[1] && (n.inputs.size() == 2 || at(2).data.size() == size_t(w.shape[0])) &&
                y.shape == std::vector<int>({a.shape[0], w.shape[0],
                (a.shape[2]+2*n.ints[0]-w.shape[2])/n.ints[2]+1,
                (a.shape[3]+2*n.ints[1]-w.shape[3])/n.ints[3]+1}), "conv shape");
            n.scratch.resize(size_t(w.shape[1])*w.shape[2]*w.shape[3]*y.shape[2]*y.shape[3]);
            if (n.ints.size() == 5) { n.indices.resize(a.shape[1], -1); n.packed.resize(w.data.size()); }
            if (n.ints[3] == 1) {
                const int OH=y.shape[2],OW=y.shape[3],H=a.shape[2],W=a.shape[3];
                for(int kh=0;kh<w.shape[2];++kh) for(int kw=0;kw<w.shape[3];++kw) {
                    int begin=std::max(0,n.ints[1]-kw), end=std::min(OW,W+n.ints[1]-kw);
                    if(begin>=end) continue;
                    for(int h=0;h<OH;++h) {
                        int ih=h*n.ints[2]+kh-n.ints[0];
                        if(ih>=0 && ih<H) n.copies.push_back({ih*W+begin+kw-n.ints[1],(kh*w.shape[3]+kw)*OH*OW+h*OW+begin,end-begin});
                    }
                }
            }
        } else if (n.code == 2 || n.code == 3 || n.code == 18) {
            inputs(n.code == 2 ? 1 : 2); Require(a.shape == y.shape, "elementwise shape");
            if (n.code == 3) Require(at(1).shape == a.shape, "add shape");
            if (n.code == 18) Require(a.data.size() % at(1).data.size() == 0 && n.floats.size() == 1, "mask shape/fill");
        } else if (n.code == 4) {
            param(1); int axis = n.ints[0]; Require(axis >= 0 && axis < rank, "concat axis");
            auto expected = a.shape; expected[axis] = 0;
            for (int i : n.inputs) { auto shape = tensors[i].shape; Require(shape.size() == size_t(rank), "concat rank");
                expected[axis] += shape[axis]; shape[axis] = a.shape[axis]; Require(shape == a.shape, "concat shape"); }
            Require(y.shape == expected, "concat output");
        } else if (n.code == 5) {
            inputs(1); Require(a.data.size() == y.data.size(), "reshape size");
        } else if (n.code == 6 || n.code == 15) {
            inputs(1); Require(n.ints.size() == y.shape.size(), "layout rank");
            n.indices.resize(y.data.size());
            std::vector<int> used(rank, 0);
            if (n.code == 6) { Require(rank == int(y.shape.size()), "permute rank");
                for (int axis = 0; axis < rank; ++axis) { int i = n.ints[axis]; Require(i >= 0 && i < rank && !used[i]++, "permute axis");
                    Require(y.shape[axis] == a.shape[i], "permute shape"); } }
            else for (size_t axis = 0; axis < y.shape.size(); ++axis) {
                int i = n.ints[axis]; Require(i == -1 || (i >= 0 && i < rank && !used[i]++ && a.shape[i] == y.shape[axis]), "broadcast axis"); }
            if (n.code == 15) for (int i = 0; i < rank; ++i) Require(used[i] || a.shape[i] == 1, "broadcast lost dimension");
            std::vector<int> strides(rank, 1);
            for (int i = rank-2; i >= 0; --i) strides[i] = strides[i+1]*a.shape[i+1];
            for (size_t i = 0; i < y.data.size(); ++i) {
                size_t rem = i; int source = 0;
                for (int axis = int(y.shape.size())-1; axis >= 0; --axis) {
                    int c = rem % y.shape[axis]; rem /= y.shape[axis];
                    if (n.ints[axis] >= 0) source += c*strides[n.ints[axis]];
                } n.indices[i] = source;
            }
        } else if (n.code == 7) {
            inputs(1); param(2); // number of trailing axes to reduce; 0 mean, 1 max, 2 meanmax
            Require(n.ints[0] > 0 && n.ints[0] < rank && n.ints[1] >= 0 && n.ints[1] <= 2, "pool parameters");
            auto expected = a.shape; expected.resize(rank-n.ints[0]); if (n.ints[1] == 2) expected.back() *= 2;
            Require(expected == y.shape, "pool output");
        } else if (n.code == 8) {
            inputs(3); param(1); Require(n.ints[0] >= 0 && n.ints[0] <= 1 && n.floats.size() == 1 && n.floats[0] > 0, "norm kind/eps");
            Require(a.shape == y.shape && at(1).data.size() == size_t(a.shape.back()) && at(2).data.size() == size_t(a.shape.back()), "norm shape");
        } else if (n.code == 9) {
            // q,k,v [B,T,D], bias [B,H,T,T], valid [B,T]; masked queries output zero.
            inputs(5); param(1); Require(rank == 3 && n.ints[0] > 0 && a.shape[2] % n.ints[0] == 0, "attention heads");
            Require(at(1).shape == a.shape && at(2).shape == a.shape && y.shape == a.shape &&
                at(3).shape == std::vector<int>({a.shape[0], n.ints[0], a.shape[1], a.shape[1]}) &&
                at(4).shape == std::vector<int>({a.shape[0], a.shape[1]}), "attention shape");
            n.scratch.resize(size_t(a.shape[1])*a.shape[1]);
        } else if (n.code == 10 || n.code == 11) {
            param(n.code == 10 ? 1 : 5);
            if (n.code == 10) inputs(5);
            // moe ints: activation, route (0 topk / 1 external id), experts, topk, shared experts
            int experts = n.code == 10 ? 1 : n.ints[2] + n.ints[4];
            Require(n.ints[0] == 0 || n.ints[0] == 1, "FFN activation");
            Require(y.shape == a.shape && experts > 0, "FFN output");
            int offset = n.code == 10 ? 1 : 2;
            if (n.code == 11) {
                inputs(2 + 4 * experts); Require((n.ints[1] == 0 || n.ints[1] == 1) && n.ints[2] > 0 &&
                    n.ints[3] > 0 && n.ints[3] <= n.ints[2] && n.ints[4] >= 0, "moe routing");
                size_t rows = a.data.size()/a.shape.back();
                Require(at(1).data.size() == rows * (n.ints[1] ? 1 : n.ints[2]), "moe route shape");
                n.indices.resize(std::max(rows,size_t(n.ints[2])));
                if(!n.ints[1]) { n.routes.resize(rows*n.ints[3]); n.routeWeights.resize(rows*(n.ints[3]+1)); }
            }
            size_t scratch = 0;
            for (int e = 0; e < experts; ++e) {
                auto& up = at(offset+4*e); auto& down = at(offset+4*e+2);
                Require(up.shape.size() == 2 && down.shape.size() == 2 && up.shape[1] == a.shape.back() &&
                    up.shape[0] == down.shape[1]*(n.ints[0] ? 2 : 1) && down.shape[0] == a.shape.back() &&
                    at(offset+4*e+1).data.size() == size_t(up.shape[0]) && at(offset+4*e+3).data.size() == size_t(down.shape[0]), "FFN weights");
                scratch = std::max(scratch, size_t(up.shape[0])*(a.data.size()/a.shape.back()));
            } n.scratch.resize(scratch + (n.code == 11 ? 2*a.data.size() : 0));
        } else if (n.code == 12) {
            inputs(3); Require(rank == 4 && a.shape == y.shape && at(1).shape.size() == 3 &&
                at(1).shape[0] == a.shape[0] && at(1).shape[2] == a.shape[1] &&
                at(2).shape == std::vector<int>({a.shape[0], at(1).shape[1], 2}), "scatter shape");
        } else if (n.code == 13) {
            inputs(2); Require(rank == 2 && at(1).shape.size() == 3 && at(1).shape[0] == a.shape[0] &&
                at(1).shape[2] == a.shape[1] && y.shape == std::vector<int>({a.shape[0], at(1).shape[1]}) && n.floats.size() == 1, "pointer shape");
        } else if (n.code == 14) {
            inputs(2); Require(at(1).shape.size() == 2, "embedding table");
            auto expected = a.shape; expected.push_back(at(1).shape[1]); Require(y.shape == expected, "embedding output");
        } else if (n.code == 16) {
            inputs(2); Require(rank == 3 && at(1).data.size() == size_t(a.shape[0]) && y.shape == std::vector<int>({a.shape[0], a.shape[2]}), "select shape");
        } else if (n.code == 19) {
            Require(n.inputs.size() == 5 || n.inputs.size() == 7, "relation inputs"); param(2);
            Require(rank == 3 && a.shape[2] == 3 && at(1).shape.size() == 3 &&
                at(1).shape[1] == at(1).shape[2] && n.ints[0] >= 0 && n.ints[1] >= 0, "relation metadata/table");
            int heads=at(1).shape[0];
            Require(at(2).shape == std::vector<int>({heads, 2*n.ints[0]+2}) &&
                at(3).shape == std::vector<int>({heads, 2*n.ints[1]+2}) &&
                (at(4).data.size() == size_t(heads) || at(4).shape == std::vector<int>({heads, 2})) &&
                y.shape == std::vector<int>({a.shape[0],heads,a.shape[1],a.shape[1]}), "relation shape");
            if(n.inputs.size() == 7) Require(at(5).shape == std::vector<int>({a.shape[0], a.shape[1]}) &&
                at(6).data.size() == size_t(heads), "target relation shape");
            if (std::all_of(n.inputs.begin()+1,n.inputs.begin()+5,[&](int i){ return tensors[i].role==2; })) {
                int K=at(1).shape[1], RB=2*n.ints[0]+2, CB=2*n.ints[1]+2;
                n.packed.resize(size_t(heads)*K*K*RB*CB*2);
                for(int h=0;h<heads;++h) for(int pair=0;pair<K*K;++pair)
                    for(int r=0;r<RB;++r) for(int c=0;c<CB;++c) for(int same=0;same<2;++same)
                        n.packed[((((h*K*K+pair)*RB+r)*CB+c)*2+same)]=at(1).data[h*K*K+pair]+
                            at(2).data[h*RB+r]+at(3).data[h*CB+c]+
                            (at(4).data.size()==size_t(heads) ? (same ? at(4).data[h] : 0.f) : at(4).data[h*2+same]);
                for(auto& prior:nodes) {
                    if(&prior==&n) break;
                    if(prior.code==19 && prior.inputs[0]==n.inputs[0] && prior.ints==n.ints &&
                        tensors[prior.inputs[1]].shape[1]==K && prior.relationIndices) {
                        n.relationIndices=prior.relationIndices; break;
                    }
                }
                if(!n.relationIndices) {
                    n.relationIndices=std::make_shared<std::vector<int>>(size_t(a.shape[0])*a.shape[1]*a.shape[1]);
                    n.buildRelationIndices=true;
                }
            }
        } else if (n.code == 20) {
            inputs(1); Require(a.shape == y.shape, "silu shape");
        } else if (n.code == 21) {
            inputs(2); Require(rank == 3 && at(1).shape.size() == 2 && at(1).shape[0] == a.shape[0] &&
                y.shape == std::vector<int>({a.shape[0], at(1).shape[1], a.shape[2]}), "gather shape");
        } else if (n.code == 22) {
            inputs(2); Require(rank == 3 && at(1).shape == std::vector<int>({a.shape[0],a.shape[1]}) &&
                y.shape == std::vector<int>({a.shape[0],a.shape[2]}), "masked pool shape");
        } else if (n.code == 24) {
            inputs(3);
            Require(rank==4 && y.shape==a.shape && at(1).data.size()==size_t(2*a.shape[1]*a.shape[1]) &&
                at(1).shape[0]==a.shape[1] && at(1).shape[1]==2*a.shape[1] && at(2).data.size()==size_t(a.shape[1]),"row context shape");
            n.scratch.resize(3*a.shape[0]*a.shape[2]*a.shape[1]);
        } else if (n.code == 17) {
            inputs(1); param(3); int axis=n.ints[0], start=n.ints[1], length=n.ints[2];
            Require(axis >= 0 && axis < rank && start >= 0 && length > 0 && start+length <= a.shape[axis], "slice range");
            auto expected=a.shape; expected[axis]=length; Require(y.shape==expected, "slice output");
        }
        if (tokenCapacity) {
            switch (n.code) {
            case 0: case 2: case 3: case 8: case 10: case 11: case 14: case 17: case 18: case 20:
                y.tokenAxes = a.tokenAxes; break;
            case 5:
                if (a.tokenAxes) {
                    Require(a.tokenAxes == 2 && y.shape.size() >= 2 && y.shape[1] == tokenCapacity,
                        "token reshape requires preserved token axis");
                    y.tokenAxes = 2;
                } break;
            case 9: y.tokenAxes = a.tokenAxes; break;
            case 19: y.tokenAxes = a.tokenAxes ? 12 : 0; break;
            case 6: case 15:
                for (size_t i=0;i<n.ints.size();++i)
                    if (n.ints[i]>=0 && (a.tokenAxes & (1u<<n.ints[i]))) y.tokenAxes |= 1u<<i;
                if (n.code==15 && a.name=="terrain" && y.shape.size()==2) y.tokenAxes=2;
                break;
            case 4: y.tokenAxes = a.tokenAxes; Require(!(y.tokenAxes & (1u<<n.ints[0])), "token concat axis"); break;
            case 7: y.tokenAxes = a.tokenAxes & ((1u<<(rank-n.ints[0]))-1); break;
            case 13: y.tokenAxes = at(1).tokenAxes & 2; break;
            }
        }
        ready[n.output] = true;
        if(n.code==0) PreparePackedLinear(at(1));
        if(n.code==10 || n.code==11) {
            int offset=n.code==10 ? 1 : 2;
            for(size_t i=offset;i<n.inputs.size();i+=4) { PreparePackedLinear(at(i)); PreparePackedLinear(at(i+2)); }
        }
        PrepareSparseConvolution(*this,n);
        if(!n.sparse) PrepareNativeConvolution(*this,n);
        // E.g. split a 1x1 kernel once at load, not once per decision.
        if (std::all_of(n.inputs.begin(), n.inputs.end(), [&](int i) { return tensors[i].role == 2; })) {
            Execute(n); SparseToHost(tensors[n.output]); NativeToHost(tensors[n.output]);
            tensors[n.output].role = 2; n.code = -1;
        }
    }
}

void Graph::SetTokenCount(int count) {
    Require(tokenCapacity && count > 0 && count <= tokenCapacity, "token capacity exceeded");
    if (count == tokenCount) return;
    tokenCount = count;
    for (auto& t : tensors) if (t.tokenAxes) {
        t.elements = 1;
        for (size_t d=0;d<t.shape.size();++d) {
            if (t.tokenAxes & (1u<<d)) t.shape[d] = count;
            t.elements *= t.shape[d];
        }
    }
    // Only index maps depend on shape; matrix packs and scratch retain capacity.
    for (auto& n : nodes) if ((n.code==6 || n.code==15) && tensors[n.output].tokenAxes) {
        auto& a=tensors[n.inputs[0]]; auto& y=tensors[n.output];
        int stride[8]; stride[a.shape.size()-1]=1;
        for (int i=int(a.shape.size())-2;i>=0;--i) stride[i]=stride[i+1]*a.shape[i+1];
        for (size_t i=0;i<y.elements;++i) {
            size_t rem=i; int source=0;
            for (int axis=int(y.shape.size())-1;axis>=0;--axis) {
                int c=rem%y.shape[axis]; rem/=y.shape[axis];
                if(n.ints[axis]>=0) source+=c*stride[n.ints[axis]];
            }
            n.indices[i]=source;
        }
    }
}

void Graph::Run(int firstStage, int lastStage) {
    for (auto& n : nodes) if (n.code >= 0 && n.stage >= firstStage && n.stage <= lastStage) Execute(n);
}
std::vector<double> Graph::Profile(int iterations) {
    std::vector<double> times(nodes.size(), 0.);
    for (int r=0; r<iterations; ++r) for (size_t i=0; i<nodes.size(); ++i) {
        if (nodes[i].code < 0) continue;
        auto start=std::chrono::steady_clock::now();
        Execute(nodes[i]);
        times[i]+=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
    }
    return times;
}
void Graph::Execute(Node& n) {
    if(ExecuteSparseConvolution(n)) return;
    if(ExecuteNativeConvolution(n)) return;
    for(int id:n.inputs) { SparseToHost(tensors[id]); NativeToHost(tensors[id]); }
    auto& a = tensors[n.inputs[0]]; auto& y = tensors[n.output];
    auto at = [&](int i) -> const Tensor& { return tensors[n.inputs[i]]; };
    float* out = y.data.data(); const float* x = a.data.data();
    const int count = int(y.elements);
    switch (n.code) {
    case 0: Linear(x, at(1), n.inputs.size()==3 ? &at(2) : nullptr, out, a.elements/a.shape.back()); break;
    case 1: case 23: {
        const auto& w=at(1); int B=a.shape[0], C=a.shape[1], H=a.shape[2], W=a.shape[3];
        int O=w.shape[0], KH=w.shape[2], KW=w.shape[3], OH=y.shape[2], OW=y.shape[3], S=OH*OW;
        float* col=n.scratch.data();
        const bool sparse=n.ints.size()==5;
        int active=C; const float* kernel=w.data.data();
        if(sparse) {
            active=0; bool changed=false;
            for(int c=0;c<C;++c) {
                bool nonzero=false;
                for(int b=0;b<B && !nonzero;++b) {
                    const float* plane=x+(b*C+c)*H*W;
                    uint64_t bits=0;
                    for(int i=0;i+1<H*W;i+=2) { uint64_t word; std::memcpy(&word,plane+i,8); bits|=word; }
                    if(H*W%2) { uint32_t word; std::memcpy(&word,plane+H*W-1,4); bits|=word; }
                    nonzero=bits!=0;
                }
                if(nonzero) { changed |= n.indices[active]!=c; n.indices[active++]=c; }
            }
            if(changed || active!=n.activeCount) {
                for(int o=0;o<O;++o) for(int c=0;c<active;++c)
                    std::copy_n(kernel+(o*C+n.indices[c])*KH*KW,KH*KW,n.packed.data()+(o*active+c)*KH*KW);
                n.activeCount=active;
            }
            kernel=n.packed.data();
        }
        int K=active*KH*KW;
        for (int b=0;b<B;++b) {
            const bool direct=!sparse && KH==1 && KW==1 && n.ints[0]==0 && n.ints[1]==0 && n.ints[2]==1 && n.ints[3]==1;
            if(!direct && n.ints[3]==1) for(int ac=0;ac<active;++ac) {
                const float* source=x+(b*C+(sparse ? n.indices[ac] : ac))*H*W;
                float* dest=col+ac*KH*KW*S;
                for(const auto& copy:n.copies) {
                    const float* src=source+copy.source; float* dst=dest+copy.destination;
                    switch(copy.count) {
                    case 10: std::memcpy(dst,src,40); break;
                    case 9: std::memcpy(dst,src,36); break;
                    case 8: std::memcpy(dst,src,32); break;
                    default: std::memcpy(dst,src,copy.count*sizeof(float)); break;
                    }
                }
            }
            for (int ac=0;ac<active && !direct && n.ints[3]!=1;++ac) for (int kh=0;kh<KH;++kh) for (int kw=0;kw<KW;++kw) {
                int c=sparse ? n.indices[ac] : ac;
                float* dst=col+((ac*KH+kh)*KW+kw)*S;
                for (int h=0;h<OH;++h) for (int v=0;v<OW;++v) {
                    int ih=h*n.ints[2]+kh-n.ints[0], iw=v*n.ints[3]+kw-n.ints[1];
                    dst[h*OW+v]=(ih>=0 && ih<H && iw>=0 && iw<W) ? x[((b*C+c)*H+ih)*W+iw] : 0.f;
                }
            }
            float* dst=out+b*O*S;
            if(K) cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,O,S,K,1.f,kernel,K,direct ? x+b*C*S : col,S,0.f,dst,S);
            else std::fill_n(dst,O*S,0.f);
            if(n.inputs.size()==3) for(int o=0;o<O;++o) for(int s=0;s<S;++s) dst[o*S+s]+=at(2).data[o];
            if(n.code==23) for(int i=0;i<O*S;++i) dst[i]=std::max(0.f,dst[i]);
        }
        if(n.sparse) SparseFromHost(y);
        break;
    }
    case 2: for(int i=0;i<count;++i) out[i]=std::max(0.f,x[i]); break;
    case 3: for(int i=0;i<count;++i) out[i]=x[i]+at(1).data[i]; break;
    case 4: {
        int axis=n.ints[0], outer=1, inner=1;
        for(int i=0;i<axis;++i) outer*=a.shape[i];
        for(size_t i=axis+1;i<a.shape.size();++i) inner*=a.shape[i];
        for(int r=0;r<outer;++r) { int offset=r*y.shape[axis]*inner;
            for(int index:n.inputs) { const auto& t=tensors[index]; int len=t.shape[axis]*inner;
                std::copy_n(t.data.data()+r*len,len,out+offset); offset+=len; }
        } break;
    }
    case 5: std::copy_n(x,a.elements,out); break;
    case 6: case 15: for(int i=0;i<count;++i) out[i]=x[n.indices[i]]; break;
    case 7: {
        int reduce=1; for(int i=int(a.shape.size())-n.ints[0];i<int(a.shape.size());++i) reduce*=a.shape[i];
        int rows=a.elements/reduce, channels=a.shape[a.shape.size()-n.ints[0]-1];
        for(int r=0;r<rows;++r) {
            float sum=0, maximum=-std::numeric_limits<float>::infinity();
            for(int c=0;c<reduce;++c) { float v=x[r*reduce+c]; sum+=v; maximum=std::max(maximum,v); }
            if(n.ints[1]==2) { int base=(r/channels)*2*channels+r%channels; out[base]=sum/reduce; out[base+channels]=maximum; }
            else out[r]=n.ints[1] ? maximum : sum/reduce;
        } break;
    }
    case 8: {
        int D=a.shape.back();
        for(int r=0;r<count/D;++r) {
            double mean=0, var=0;
            if(!n.ints[0]) { for(int c=0;c<D;++c) mean+=x[r*D+c]; mean/=D; }
            for(int c=0;c<D;++c) { double d=x[r*D+c]-mean; var+=d*d; }
            float inv=1/std::sqrt(var/D+n.floats[0]);
            for(int c=0;c<D;++c) out[r*D+c]=(x[r*D+c]-float(mean))*inv*at(1).data[c]+at(2).data[c];
        } break;
    }
    case 9: {
        int B=a.shape[0], T=a.shape[1], D=a.shape[2], H=n.ints[0], K=D/H;
        float* scores=n.scratch.data(); const float* key=at(1).data.data(); const float* value=at(2).data.data();
        for(int b=0;b<B;++b) for(int h=0;h<H;++h) {
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,T,T,K,1.f/std::sqrt(float(K)),
                x+b*T*D+h*K,D,key+b*T*D+h*K,D,0.f,scores,T);
            for(int i=0;i<T;++i) TokenSoftmax(scores+i*T,at(3).data.data()+((b*H+h)*T+i)*T,
                at(4).data.data()+b*T,T,at(4).data[b*T+i]>0);
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,T,K,T,1.f,scores,T,value+b*T*D+h*K,D,0.f,out+b*T*D+h*K,D);
        } break;
    }
    case 10: FFN(x,a.elements/a.shape.back(),at(1),at(2),at(3),at(4),n.ints[0],n.scratch.data(),out); break;
    case 11: {
        int D=a.shape.back(), E=n.ints[2], shared=n.ints[4], rows=count/D;
        std::fill_n(out,count,0.f); float* tmp=n.scratch.data();
        float* packed=tmp+n.scratch.size()-2*count;
        float* expert=packed+count;
        for(int e=0;e<shared;++e) {
            int base=2+4*(E+e);
            FFN(x,rows,at(base),at(base+1),at(base+2),at(base+3),n.ints[0],tmp,expert);
            for(int i=0;i<count;++i) out[i]+=expert[i];
        }
        if(n.ints[1]) {
            for(int r=0;r<rows;++r) { float id=at(1).data[r];
                Require(std::isfinite(id) && id>=0 && id<E && id==std::floor(id), "external expert id"); }
            for(int e=0;e<E;++e) {
                int length=0;
                for(int r=0;r<rows;++r) if(int(at(1).data[r])==e) {
                    n.indices[length]=r; std::copy_n(x+r*D,D,packed+length*D); ++length;
                }
                if(!length) continue;
                int base=2+4*e;
                FFN(packed,length,at(base),at(base+1),at(base+2),at(base+3),n.ints[0],tmp,expert);
                for(int r=0;r<length;++r) for(int d=0;d<D;++d) out[n.indices[r]*D+d]+=expert[r*D+d];
            }
            break;
        }
        const int topk=n.ints[3];
        for(int r=0;r<rows;++r) {
            const float* logits=at(1).data.data()+r*E;
            std::iota(n.indices.begin(),n.indices.begin()+E,0);
            std::partial_sort(n.indices.begin(),n.indices.begin()+topk,n.indices.begin()+E,[&](int l,int r) {
                return logits[l]==logits[r] ? l<r : logits[l]>logits[r]; });
            float maximum=logits[n.indices[0]],total=0;
            for(int k=0;k<topk;++k) total+=std::exp(logits[n.indices[k]]-maximum);
            for(int k=0;k<topk;++k) {
                n.routes[r*topk+k]=n.indices[k];
                n.routeWeights[r*topk+k]=std::exp(logits[n.indices[k]]-maximum)/total;
            }
        }
        for(int e=0;e<E;++e) {
            int length=0;
            for(int r=0;r<rows;++r) for(int k=0;k<topk;++k) if(n.routes[r*topk+k]==e) {
                n.indices[length]=r;
                n.routeWeights[rows*topk+length]=n.routeWeights[r*topk+k];
                std::copy_n(x+r*D,D,packed+length*D); ++length;
            }
            if(!length) continue;
            int base=2+4*e;
            FFN(packed,length,at(base),at(base+1),at(base+2),at(base+3),n.ints[0],tmp,expert);
            for(int r=0;r<length;++r) for(int d=0;d<D;++d)
                out[n.indices[r]*D+d]+=n.routeWeights[rows*topk+r]*expert[r*D+d];
        } break;
    }
    case 12: {
        std::copy_n(x,a.elements,out);
        int B=a.shape[0],C=a.shape[1],H=a.shape[2],W=a.shape[3],T=at(1).shape[1];
        for(int b=0;b<B;++b) for(int t=0;t<T;++t) {
            float rf=at(2).data[(b*T+t)*2], cf=at(2).data[(b*T+t)*2+1];
            if(!(rf>=0 && rf<H && cf>=0 && cf<W)) continue;
            int r=int(rf),c=int(cf);
            for(int d=0;d<C;++d) out[((b*C+d)*H+r)*W+c]+=at(1).data[(b*T+t)*C+d];
        } break;
    }
    case 13: {
        int B=a.shape[0],D=a.shape[1],T=at(1).shape[1];
        for(int b=0;b<B;++b) cblas_sgemv(CblasRowMajor,CblasNoTrans,T,D,n.floats[0],at(1).data.data()+b*T*D,D,x+b*D,1,0.f,out+b*T,1);
        break;
    }
    case 14: {
        int D=at(1).shape[1],V=at(1).shape[0];
        for(size_t i=0;i<a.elements;++i) { float id=std::max(0.f,x[i]+(n.ints.empty() ? 0 : n.ints[0]));
            if(!n.ints.empty()) id=std::min(id,float(V-1));
            Require(std::isfinite(id) && id<V && id==std::floor(id), "embedding id");
            std::copy_n(at(1).data.data()+int(id)*D,D,out+i*D); }
        break;
    }
    case 16: {
        int B=a.shape[0],T=a.shape[1],D=a.shape[2];
        for(int b=0;b<B;++b) { float id=at(1).data[b]; Require(std::isfinite(id) && id>=0 && id<T && id==std::floor(id),"select id");
            std::copy_n(x+(b*T+int(id))*D,D,out+b*D); } break;
    }
    case 17: {
        int axis=n.ints[0],inner=1,outer=1;
        for(int i=0;i<axis;++i) outer*=a.shape[i];
        for(size_t i=axis+1;i<a.shape.size();++i) inner*=a.shape[i];
        for(int r=0;r<outer;++r) std::copy_n(x+(r*a.shape[axis]+n.ints[1])*inner,n.ints[2]*inner,out+r*n.ints[2]*inner);
        break;
    }
    case 18: {
        int repeat=count/at(1).elements;
        for(int i=0;i<count;++i) out[i]=at(1).data[i/repeat]>=0 ? x[i] : n.floats[0];
        break;
    }
    case 19: {
        int B=a.shape[0],T=a.shape[1],H=at(1).shape[0],K=at(1).shape[1],R=n.ints[0],C=n.ints[1];
        for(int i=0;i<B*T;++i) Require(x[i*3]>=0 && x[i*3]<K && x[i*3]==std::floor(x[i*3]) &&
            std::isfinite(x[i*3+1]) && std::isfinite(x[i*3+2]), "relation metadata");
        if(n.relationIndices) {
            int stride=K*K*(2*R+2)*(2*C+2)*2;
            int* indices=n.relationIndices->data();
            if(n.buildRelationIndices) for(int b=0;b<B;++b) for(int i=0;i<T;++i) for(int j=0;j<T;++j) {
                const float* left=x+(b*T+i)*3; const float* right=x+(b*T+j)*3;
                int dr=left[1]<0 || right[1]<0 ? 2*R+1 : int(std::clamp(left[1]-right[1],float(-R),float(R)))+R;
                int dc=left[2]<0 || right[2]<0 ? 2*C+1 : int(std::clamp(left[2]-right[2],float(-C),float(C)))+C;
                int same=left[1]>=0 && left[2]>=0 && left[1]==right[1] && left[2]==right[2];
                indices[(b*T+i)*T+j]=(((int(left[0])*K+int(right[0]))*(2*R+2)+dr)*(2*C+2)+dc)*2+same;
            }
            for(int b=0;b<B;++b) for(int h=0;h<H;++h) for(int i=0;i<T;++i) {
                float* dst=out+((b*H+h)*T+i)*T;
                const float* table=n.packed.data()+h*stride;
                const int* index=indices+(b*T+i)*T;
                for(int j=0;j<T;++j) dst[j]=table[index[j]];
                if(n.inputs.size()==7) {
                    float target=at(5).data[b*T+i];
                    if(target>=0 && target<T && target==int(target)) dst[int(target)]+=at(6).data[h];
                }
            }
            break;
        }
        for(int b=0;b<B;++b) for(int i=0;i<T;++i) {
            const float* left=x+(b*T+i)*3;
            for(int j=0;j<T;++j) {
                const float* right=x+(b*T+j)*3;
                int dr=left[1]<0 || right[1]<0 ? 2*R+1 : int(std::clamp(left[1]-right[1],float(-R),float(R)))+R;
                int dc=left[2]<0 || right[2]<0 ? 2*C+1 : int(std::clamp(left[2]-right[2],float(-C),float(C)))+C;
                bool same=left[1]>=0 && left[2]>=0 && left[1]==right[1] && left[2]==right[2];
                for(int h=0;h<H;++h) out[((b*H+h)*T+i)*T+j]=
                    at(1).data[(h*K+int(left[0]))*K+int(right[0])] + at(2).data[h*(2*R+2)+dr] +
                    at(3).data[h*(2*C+2)+dc] + (at(4).data.size()==size_t(H) ? (same ? at(4).data[h] : 0.f) : at(4).data[h*2+int(same)]) +
                    (n.inputs.size()==7 && at(5).data[b*T+i]==j ? at(6).data[h] : 0.f);
            }
        } break;
    }
    case 20: TokenSilu(out,x,1,count,false); break;
    case 21: {
        int B=a.shape[0],T=a.shape[1],D=a.shape[2],M=at(1).shape[1];
        for(int b=0;b<B;++b) for(int m=0;m<M;++m) {
            float id=std::max(0.f,at(1).data[b*M+m]);
            Require(std::isfinite(id) && id<T && id==std::floor(id), "gather index");
            std::copy_n(x+(b*T+int(id))*D,D,out+(b*M+m)*D);
        } break;
    }
    case 22: {
        int B=a.shape[0],T=a.shape[1],D=a.shape[2];
        std::fill_n(out,y.elements,0.f);
        for(int b=0;b<B;++b) {
            float total=0;
            for(int t=0;t<T;++t) {
                float w=at(1).data[b*T+t]; total+=w;
                for(int d=0;d<D;++d) out[b*D+d]+=x[(b*T+t)*D+d]*w;
            }
            for(int d=0;d<D;++d) out[b*D+d]/=std::max(1.f,total);
        } break;
    }
    case 24: {
        int B=a.shape[0],C=a.shape[1],H=a.shape[2],W=a.shape[3];
        float* rows=n.scratch.data(); float* projection=rows+B*H*2*C;
        for(int b=0;b<B;++b) for(int h=0;h<H;++h) for(int c=0;c<C;++c) {
            const float* values=x+((b*C+c)*H+h)*W;
            float sum=0,maximum=-std::numeric_limits<float>::infinity();
            for(int w=0;w<W;++w) { sum+=values[w]; maximum=std::max(maximum,values[w]); }
            rows[(b*H+h)*2*C+c]=sum/W; rows[(b*H+h)*2*C+C+c]=maximum;
        }
        Linear(rows,at(1),&at(2),projection,B*H);
        for(int b=0;b<B;++b) for(int c=0;c<C;++c) for(int h=0;h<H;++h)
            for(int w=0;w<W;++w) {
                int at=((b*C+c)*H+h)*W+w;
                out[at]=std::max(0.f,x[at]+projection[(b*H+h)*C+c]);
            }
        break;
    }
    }
}
}
