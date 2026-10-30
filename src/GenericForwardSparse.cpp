#include "GenericForwardSparse.h"
#include "GenericForwardDnnl.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>

namespace pvz::net {
struct ChannelTensor {
    int batch,channels,cells;
    float* host;
    std::vector<float> values;
    bool dirty=false;
    explicit ChannelTensor(Tensor& t) : batch(t.shape[0]),channels(t.shape[1]),cells(t.shape[2]*t.shape[3]),
        host(t.data.data()),values(t.data.size()) {}
    void ToHost() {
        if(!dirty) return;
        for(int b=0;b<batch;++b) for(int c=0;c<channels;++c) for(int s=0;s<cells;++s)
            host[(b*channels+c)*cells+s]=values[(b*cells+s)*channels+c];
        dirty=false;
    }
    void FromHost() {
        for(int b=0;b<batch;++b) for(int c=0;c<channels;++c) for(int s=0;s<cells;++s)
            values[(b*cells+s)*channels+c]=host[(b*channels+c)*cells+s];
        dirty=false;
    }
};
struct SparseConvolution {
    int B,C,O,H,W,S,P;
    bool relu;
    const float* input;
    Tensor* inputTensor;
    std::shared_ptr<ChannelTensor> source,output;
    std::vector<float> kernel,bias,values;
    std::vector<int> indices,offsets,neighbors;
    explicit SparseConvolution(Graph& g,Node& n) {
        auto& a=g.tensors[n.inputs[0]]; auto& w=g.tensors[n.inputs[1]]; auto& y=g.tensors[n.output];
        B=a.shape[0]; C=a.shape[1]; H=a.shape[2]; W=a.shape[3]; O=w.shape[0];
        int KH=w.shape[2],KW=w.shape[3]; P=KH*KW; S=y.shape[2]*y.shape[3];
        relu=n.op=="conv_relu"; input=a.data.data(); inputTensor=&a; source=a.channelsLast;
        output=std::make_shared<ChannelTensor>(y); y.channelsLast=output;
        kernel.resize(w.data.size()); bias.resize(O,0.f);
        if(n.inputs.size()==3) std::copy_n(g.tensors[n.inputs[2]].data.data(),O,bias.data());
        for(int k=0;k<P;++k) for(int c=0;c<C;++c) for(int o=0;o<O;++o)
            kernel[(k*C+c)*O+o]=w.data[(o*C+c)*P+k];
        indices.resize(H*W*C); values.resize(H*W*C); offsets.resize(H*W+1);
        neighbors.resize(S*P,-1);
        for(int s=0;s<S;++s) for(int kh=0;kh<KH;++kh) for(int kw=0;kw<KW;++kw) {
            int h=s/y.shape[3]*n.ints[2]+kh-n.ints[0],col=s%y.shape[3]*n.ints[3]+kw-n.ints[1];
            if(h>=0 && h<H && col>=0 && col<W) neighbors[s*P+kh*KW+kw]=h*W+col;
        }
    }
    template<int Blocks> __attribute__((target("avx2,fma")))
    void Compute(int b,int startChannel) {
        for(int s=0;s<S;++s) {
            __m256 sum[Blocks];
            for(int i=0;i<Blocks;++i) sum[i]=_mm256_loadu_ps(bias.data()+startChannel+i*8);
            for(int k=0;k<P;++k) {
                int cell=neighbors[s*P+k]; if(cell<0) continue;
                const float* w=kernel.data()+k*C*O+startChannel;
                for(int p=offsets[cell];p<offsets[cell+1];++p) {
                    const float* row=w+indices[p];
                    __m256 x=_mm256_set1_ps(values[p]);
                    for(int i=0;i<Blocks;++i) sum[i]=_mm256_fmadd_ps(x,_mm256_loadu_ps(row+i*8),sum[i]);
                }
            }
            float* dest=output->values.data()+(b*S+s)*O+startChannel;
            for(int i=0;i<Blocks;++i) _mm256_storeu_ps(dest+i*8,relu ? _mm256_max_ps(sum[i],_mm256_setzero_ps()) : sum[i]);
        }
    }
    bool Run() {
        if(!source) NativeToHost(*inputTensor);
        const int batch=B, channels=C, cells=H*W, outputs=O;
        int* index=indices.data(); int* offset=offsets.data(); float* valueOut=values.data();
        const float* src=source ? source->values.data() : input;
        for(int b=0;b<batch;++b) {
            int count=0;
            for(int s=0;s<cells;++s) {
                offset[s]=count;
                for(int c=0;c<channels;++c) {
                    float value=source ? src[(b*cells+s)*channels+c] : src[(b*channels+c)*cells+s];
                    if(value!=0.f) { index[count]=c*outputs; valueOut[count++]=value; }
                }
            }
            offset[cells]=count;
            // At high density BLAS is faster than indexed FMA. Reuse its prepared buffers.
            if(count>cells*channels*3/4) return false;
            for(int c=0;c<O;c+=64) switch(std::min(64,O-c)/8) {
                case 8: Compute<8>(b,c);break; case 7: Compute<7>(b,c);break;
                case 6: Compute<6>(b,c);break; case 5: Compute<5>(b,c);break;
                case 4: Compute<4>(b,c);break; case 3: Compute<3>(b,c);break;
                case 2: Compute<2>(b,c);break; case 1: Compute<1>(b,c);break;
            }
        }
        output->dirty=true;
        return true;
    }
};
void PrepareSparseConvolution(Graph& g,Node& n) {
    if(n.op!="conv" && n.op!="conv_relu") return;
    if(!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma")) return;
    const auto& a=g.tensors[n.inputs[0]]; const auto& w=g.tensors[n.inputs[1]]; const auto& y=g.tensors[n.output];
    if(n.ints.size()!=5 || w.role!=2 || w.shape[0]<16 || w.shape[0]%8 || a.shape[1]<8 || y.shape[2]*y.shape[3]<16 || a.shape[0]>8) return;
    n.sparse=std::make_shared<SparseConvolution>(g,n);
}
bool ExecuteSparseConvolution(Node& n) {
    if(!n.sparse) return false;
    return n.sparse->Run();
}
void SparseToHost(Tensor& t) { if(t.channelsLast) t.channelsLast->ToHost(); }
float* SparseChannelsData(Tensor& t) { return t.channelsLast ? t.channelsLast->values.data() : nullptr; }
void SparseFromHost(Tensor& t) { if(t.channelsLast) t.channelsLast->FromHost(); }
}
#else
namespace pvz::net {
void PrepareSparseConvolution(Graph&,Node&) {}
bool ExecuteSparseConvolution(Node&) { return false; }
void SparseToHost(Tensor&) {}
float* SparseChannelsData(Tensor&) { return nullptr; }
void SparseFromHost(Tensor&) {}
}
#endif
