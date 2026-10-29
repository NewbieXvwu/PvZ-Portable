#include "GenericForwardDnnl.h"
#include "GenericForwardSparse.h"
#ifdef PVZ_GENERIC_DNNL
#include <oneapi/dnnl/dnnl.hpp>
#include <omp.h>
#include <array>
#include <stdexcept>
namespace pvz::net {
using namespace dnnl;
static void Execute(const primitive& op, const stream& stream, const dnnl_exec_arg_t* args, int count) {
    if(dnnl_primitive_execute(op.get(),stream.get(),count,args)!=dnnl_success)
        throw std::runtime_error("generic forward: oneDNN execution failed");
}
struct NativeTensor {
    engine eng;
    stream queue;
    memory value, host;
    reorder toHost;
    std::array<dnnl_exec_arg_t,2> hostArgs;
    bool dirty=false;
    NativeTensor(const engine& e, const memory::desc& desc, Tensor& t)
        : eng(e),queue(e),value(desc,e),
          host(memory::desc(memory::dims(t.shape.begin(),t.shape.end()),memory::data_type::f32,memory::format_tag::nchw),e,t.data.data()),
          toHost(value,host),hostArgs{{{DNNL_ARG_FROM,value.get()},{DNNL_ARG_TO,host.get()}}} {}
};
struct NativeConvolution {
    engine eng{engine::kind::cpu,0};
    stream queue{eng};
    memory src,weights,bias,dst,scratch;
    convolution_forward op;
    reorder fromHost;
    std::shared_ptr<NativeTensor> source,output;
    bool fromSparse=false;
    std::vector<dnnl_exec_arg_t> args;
    std::array<dnnl_exec_arg_t,2> sourceArgs{};
    explicit NativeConvolution(Graph& g, Node& n)
        : eng(g.tensors[n.inputs[0]].native ? g.tensors[n.inputs[0]].native->eng : engine(engine::kind::cpu,0)),queue(eng) {
        omp_set_num_threads(1);
        auto& a=g.tensors[n.inputs[0]]; auto& w=g.tensors[n.inputs[1]]; auto& y=g.tensors[n.output];
        auto md=[](const Tensor& t,memory::format_tag tag) {
            return memory::desc(memory::dims(t.shape.begin(),t.shape.end()),memory::data_type::f32,tag);
        };
        auto srcDesc=md(a,memory::format_tag::nhwc), dstDesc=md(y,memory::format_tag::nhwc);
        auto biasDesc=n.inputs.size()==3 ? md(g.tensors[n.inputs[2]],memory::format_tag::a) : memory::desc();
        primitive_attr attr; attr.set_scratchpad_mode(scratchpad_mode::user);
        attr.set_fpmath_mode(fpmath_mode::strict);
        if(n.code==23) { post_ops post; post.append_eltwise(algorithm::eltwise_relu,0,0); attr.set_post_ops(post); }
        convolution_forward::primitive_desc pd(eng,prop_kind::forward_inference,algorithm::convolution_direct,
            srcDesc,md(w,memory::format_tag::any),biasDesc,dstDesc,
            {n.ints[2],n.ints[3]},{n.ints[0],n.ints[1]},{n.ints[0],n.ints[1]},attr);
        op=convolution_forward(pd);
        if(a.native) { source=a.native; src=source->value; }
        else if(float* values=SparseChannelsData(a)) { src=memory(srcDesc,eng,values); fromSparse=true; }
        else {
            auto host=memory(md(a,memory::format_tag::nchw),eng,a.data.data());
            src=memory(srcDesc,eng); fromHost=reorder(host,src);
            sourceArgs={{{DNNL_ARG_FROM,host.get()},{DNNL_ARG_TO,src.get()}}};
            // Keep the source memory handle alive in args; oneDNN memory references raw storage.
            sourceHost=host;
        }
        weights=memory(pd.weights_desc(),eng);
        auto plainWeight=memory(md(w,memory::format_tag::oihw),eng,w.data.data());
        reorder(plainWeight,weights).execute(queue,plainWeight,weights);
        output=std::make_shared<NativeTensor>(eng,dstDesc,y); y.native=output; dst=output->value;
        scratch=memory(pd.scratchpad_desc(),eng);
        args={{DNNL_ARG_SRC,src.get()},{DNNL_ARG_WEIGHTS,weights.get()},
              {DNNL_ARG_DST,dst.get()},{DNNL_ARG_SCRATCHPAD,scratch.get()}};
        if(n.inputs.size()==3) { bias=memory(biasDesc,eng,g.tensors[n.inputs[2]].data.data()); args.push_back({DNNL_ARG_BIAS,bias.get()}); }
        queue.wait();
    }
    memory sourceHost;
    void Run() {
        if(!source && !fromSparse) Execute(fromHost,queue,sourceArgs.data(),sourceArgs.size());
        Execute(op,queue,args.data(),args.size()); output->dirty=true;
    }
};
void PrepareNativeConvolution(Graph& g,Node& n) {
    if(n.code!=1 && n.code!=23) return;
    const auto& a=g.tensors[n.inputs[0]]; const auto& w=g.tensors[n.inputs[1]];
    // Sparse first layers and tiny heads retain the portable kernels.
    if(n.ints.size()!=4 || a.shape[1]<16 || w.shape[0]<16 || g.tensors[n.output].shape[2]*g.tensors[n.output].shape[3]<16 || w.role!=2) return;
    n.native=std::make_shared<NativeConvolution>(g,n);
}
bool ExecuteNativeConvolution(Node& n) {
    if(!n.native) return false;
    n.native->Run(); return true;
}
void NativeToHost(Tensor& t) {
    if(t.native && t.native->dirty) {
        Execute(t.native->toHost,t.native->queue,t.native->hostArgs.data(),t.native->hostArgs.size());
        t.native->queue.wait(); t.native->dirty=false;
    }
}
}
#else
namespace pvz::net {
void PrepareNativeConvolution(Graph&,Node&) {}
bool ExecuteNativeConvolution(Node&) { return false; }
void NativeToHost(Tensor&) {}
}
#endif
