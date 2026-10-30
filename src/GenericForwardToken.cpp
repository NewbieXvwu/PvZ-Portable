#include "GenericForwardToken.h"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string_view>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define TOKEN_AVX __attribute__((target("avx2,fma")))
#define TOKEN_X86 1
#endif

namespace pvz::net {
struct PackedLinear {
    int input, output, precision=0;
    std::vector<float> values, scales;
    std::vector<uint16_t> bf16;
    std::vector<int8_t> int8;
};
namespace {
bool VectorCPU() {
#ifdef TOKEN_X86
    static bool available=__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    return available;
#else
    return false;
#endif
}
#ifdef TOKEN_X86
template<int Precision> TOKEN_AVX __m256 Weight8(const PackedLinear& w,size_t index) {
    if constexpr(Precision==0) return _mm256_loadu_ps(w.values.data()+index);
    if constexpr(Precision==1) return _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w.bf16.data()+index))),16));
    if constexpr(Precision==2) return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w.int8.data()+index))));
}
template<int Rows,int Precision> TOKEN_AVX void MatrixTile(const PackedLinear& w,const float* x,const float* bias,float* y,int c) {
    __m256 accum[Rows][2];
    for(int r=0;r<Rows;++r) for(int j=0;j<2;++j) accum[r][j]=_mm256_setzero_ps();
    size_t packed=size_t(c/16)*w.input*16;
    for(int k=0;k<w.input;++k) {
        __m256 a=Weight8<Precision>(w,packed+k*16), b=Weight8<Precision>(w,packed+k*16+8);
        for(int r=0;r<Rows;++r) {
            __m256 v=_mm256_set1_ps(x[r*w.input+k]);
            accum[r][0]=_mm256_fmadd_ps(v,a,accum[r][0]);
            accum[r][1]=_mm256_fmadd_ps(v,b,accum[r][1]);
        }
    }
    int tail=w.output-c;
    for(int r=0;r<Rows;++r) for(int j=0;j<2 && j*8<tail;++j) {
        const __m256i mask=_mm256_cmpgt_epi32(_mm256_set1_epi32(tail-j*8),_mm256_setr_epi32(0,1,2,3,4,5,6,7));
        __m256 v=accum[r][j];
        if constexpr(Precision==2) v=_mm256_mul_ps(v,_mm256_maskload_ps(w.scales.data()+c+j*8,mask));
        if(bias) v=_mm256_add_ps(v,_mm256_maskload_ps(bias+c+j*8,mask));
        _mm256_maskstore_ps(y+r*w.output+c+j*8,mask,v);
    }
}
// Range reduction + degree-7 Taylor polynomial on [-ln2/2,ln2/2].
// Max relative error < 3e-7 over the finite normal range; no global fast-math.
TOKEN_AVX __m256 Exp8(__m256 value) {
    value=_mm256_max_ps(_mm256_set1_ps(-87.f),_mm256_min_ps(_mm256_set1_ps(87.f),value));
    __m256 n=_mm256_round_ps(_mm256_mul_ps(value,_mm256_set1_ps(1.4426950408889634f)),_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    __m256 r=_mm256_fnmadd_ps(n,_mm256_set1_ps(0.693145751953125f),value);
    r=_mm256_fnmadd_ps(n,_mm256_set1_ps(1.428606765330187e-6f),r);
    __m256 p=_mm256_set1_ps(1.f/5040.f);
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/720.f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/120.f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/24.f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f/6.f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(.5f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f));
    p=_mm256_fmadd_ps(p,r,_mm256_set1_ps(1.f));
    __m256 scale=_mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n),_mm256_set1_epi32(127)),23));
    return _mm256_mul_ps(p,scale);
}
TOKEN_AVX void Softmax8(float* s,const float* bias,const float* mask,int T) {
    float maximum=-INFINITY;
    __m256 high=_mm256_set1_ps(-INFINITY);
    int j=0;
    for(;j+8<=T;j+=8) {
        __m256 v=_mm256_add_ps(_mm256_loadu_ps(s+j),_mm256_loadu_ps(bias+j));
        __m256 valid=_mm256_cmp_ps(_mm256_loadu_ps(mask+j),_mm256_setzero_ps(),_CMP_GT_OQ);
        v=_mm256_blendv_ps(_mm256_set1_ps(-INFINITY),v,valid);
        _mm256_storeu_ps(s+j,v); high=_mm256_max_ps(high,v);
    }
    float tmp[8]; _mm256_storeu_ps(tmp,high);
    for(float v:tmp) maximum=std::max(maximum,v);
    for(;j<T;++j) { s[j]=mask[j]>0 ? s[j]+bias[j] : -INFINITY; maximum=std::max(maximum,s[j]); }
    if(!std::isfinite(maximum)) { std::fill_n(s,T,0.f); return; }
    __m256 total=_mm256_setzero_ps(); j=0;
    for(;j+8<=T;j+=8) {
        __m256 v=_mm256_sub_ps(_mm256_loadu_ps(s+j),_mm256_set1_ps(maximum));
        __m256 e=_mm256_and_ps(Exp8(v),_mm256_cmp_ps(v,_mm256_set1_ps(-87.f),_CMP_GT_OQ));
        total=_mm256_add_ps(total,e); _mm256_storeu_ps(s+j,e);
    }
    _mm256_storeu_ps(tmp,total); float sum=0;
    for(float v:tmp) sum+=v;
    for(;j<T;++j) { s[j]=std::exp(s[j]-maximum); sum+=s[j]; }
    __m256 inv=_mm256_set1_ps(1.f/sum); j=0;
    for(;j+8<=T;j+=8) _mm256_storeu_ps(s+j,_mm256_mul_ps(_mm256_loadu_ps(s+j),inv));
    for(;j<T;++j) s[j]/=sum;
}
TOKEN_AVX void Silu8(float* y,const float* x,int rows,int width,bool gated) {
    for(int r=0;r<rows;++r) {
        const float* src=x+r*width*(gated ? 2 : 1); float* dst=y+r*width;
        int i=0;
        for(;i+8<=width;i+=8) {
            __m256 v=_mm256_loadu_ps(src+i);
            __m256 e=Exp8(_mm256_sub_ps(_mm256_setzero_ps(),v));
            __m256 result=_mm256_div_ps(v,_mm256_add_ps(_mm256_set1_ps(1.f),e));
            if(gated) result=_mm256_mul_ps(result,_mm256_loadu_ps(src+width+i));
            _mm256_storeu_ps(dst+i,result);
        }
        for(;i<width;++i) dst[i]=src[i]/(1.f+std::exp(-src[i]))*(gated ? src[width+i] : 1.f);
    }
}
#endif
}
void PreparePackedLinear(Tensor& w) {
    if(w.linear || w.role!=2 || w.shape.size()!=2 || w.shape[0]<16 || w.shape[1]<24 || !VectorCPU()) return;
    const char* mode=std::getenv("PVZ_GENERIC_LINEAR");
    if(mode && std::string_view(mode)=="blas") return;
    auto p=std::make_shared<PackedLinear>(); p->input=w.shape[1]; p->output=w.shape[0];
    const char* precision=std::getenv("PVZ_GENERIC_PRECISION");
    if(precision && std::string_view(precision)=="bf16") p->precision=1;
    if(precision && std::string_view(precision)=="int8") p->precision=2;
    size_t size=size_t((p->output+15)/16)*p->input*16;
    if(p->precision==0) p->values.resize(size);
    if(p->precision==1) p->bf16.resize(size);
    if(p->precision==2) {
        p->int8.resize(size); p->scales.resize(p->output);
        for(int c=0;c<p->output;++c) {
            float high=0; for(int k=0;k<p->input;++k) high=std::max(high,std::abs(w.data[c*p->input+k]));
            p->scales[c]=high>0 ? high/127.f : 1.f;
        }
    }
    for(int c=0;c<p->output;++c) for(int k=0;k<p->input;++k) {
        size_t index=(size_t(c/16)*p->input+k)*16+c%16;
        float v=w.data[c*p->input+k];
        if(p->precision==0) p->values[index]=v;
        if(p->precision==1) {
            uint32_t bits=std::bit_cast<uint32_t>(v);
            p->bf16[index]=uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
        }
        if(p->precision==2) p->int8[index]=int8_t(std::clamp(std::nearbyint(v/p->scales[c]),-127.f,127.f));
    }
    w.linear=std::move(p);
}
bool PackedMatrix(const Tensor& w,const float* x,const float* bias,float* y,int rows) {
#ifdef TOKEN_X86
    if(!w.linear || rows<4) return false;
    const auto& p=*w.linear;
    auto compute=[&]<int Precision>() {
    for(int c=0;c<p.output;c+=16) for(int r=0;r<rows;r+=6) {
        const float* src=x+r*p.input; float* dst=y+r*p.output;
        switch(std::min(6,rows-r)) {
        case 6: MatrixTile<6,Precision>(p,src,bias,dst,c); break;
        case 5: MatrixTile<5,Precision>(p,src,bias,dst,c); break;
        case 4: MatrixTile<4,Precision>(p,src,bias,dst,c); break;
        case 3: MatrixTile<3,Precision>(p,src,bias,dst,c); break;
        case 2: MatrixTile<2,Precision>(p,src,bias,dst,c); break;
        case 1: MatrixTile<1,Precision>(p,src,bias,dst,c); break;
        }
    }
    };
    if(p.precision==0) compute.template operator()<0>();
    else if(p.precision==1) compute.template operator()<1>();
    else compute.template operator()<2>();
    return true;
#else
    return false;
#endif
}
void TokenSoftmax(float* scores,const float* bias,const float* mask,int T,bool queryValid) {
    if(!queryValid) { std::fill_n(scores,T,0.f); return; }
#ifdef TOKEN_X86
    if(VectorCPU()) { Softmax8(scores,bias,mask,T); return; }
#endif
    float high=-INFINITY,total=0;
    for(int j=0;j<T;++j) { scores[j]=mask[j]>0 ? scores[j]+bias[j] : -INFINITY; high=std::max(high,scores[j]); }
    for(int j=0;j<T;++j) { scores[j]=std::isfinite(high) ? std::exp(scores[j]-high) : 0.f; total+=scores[j]; }
    for(int j=0;j<T;++j) scores[j]=total>0 ? scores[j]/total : 0.f;
}
void TokenSilu(float* y,const float* x,int rows,int width,bool gated) {
#ifdef TOKEN_X86
    if(VectorCPU()) { Silu8(y,x,rows,width,gated); return; }
#endif
    for(int r=0;r<rows;++r) for(int c=0;c<width;++c) {
        const float* src=x+r*width*(gated ? 2 : 1);
        y[r*width+c]=src[c]/(1.f+std::exp(-src[c]))*(gated ? src[width+c] : 1.f);
    }
}
}
