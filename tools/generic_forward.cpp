// Standalone parity/benchmark driver; never linked into the production engine.
#include "GenericForward.h"
#include <chrono>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>
int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::runtime_error("usage: generic-forward model.bin inputs.f32 outputs.f32 [iterations]");
        pvz::net::Graph graph; graph.Load(argv[1]);
        std::ifstream in(argv[2], std::ios::binary);
        for (auto& t : graph.tensors) if(t.role == 1) {
            in.read(reinterpret_cast<char*>(t.data.data()), t.data.size()*4);
            if(!in) throw std::runtime_error("truncated input");
        }
        graph.Run();
        graph.Materialize();
        std::ofstream out(argv[3], std::ios::binary);
        for (const auto& n : graph.nodes) {
            const auto& t=graph.tensors[n.output];
            out.write(reinterpret_cast<const char*>(t.data.data()),t.data.size()*4);
        }
        int iterations=argc>4 ? std::stoi(argv[4]) : 1;
        auto start=std::chrono::steady_clock::now();
        auto cpuStart=std::clock();
        for(int i=0;i<iterations;++i) graph.Run();
        double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/iterations;
        double cpuUs=double(std::clock()-cpuStart)*1000000/CLOCKS_PER_SEC/iterations;
        std::cout << "{\"us\":" << us << ",\"cpu_us\":" << cpuUs << ",\"iterations\":" << iterations;
        if(argc>5 && std::string(argv[5])=="profile") {
            auto times=graph.Profile(iterations);
            std::cout<<",\"profile\":[";
            for(size_t i=0;i<times.size();++i) std::cout<<(i ? "," : "")<<"{\"node\":\""<<graph.tensors[graph.nodes[i].output].name
                <<"\",\"op\":\""<<graph.nodes[i].op<<"\",\"us\":"<<times[i]<<'}';
            std::cout<<']';
        }
        std::cout << "}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
