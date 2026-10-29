#pragma once
#include "GenericForward.h"
namespace pvz::net {
void PrepareNativeConvolution(Graph& graph, Node& node);
bool ExecuteNativeConvolution(Node& node);
void NativeToHost(Tensor& tensor);
}
