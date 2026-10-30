#pragma once
#include "GenericForward.h"
namespace pvz::net {
void PrepareSparseConvolution(Graph& graph, Node& node);
bool ExecuteSparseConvolution(Node& node);
void SparseToHost(Tensor& tensor);
float* SparseChannelsData(Tensor& tensor);
void SparseFromHost(Tensor& tensor);
}
