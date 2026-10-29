#pragma once
#include "GenericForward.h"

namespace pvz::net {
void PreparePackedLinear(Tensor& weight);
bool PackedMatrix(const Tensor& weight, const float* x, const float* bias, float* y, int rows);
void TokenSoftmax(float* scores, const float* bias, const float* mask, int tokens, bool queryValid);
void TokenSilu(float* output, const float* input, int rows, int width, bool gated);
}
