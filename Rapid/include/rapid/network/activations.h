#pragma once

#include "../internal.h"

#define SIGMOID(x) (1 / (1 + exp((-(x)))))
#define TANH(x) (std::tanh((x)))
#define RELU(x) ((x) > 0 ? (x) : 0)
#define LEAKY_RELU(x) ((x) > 0 ? (x) : ((x) * 0.2))

#define D_SIGMOID(y) ((y) * (1 - (y)))
#define D_TANH(y) (1 - ((y) * (y)))
#define D_RELU(y) ((y) > 0 ? 1 : 0)
#define D_LEAKY_RELU(y) ((y) > 0 ? 1 : 0.2)
