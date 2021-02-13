#pragma once

#include "../../internal.h"

namespace rapid
{
	template<typename layerType>
	class deep : public Layer<layerType>
	{
	public:
		size_t nodes = 0;
		layerType learningRate = 0.01;
		std::function<layerType(layerType)> activation;
		std::function<layerType(layerType)> derivative;
		Matrix<layerType> weights;
		Matrix<layerType> bias;
		std::string initializer = "random";

		Matrix<layerType> previousInput;
		Matrix<layerType> previousOutput;

		deep(size_t size)
		{
			nodes = size;

			activation = Activations<layerType>::tanh().activation;
			derivative = Activations<layerType>::tanh().derivative;
		}

		deep(size_t size, layerType lr)
		{
			nodes = size;
			learningRate = lr;

			activation = Activations<layerType>::tanh().activation;
			derivative = Activations<layerType>::tanh().derivative;
		}

		deep(size_t size, layerType lr, const NetworkActivation<layerType> &activationFunction)
		{
			nodes = size;
			learningRate = lr;
			activation = activationFunction.activation;
			derivative = activationFunction.derivative;

			if (activationFunction.name == "relu" || activationFunction.name == "leaky relu")
				initializer = "he";
			else if (activationFunction.name == "tanh")
				initializer = "xavier";
		}

		deep(size_t size, layerType lr, const std::function<layerType(layerType)> &activationFunction, const std::function<layerType(layerType)> &derivativeFunction, const std::string &initializerMode)
		{
			nodes = size;
			learningRate = lr;
			activation = activationFunction;
			derivative = derivativeFunction;
			initializer = initializerMode;
		}

		deep(size_t size, layerType lr, const std::string &activationName)
		{
			nodes = size;
			learningRate = lr;
			
			if (activationName == "sigmoid")
			{
				activation = Activations<layerType>::sigmoid().activation;
				derivative = Activations<layerType>::sigmoid().derivative;
			}
			else if (activationName == "tanh")
			{
				activation = Activations<layerType>::tanh().activation;
				derivative = Activations<layerType>::tanh().derivative;
				initializer = "xavier";
			}
			else if (activationName == "relu")
			{
				activation = Activations<layerType>::relu().activation;
				derivative = Activations<layerType>::relu().derivative;
				initializer = "he";
			}
			else if (activationName == "leaky relu")
			{
				activation = Activations<layerType>::leakyRelu().activation;
				derivative = Activations<layerType>::leakyRelu().derivative;
				initializer = "he";
			}
			else
				RapidError("Network Error", "Invalid activation name. Pass function and derivative instead").display();
		}

		deep(size_t size, layerType lr, const std::string &activationName, const std::string &initializerMode)
		{
			nodes = size;
			learningRate = lr;

			if (activationName == "sigmoid")
			{
				activation = Activations<layerType>::sigmoid().activation;
				derivative = Activations<layerType>::sigmoid().derivative;
			}
			else if (activationName == "tanh")
			{
				activation = Activations<layerType>::tanh().activation;
				derivative = Activations<layerType>::tanh().derivative;
			}
			else if (activationName == "relu")
			{
				activation = Activations<layerType>::relu().activation;
				derivative = Activations<layerType>::relu().derivative;
			}
			else if (activationName == "leaky relu")
			{
				activation = Activations<layerType>::leakyRelu().activation;
				derivative = Activations<layerType>::leakyRelu().derivative;
			}
			else
				RapidError("Network Error", "Invalid activation name. Pass function and derivative instead").display();

			initializer = initializerMode;
		}

		void compile(size_t nodesBefore, size_t nodesAfter) override
		{
			if (initializer == "zeros" || initializer == "zero")
			{
				weights = Matrix<layerType>(nodes, nodesBefore, 0);
				bias = Matrix<layerType>(nodes, 1, 0);
			}
			else if (initializer == "random")
			{
				weights = Matrix<layerType>::random(nodes, nodesBefore);
				bias = Matrix<layerType>(nodes, 1, 0);
			}
			else if (initializer == "inverse root")
			{
				auto range = std::pow(nodes, -0.5);
				weights = Matrix<layerType>::random(nodes, nodesBefore, -range, range);
				bias = Matrix<layerType>::random(nodes, 1, -range, range);
			}
			else if (initializer == "HE" || initializer == "He" || initializer == "he")
			{
				weights = Matrix<layerType>::random(nodes, nodesBefore) * sqrt(2. / (layerType) nodesBefore);
				bias = Matrix<layerType>(nodes, 1, 0);
			}
			else if (initializer == "XAVIER" || initializer == "Xavier" || initializer == "xavier")
			{
				weights = Matrix<layerType>::random(nodes, nodesBefore) * sqrt(1. / (layerType) nodesBefore);
				bias = Matrix<layerType>(nodes, 1, 0);
			}
			else
				RapidError("Network Error", "Invalid initializer").display();
		}

		Matrix<layerType> feedForward(const Matrix<layerType> &input, bool saveLayerdata) override
		{
			if (!saveLayerdata)
				return (weights.dot(input) + bias).mapped(activation);
			
			auto res = (weights.dot(input) + bias).mapped(activation);

			previousInput = input;
			previousOutput = res;

			return res;
		}

		Matrix<layerType> backpropagate(const Matrix<layerType> &error) override
		{
			auto gradient = previousOutput.mapped(derivative);
			gradient *= error;
			gradient *= learningRate;

			auto transposed = previousInput.transposed();

			auto weightDelta = gradient.dot(transposed);

			weights += weightDelta;
			bias += gradient;

			return weights.transposed().dot(error);
		}

		size_t getSize() override
		{
			return nodes;
		}

		void setLearningRate(layerType newLr) override
		{
			learningRate = newLr;
		}

		layerType getLearningRate() override
		{
			return learningRate;
		}
	};
}
