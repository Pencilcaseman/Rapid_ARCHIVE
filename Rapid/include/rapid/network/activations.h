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

namespace rapid
{
	template<typename networkType>
	struct NetworkActivation
	{
		std::string name;
		std::function<networkType(networkType)> activation;
		std::function<networkType(networkType)> derivative;
	};

	template<typename activationType>
	class Activations
	{
	public:

		/*
		NetworkActivation<activationType> sigmoid;
		NetworkActivation<activationType> tanh;
		NetworkActivation<activationType> relu;
		NetworkActivation<activationType> leakyRelu;

		Activations()
		{
			std::function<double(double)> _sigmoidActivation = [](double x)
			{
				return SIGMOID(x);
			};

			std::function<double(double)> _tanhActivation = [](double x)
			{
				return TANH(x);
			};

			std::function<double(double)> _reluActivation = [](double x)
			{
				return RELU(x);
			};

			std::function<double(double)> _leakyReluActivation = [](double x)
			{
				return LEAKY_RELU(x);
			};

			std::function<double(double)> _sigmoidDerivative = [](double x)
			{
				return D_SIGMOID(x);
			};

			std::function<double(double)> _tanhDerivative = [](double x)
			{
				return D_TANH(x);
			};

			std::function<double(double)> _reluDerivative = [](double x)
			{
				return D_RELU(x);
			};

			std::function<double(double)> _leakyReluDerivative = [](double x)
			{
				return D_LEAKY_RELU(x);
			};

			sigmoid.activation = _sigmoidActivation;
			tanh.activation = _tanhActivation;
			relu.activation = _reluActivation;
			leakyRelu.activation = _leakyReluActivation;

			sigmoid.derivative = _sigmoidDerivative;
			tanh.derivative = _tanhDerivative;
			relu.derivative = _reluDerivative;
			leakyRelu.derivative = _leakyReluDerivative;

			sigmoid.name = "sigmoid";
			tanh.name = "tanh";
			relu.name = "relu";
			leakyRelu.name = "leaky relu";
		}
		*/

		inline static NetworkActivation<activationType> sigmoid()
		{
			static NetworkActivation<activationType> res;

			static std::function<double(double)> _sigmoidActivation = [](double x)
			{
				return SIGMOID(x);
			};

			static std::function<double(double)> _sigmoidDerivative = [](double x)
			{
				return D_SIGMOID(x);
			};

			return {"sigmoid", _sigmoidActivation, _sigmoidDerivative};
		}

		inline static NetworkActivation<activationType> tanh()
		{
			static NetworkActivation<activationType> res;

			static std::function<double(double)> _tanhActivation = [](double x)
			{
				return TANH(x);
			};

			static std::function<double(double)> _tanhDerivative = [](double x)
			{
				return D_TANH(x);
			};

			return {"tanh", _tanhActivation, _tanhDerivative};
		}

		inline static NetworkActivation<activationType> relu()
		{
			static NetworkActivation<activationType> res;

			static std::function<double(double)> _reluActivation = [](double x)
			{
				return RELU(x);
			};

			static std::function<double(double)> _reluDerivative = [](double x)
			{
				return D_RELU(x);
			};

			return {"relu", _reluActivation, _reluDerivative};
		}

		inline static NetworkActivation<activationType> leakyRelu()
		{
			static NetworkActivation<activationType> res;

			static std::function<double(double)> _leakyReluActivation = [](double x)
			{
				return LEAKY_RELU(x);
			};

			static std::function<double(double)> _leakyReluDerivative = [](double x)
			{
				return D_LEAKY_RELU(x);
			};

			return {"sigmoid", _leakyReluActivation, _leakyReluDerivative};
		}
	};
}
