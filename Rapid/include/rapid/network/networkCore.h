#pragma once

#include "../internal.h"
#include "activations.h"
#include "layers/layerBase.h"
#include "layers/deepLayer.h"
#include "layers/lstmLayer.h"

namespace rapid
{
	template<typename networkType>
	struct NetworkData
	{
		Matrix<networkType> input;
		Matrix<networkType> output;
	};

	class input
	{
	public:
		size_t nodes;

		input(size_t networkNodes) : nodes(networkNodes)
		{}
	};

	template<typename t>
	std::vector<Matrix<t>> toCategorical(const std::vector<size_t> &data, size_t numClasses)
	{
		std::vector<Matrix<t>> res;

		for (const auto &val : data)
		{
			rapidAssert(val < numClasses, "Class exceeds number of classes specified");

			auto tmp = Matrix<t>::zeros(numClasses, 1);

			tmp[val][0] = 1;

			res.emplace_back(tmp);
		}

		return res;
	}

	template<typename networkType>
	class Network
	{
	public:
		std::vector<std::shared_ptr<Layer<networkType>>> layers;
		std::vector<size_t> nodes;

		bool compiled = false;

		std::vector<NetworkData<networkType>> trainingData;
		size_t epochs = 1;
		size_t batchSize = 1;
		networkType decayRate = 0;

		networkType currentLoss = 0;

		bool verbose = false;

		Network() = default;

		template<typename layerType>
		void add(const layerType &layer)
		{
			static_assert(false, "Invalid network layer type");
		}

		template<>
		void add(const input &layer)
		{
			if (nodes.empty())
				nodes.emplace_back(layer.nodes);
			else
				RapidError("Network Error", "Input layer can only be added as the initial layer").display();
		}
		
		template<>
		void add(const deep<networkType> &layer)
		{
			if (!nodes.empty())
				layers.emplace_back(std::make_shared<deep<networkType>>(layer.nodes, layer.learningRate, layer.activation, layer.derivative, layer.initializer));

			nodes.emplace_back(layer.nodes);
		}

		template<>
		void add(const lstm<networkType> &layer)
		{
			if (!nodes.empty())
				layers.emplace_back(std::make_shared<lstm<networkType>>(layer.outputSize, layer.recurrenceLevel, layer.learningRate));

			nodes.emplace_back(layer.outputSize);
		}

		void compile()
		{
			for (size_t i = 0; i < nodes.size() - 1; i++)
				layers[i]->compile(nodes[i], nodes[i + 1]);

			compiled = true;
		}

		void setLearningRate(networkType lr)
		{
			for (const auto &layer : layers)
				layer->setLearningRate(lr);
		}

		void setEpochs(size_t size)
		{
			epochs = size;
		}

		void setDecayRate(networkType dr)
		{
			decayRate = dr;
		}

		void setBatchSize(size_t size)
		{
			batchSize = size;
		}

		void setTrainingData(const std::vector<NetworkData<networkType>> &newTrainingData)
		{
			trainingData = newTrainingData;
		}

		void setVerbose(bool isVerbose)
		{
			verbose = isVerbose;
		}

		inline Matrix<networkType> feedForward(const Matrix<networkType> &input)
		{
			return feedForward(NetworkData<networkType> {input});
		}

		inline Matrix<networkType> feedForward(const NetworkData<networkType> &input) const
		{
			rapidAssert(compiled, "Network has not yet been compiled");

			Matrix<networkType> current;

			if (input.input.size().rows == nodes[0] && input.input.size().cols == 1) // Input is column vector
				current = input.input;
			else if (input.input.size().cols == nodes[0] && input.input.size().rows == 1) // Input is a vector, so transpose it
				current = input.input.transposed();
			else
				RapidError("Network Error", "Invalid shape for input data").display();

			for (const auto &layer : layers)
				current = layer->feedForward(current, false);

			return current;
		}

		inline void backpropagate(const NetworkData<networkType> &data)
		{
			rapidAssert(compiled, "Network has not yet been compiled");

			Matrix<networkType> input, output;

			// Correct input
			if (data.input.size().rows == nodes[0] && data.input.size().cols == 1) // Data is a column vector
				input = data.input;
			else if (data.input.size().cols == nodes[0] && data.input.size().rows == 1) // Data is a vector
				input = data.input.transposed();
			else
				RapidError("Network Error", "Invalid input data for neural network. Size was invalid.").display();

			// Correct output
			auto end = nodes.size() - 1;
			if (data.output.size().rows == nodes[end] && data.output.size().cols == 1) // Data is a column vector
				output = data.output;
			else if (data.output.size().cols == nodes[end] && data.output.size().rows == 1) // Data is a vector
				output = data.output.transposed();
			else
				RapidError("Network Error", "Invalid output data for neural network. Size was invalid.").display();

			auto current = input;
			for (const auto &layer : layers)
			{
				current = layer->feedForward(current, true);
			}

			auto error = output - current;

			currentLoss = error.largest();

			for (long long i = layers.size() - 1; i >= 0; i--)
			{
				error = layers[i]->backpropagate(error);
			}
		}

		inline void fit(const std::vector<NetworkData<networkType>> &trainingData)
		{
			fit(trainingData, epochs, batchSize);
		}

		inline void fit(const std::vector<NetworkData<networkType>> &trainingData, size_t trainingEpochs)
		{

			fit(trainingData, trainingEpochs, batchSize);
		}
		
		inline void fit(const std::vector<NetworkData<networkType>> &newTrainingData, size_t trainingEpochs, size_t trainingBatchSize)
		{
			// Loop for the correct number of training epochs
			for (size_t epoch = 0; epoch < trainingEpochs + 1; epoch++)
			{
				size_t batch = 0;														// Current batch index
				size_t nextBatch = rapidMin(trainingBatchSize, newTrainingData.size());	// Next batch index

				while (batch < newTrainingData.size())
				{
					for (; batch < nextBatch; batch++)
					{
						backpropagate(newTrainingData[batch]);
					}

					nextBatch += trainingBatchSize;
					nextBatch = rapid::clamp(nextBatch, newTrainingData.size());
				}

				// Learning rate decay
				for (const auto &layer : layers)
					layer->setLearningRate((1 / (1 + (networkType) epoch * decayRate)) * layer->getLearningRate());

				if (verbose)
					std::cout << "Epoch: " << epoch << "/" << trainingEpochs << " | Loss: " << currentLoss << "\n";
			}
		}
	};
}
