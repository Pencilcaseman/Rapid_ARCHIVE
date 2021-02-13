#pragma once

#error "Depreciated! Use 'networkCore.h'"

#include "../internal.h"
#include "../math.h"
#include "../matrix.h"
#include "../graphics/messageBoxCore.h"

namespace rapid
{
	void initializeActivations();

	template<typename networkType>
	struct Metric
	{
		std::string name;
		std::vector<networkType> values;
		bool log;
	};

	template<typename networkType>
	struct NetworkData
	{
		Matrix<networkType> input;
		Matrix<networkType> output;
	};

	template<typename networkType>
	struct NetworkActivation
	{
		std::string name;
		std::function<networkType(networkType)> activation;
		std::function<networkType(networkType)> derivative;
	};

#ifndef RAPID_ACTIVATIONS
#define RAPID_ACTIVATIONS

	NetworkActivation<RAPID_ACTIVATION_DEFAULT> sigmoidActivation;
	NetworkActivation<RAPID_ACTIVATION_DEFAULT> tanhActivation;
	NetworkActivation<RAPID_ACTIVATION_DEFAULT> reluActivation;
	NetworkActivation<RAPID_ACTIVATION_DEFAULT> leakyReluActivation;

#endif

	template<typename networkType>
	class DeepNetwork
	{
	public:
		std::vector<size_t> nodes;
		std::vector<Matrix<networkType>> layers;
		std::vector<Matrix<networkType>> biases;
		std::vector<NetworkActivation<networkType>> activations;
		std::vector<networkType> learningRates;

		std::vector<size_t> temporaryNodes;
		std::vector<NetworkActivation<networkType>> temporaryActivations;
		std::vector<networkType> temporaryLearningRates;
		float defaultLearningRate = 0.1;
		float randomMin = -1;
		float randomMax = 1;

		bool constructed = false;

	public:

		DeepNetwork() = default;

		void setRandomRange(float min, float max)
		{
			randomMin = min;
			randomMax = max;
		}

		void addLayers(const std::vector<size_t> &numNodes, const std::vector<NetworkActivation<networkType>> &layerActivations = {}, const std::vector<float> &layerLearningRates = {})
		{
			for (size_t i = 0; i < numNodes.size(); i++)
			{
				temporaryNodes.emplace_back(numNodes[i]);

				if (i < layerActivations.size())
					temporaryActivations.emplace_back(layerActivations[i]);
				else
					temporaryActivations.emplace_back(NetworkActivation<networkType>());

				if (i < layerLearningRates.size())
					temporaryLearningRates.emplace_back(layerLearningRates[i]);
				else
					temporaryLearningRates.emplace_back(defaultLearningRate);
			}
		}

		void addLayer(size_t numNodes, float layerLearningRate, const NetworkActivation<networkType> &activation)
		{
			temporaryNodes.emplace_back(numNodes);
			temporaryActivations.emplace_back(activation);
			temporaryLearningRates.emplace_back(layerLearningRate);
		}

		void construct()
		{
			// TODO: Allow networks that have already been constructed to be updated by calling this function

			// Erase the current network
			nodes = temporaryNodes;
			activations = temporaryActivations;
			learningRates = temporaryLearningRates;
			layers.clear();
			biases.clear();

			for (size_t i = 0; i < nodes.size() - 1; i++)
			{
				auto range = pow(nodes[i], -0.5);
				// layers.emplace_back(Matrix<networkType>::random(nodes[i + 1], nodes[i], randomMin, randomMax));
				// biases.emplace_back(Matrix<networkType>::random(nodes[i + 1], 1, randomMin, randomMax));
				layers.emplace_back(Matrix<networkType>::random(nodes[i + 1], nodes[i], -range, range));
				biases.emplace_back(Matrix<networkType>::random(nodes[i + 1], 1, -range, range));
			}

			initializeActivations();

			constructed = true;
		}

		Matrix<networkType> feedForward(const NetworkData<networkType> &input) const
		{
			rapidAssert(constructed, "Network has not been constructed yet");

			Matrix<networkType> current;

			if (input.input.size().rows == nodes[0] && input.input.size().cols == 1) // Input is column vector
				current = input.input;
			else if (input.input.size().cols == nodes[0] && input.input.size().rows == 1) // Input is a vector, so transpose it
				current = input.input.transposed();
			else
				RapidError("Network Error", "Invalid shape for input data").display();

			for (size_t i = 0; i < nodes.size() - 1; i++)
			{
				current = layers[i].dot(current);
				current += biases[i];
				current.map(activations[i].activation);
			}

			return current;
		}

		void backpropagate(const NetworkData<networkType> &data)
		{
			rapidAssert(constructed, "Network has not been constructed yet");

			Matrix<networkType> input, output; // Corrected data for the network

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

			std::vector<Matrix<networkType>> layerData;						// Layer data
			std::vector<Matrix<networkType>> errorData(nodes.size() - 1);	// Error data

			Matrix<networkType> current(input);								// The current state of the network data
			for (size_t i = 0; i < nodes.size() - 1; i++)
			{
				current = layers[i].dot(current);
				current += biases[i];
				current.map(activations[i].activation);

				layerData.emplace_back(current);
			}

			errorData[errorData.size() - 1] = output - layerData[layerData.size() - 1];

			// std::cout << "Mean squared error: " << pow(errorData[errorData.size() - 1].mean(), 2) << "\n";

			for (long long i = nodes.size() - 2; i >= 0; i--)
			{
				auto gradient = layerData[i].mapped(activations[i].derivative);
				gradient *= errorData[i];
				gradient *= learningRates[i];

				Matrix<networkType> transposed;
				if (i > 0)
					transposed = layerData[i - 1].transposed();
				else
					transposed = input.transposed();

				auto weightDelta = gradient.dot(transposed);

				layers[i] += weightDelta;
				biases[i] += gradient;

				if (i > 0)
					errorData[i - 1] = layers[i].transposed().dot(errorData[i]);
			}
		}

		void saveToFile(const std::string &dir)
		{
			createDirectory(dir.substr(0, dir.find_last_of("/")));

			// Extract the filename
			auto fileTemp = dir.substr(dir.find_last_of("/") + 1, dir.length());													// Filename including extension
			auto filenameEnd = fileTemp.find_last_of(".") == std::string::npos ? fileTemp.length() : fileTemp.find_last_of(".");	// End of the filename excluding extension
			auto filename = fileTemp.substr(0, filenameEnd);																		// Extracted filename

			std::fstream file;
			file.open(dir, std::ios::out);

			// Dump number of nodes
			file << nodes.size() << "\n";

			// Dump nodes of each layer
			for (const auto &node : nodes)
				file << node << "\n";

			// Dump learning rates
			for (const auto &lr : learningRates)
				file << lr << "\n";

			// Dump activation names
			for (const auto &activation : activations)
				file << activation.name << "\n";

			// Dump layers externally and dump filename
			for (size_t i = 0; i < layers.size(); i++)
			{
				file << filename << "_layer_" << i << ".mat" << "\n";
				layers[i].saveToFile(dir.substr(0, dir.find_last_of("/")) + "/" + filename + "_layer_" + rapidCast<std::string>(i) + ".mat");
			}

			// Dump biases externally and dump filename
			for (size_t i = 0; i < biases.size(); i++)
			{
				file << filename << "_bias_" << i << ".mat" << "\n";
				biases[i].saveToFile(dir.substr(0, dir.find_last_of("/")) + "/" + filename + "_bias_" + rapidCast<std::string>(i) + ".mat");
			}
		}

		static DeepNetwork loadFromFile(const std::string &dir)
		{
			DeepNetwork result;

			// Extract the filename
			auto filePath = dir.substr(0, dir.find_last_of("/"));

			std::fstream file;
			file.open(dir, std::ios::in);

			// Load number of nodes
			size_t nodes;
			file >> nodes;
			result.temporaryNodes = std::vector<size_t>(nodes);

			// Load nodes of each layer
			for (size_t i = 0; i < nodes; i++)
				file >> result.temporaryNodes[i];

			// Load learning rates
			result.temporaryLearningRates = std::vector<networkType>(nodes);
			for (size_t i = 0; i < nodes; i++)
				file >> result.temporaryLearningRates[i];

			// Load activation names
			result.temporaryActivations = std::vector<NetworkActivation<networkType>>(nodes);

			std::getline(file, result.temporaryActivations[0].name);
			for (size_t i = 0; i < nodes; i++)
			{
				std::getline(file, result.temporaryActivations[i].name);

				if (result.temporaryActivations[i].name == "sigmoid")
					result.temporaryActivations[i] = sigmoidActivation;
				else if (result.temporaryActivations[i].name == "tanh")
					result.temporaryActivations[i] = tanhActivation;
				else if (result.temporaryActivations[i].name == "relu")
					result.temporaryActivations[i] = reluActivation;
				else if (result.temporaryActivations[i].name == "leaky relu")
					result.temporaryActivations[i] = leakyReluActivation;
				else
					result.temporaryActivations[i] = {"NONE",
					std::function<networkType(networkType)>([](networkType x)
					{
						return 0;
					}),
					std::function<networkType(networkType)>([](networkType x)
					{
						return 0;
					})
				};
			}

			// Construct the network (for safety purposes)
			result.construct();

			// Load layers externally and Load filename
			for (size_t i = 0; i < nodes - 1; i++)
			{
				std::string layerPath;
				std::getline(file, layerPath);
				result.layers[i] = Matrix<networkType>::loadFromFile(filePath + "/" + layerPath);
			}

			// Load biases externally and Load filename
			for (size_t i = 0; i < nodes - 1; i++)
			{
				std::string biasPath;
				std::getline(file, biasPath);
				result.biases[i] = Matrix<networkType>::loadFromFile(filePath + "/" + biasPath);
			}

			return result;
		}
	};
}

#include "activations.h"
