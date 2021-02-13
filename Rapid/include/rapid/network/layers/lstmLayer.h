#pragma once

#include "../../internal.h"

namespace rapid
{
	namespace imp
	{
		template<typename cellType>
		class LstmParam
		{
		public:
			// Size variables
			size_t memoryCellSize;
			size_t inputSize;
			size_t concatenatedSize;

			// Weight matrices
			Matrix<cellType> inputWeight;
			Matrix<cellType> forgetWeight;
			Matrix<cellType> outputWeight;
			Matrix<cellType> gateWeight;

			// Bias terms
			Matrix<cellType> inputBias;
			Matrix<cellType> forgetBias;
			Matrix<cellType> outputBias;
			Matrix<cellType> gateBias;

			// Derivatives of the loss function w.r.t. all parameters
			Matrix<cellType> inputWeightDiff;
			Matrix<cellType> forgetWeightDiff;
			Matrix<cellType> outputWeightDiff;
			Matrix<cellType> gateWeightDiff;

			Matrix<cellType> inputBiasDiff;
			Matrix<cellType> forgetBiasDiff;
			Matrix<cellType> outputBiasDiff;
			Matrix<cellType> gateBiasDiff;

			LstmParam()
			{}

			LstmParam(size_t memCellSize, size_t inputDims)
			{
				inputSize = inputDims;
				memoryCellSize = memCellSize;
				concatenatedSize = inputDims + memCellSize;

				// Initialize the weight matrices
				inputWeight = Matrix<cellType>::random(memoryCellSize, concatenatedSize, -0.1, 0.1);
				forgetWeight = Matrix<cellType>::random(memoryCellSize, concatenatedSize, -0.1, 0.1);
				outputWeight = Matrix<cellType>::random(memoryCellSize, concatenatedSize, -0.1, 0.1);
				gateWeight = Matrix<cellType>::random(memoryCellSize, concatenatedSize, -0.1, 0.1);

				// Initialize the bias terms
				inputBias = Matrix<cellType>::random(memoryCellSize, 1, -0.1, 0.1);
				forgetBias = Matrix<cellType>::random(memoryCellSize, 1, -0.1, 0.1);
				outputBias = Matrix<cellType>::random(memoryCellSize, 1, -0.1, 0.1);
				gateBias = Matrix<cellType>::random(memoryCellSize, 1, -0.1, 0.1);

				// Initialize the derivative matrices
				inputWeightDiff = Matrix<cellType>::zerosLike(inputWeight);
				forgetWeightDiff = Matrix<cellType>::zerosLike(forgetWeight);
				outputWeightDiff = Matrix<cellType>::zerosLike(outputWeight);
				gateWeightDiff = Matrix<cellType>::zerosLike(gateWeight);

				inputBiasDiff = Matrix<cellType>::zerosLike(inputBias);
				forgetBiasDiff = Matrix<cellType>::zerosLike(forgetBias);
				outputBiasDiff = Matrix<cellType>::zerosLike(outputBias);
				gateBiasDiff = Matrix<cellType>::zerosLike(gateBias);
			}

			void applyDiff(cellType learningRate)
			{
				// Update weights and bias terms
				inputWeight -= inputWeightDiff * learningRate;
				forgetWeight -= forgetWeightDiff * learningRate;
				outputWeight -= outputWeightDiff * learningRate;
				gateWeight -= gateWeightDiff * learningRate;

				inputBias -= inputBiasDiff * learningRate;
				forgetBias -= forgetBiasDiff * learningRate;
				outputBias -= outputBiasDiff * learningRate;
				gateBias -= gateBiasDiff * learningRate;

				// Reset derivative terms to zero
				inputWeightDiff = Matrix<cellType>::zerosLike(inputWeightDiff);
				forgetWeightDiff = Matrix<cellType>::zerosLike(forgetWeightDiff);
				outputWeightDiff = Matrix<cellType>::zerosLike(outputWeightDiff);
				gateWeightDiff = Matrix<cellType>::zerosLike(gateWeightDiff);

				inputBiasDiff = Matrix<cellType>::zerosLike(inputBiasDiff);
				forgetBiasDiff = Matrix<cellType>::zerosLike(forgetBiasDiff);
				outputBiasDiff = Matrix<cellType>::zerosLike(outputBiasDiff);
				gateBiasDiff = Matrix<cellType>::zerosLike(gateBiasDiff);
			}
		};

		template<typename stateType>
		class LstmState
		{
		public:
			Matrix<stateType> inputGate;
			Matrix<stateType> forgetGate;
			Matrix<stateType> outputGate;
			Matrix<stateType> gateGate;

			Matrix<stateType> state;
			Matrix<stateType> output;
			Matrix<stateType> bottomDiffOutput;
			Matrix<stateType> bottomDiffState;

			LstmState() = default;

			LstmState(size_t memoryCellSize, size_t inputSize)
			{
				inputGate = Matrix<stateType>::zeros(1, memoryCellSize);
				forgetGate = Matrix<stateType>::zeros(1, memoryCellSize);
				outputGate = Matrix<stateType>::zeros(1, memoryCellSize);
				gateGate = Matrix<stateType>::zeros(1, memoryCellSize);
				state = Matrix<stateType>::zeros(1, memoryCellSize);
				output = Matrix<stateType>::zeros(1, memoryCellSize);

				bottomDiffState = Matrix<stateType>::zerosLike(state);
				bottomDiffOutput = Matrix<stateType>::zerosLike(output);
			}
		};

		template<typename nodeType>
		class LstmNode
		{
		public:
			// Store references to the parameters
			LstmState<nodeType> state;
			// LstmParam<nodeType> param;
			std::shared_ptr<imp::LstmParam<nodeType>> param;

			// Non-recurrent input concatenated with the recurrent input (output of the previous block)
			Matrix<nodeType> xc;

			// Previous data
			Matrix<nodeType> sPrev, hPrev;

			// Activations
			NetworkActivation<nodeType> sigmoid;
			NetworkActivation<nodeType> tanh;

			// LstmNode(const LstmParam<nodeType> &nodeParam, const LstmState<nodeType> &nodeState)
			LstmNode(const std::shared_ptr<imp::LstmParam<nodeType>> &nodeParam, const LstmState<nodeType> &nodeState)
			{
				state = nodeState;
				param = nodeParam;

				sigmoid = Activations<nodeType>::sigmoid();
				tanh = Activations<nodeType>::tanh();
			}

			// Calculate output of the relevant gates based on input data
			inline void bottomDataIs(const Matrix<nodeType> &x, const Matrix<nodeType> &sPrevInput = Matrix<nodeType>(), const Matrix<nodeType> &hPrevInput = Matrix<nodeType>())
			{
				// If this is the first block in the chain, these will be uninitialized
				if (sPrevInput.size().rows == 0 && sPrevInput.size().cols == 0)
					sPrev = Matrix<nodeType>::zerosLike(state.state);
				else
					sPrev = sPrevInput;

				if (hPrevInput.size().rows == 0 && hPrevInput.size().cols == 0)
					hPrev = Matrix<nodeType>::zerosLike(state.output);
				else
					hPrev = hPrevInput;

				// Concatenate the input and recurrent input
				xc = Matrix<nodeType>::hstack(x.transposed(), hPrev);

				// Calculate the output of the gates
				state.forgetGate = (param->forgetWeight.dot(xc.transposed()) + param->forgetBias).mapped(sigmoid.activation);
				state.inputGate = (param->inputWeight.dot(xc.transposed()) + param->inputBias).mapped(sigmoid.activation);
				state.outputGate = (param->outputWeight.dot(xc.transposed()) + param->outputBias).mapped(sigmoid.activation);
				state.gateGate = (param->gateWeight.dot(xc.transposed()) + param->gateBias).mapped(tanh.activation);

				state.state = ((sPrev * state.forgetGate.transposed()).transposed() + state.gateGate * state.inputGate).transposed();
				state.output = state.state * state.outputGate.transposed();
			}

			// Calculate derivatives based on error
			inline void topDiffIs(const Matrix<nodeType> &topDiffH, const Matrix<nodeType> &topDiffS)
			{
				// Calculate the errors

				auto stateDiff = state.outputGate.transposed() * topDiffH + topDiffS;
				auto outputDiff = state.state * topDiffH;
				auto inputDiff = state.gateGate.transposed() * stateDiff;
				auto gateDiff = state.inputGate.transposed() * stateDiff;
				auto forgetDiff = sPrev * stateDiff;

				// Calculate the derivatives w.r.t. the vectors inside sigmoid and tanh functions

				auto inputDiff_input = state.inputGate.mapped(sigmoid.derivative) * inputDiff.transposed();
				auto forgetDiff_input = state.forgetGate.mapped(sigmoid.derivative) * forgetDiff.transposed();
				auto outputDiff_input = state.outputGate.mapped(sigmoid.derivative) * outputDiff.transposed();
				auto gateDiff_input = state.gateGate.mapped(tanh.derivative) * gateDiff.transposed();

				// Calculate derivatives w.r.t. input values
				param->inputWeightDiff += Matrix<nodeType>::outer(inputDiff_input, xc);
				param->forgetWeightDiff += Matrix<nodeType>::outer(forgetDiff_input, xc);
				param->outputWeightDiff += Matrix<nodeType>::outer(outputDiff_input, xc);
				param->gateWeightDiff += Matrix<nodeType>::outer(gateDiff_input, xc);

				param->inputBiasDiff += inputDiff_input;
				param->forgetBiasDiff += forgetDiff_input;
				param->outputBiasDiff += outputDiff_input;
				param->gateBiasDiff += gateDiff_input;

				// Calculate the bottom derivatives

				auto dxc = Matrix<nodeType>::zerosLike(xc).transposed();
				dxc += param->inputWeight.transposed().dot(inputDiff_input);
				dxc += param->forgetWeight.transposed().dot(forgetDiff_input);
				dxc += param->outputWeight.transposed().dot(outputDiff_input);
				dxc += param->gateWeight.transposed().dot(gateDiff_input);

				// Save the derivatives
				state.bottomDiffState = stateDiff.transposed() * state.forgetGate;
				state.bottomDiffOutput = dxc.subMatrix(param->inputSize).transposed();
			}
		};

		template<typename networkType>
		class LstmNetwork
		{
		public:
			// imp::LstmParam<networkType> lstmParam; // Current LSTM cell
			std::shared_ptr<imp::LstmParam<networkType>> lstmParam; // Current LSTM cell
			std::vector<imp::LstmNode<networkType>> lstmNodeList; // List of previous histories
			std::vector<Matrix<networkType>> xList; // List of input values

			LstmNetwork()
			{};

			// LstmNetwork(imp::LstmParam<networkType> newLstmParam)
			LstmNetwork(const std::shared_ptr<imp::LstmParam<networkType>> &newLstmParam)
			{
				lstmParam = newLstmParam;
			}

			template<typename networkLossLayer>
			inline Matrix<networkType> yListIs(const std::vector<Matrix<networkType>> &yList, const networkLossLayer &lossLayer)
			{
				// Updates derivatives by setting the target sequence with the corresponding loss
				// This does not update any parameters

				rapidAssert(yList.size() == xList.size(), "Size of input does not equal size of output");
				auto idx = (long long) xList.size() - 1;

				// First node gets the loss from the labeled data
				auto loss = lossLayer.loss(lstmNodeList[idx].state.output, yList[idx]);
				auto diffH = lossLayer.bottomDiff(lstmNodeList[idx].state.output, yList[idx]);

				// S is not affecting the loss due to h(t+1), so set it to zero
				auto diffS = Matrix<networkType>::zeros(1, lstmParam->memoryCellSize);
				lstmNodeList[idx].topDiffIs(diffH, diffS);
				idx--;

				// The following nodes also get gradients from next nodes, so add gradients to diffH
				// The error is also propagated along
				while (idx >= 0)
				{
					loss += lossLayer.loss(lstmNodeList[idx].state.output, yList[idx]);
					diffH = lossLayer.bottomDiff(lstmNodeList[idx].state.output, yList[idx]);
					diffH += lstmNodeList[idx + 1].state.bottomDiffOutput;
					diffS = lstmNodeList[idx + 1].state.bottomDiffState.transposed();
					lstmNodeList[idx].topDiffIs(diffH, diffS);
					idx--;
				}

				return loss;
			}

			inline Matrix<networkType> yErrorIs(const std::vector<Matrix<networkType>> &lossLayerLoss, const std::vector<Matrix<networkType>> &lossLayerBottomDiff)
			{
				// Updates derivatives by setting the target sequence with the corresponding loss
				// This does not update any parameters

				auto idx = (long long) xList.size() - 1;

				// First node gets the loss from the labeled data
				auto loss = lossLayerLoss[idx];
				auto diffH = lossLayerBottomDiff[idx];

				// S is not affecting the loss due to h(t+1), so set it to zero
				auto diffS = Matrix<networkType>::zeros(1, lstmParam->memoryCellSize);
				lstmNodeList[idx].topDiffIs(diffH, diffS);
				idx--;

				// The following nodes also get gradients from next nodes, so add gradients to diffH
				// The error is also propagated along
				while (idx >= 0)
				{
					loss += lossLayerLoss[idx];
					diffH = lossLayerBottomDiff[idx];
					diffH += lstmNodeList[idx + 1].state.bottomDiffOutput;
					diffS = lstmNodeList[idx + 1].state.bottomDiffState.transposed();
					lstmNodeList[idx].topDiffIs(diffH, diffS);
					idx--;
				}

				return loss;
			}

			// Clear the input array
			inline void clearX()
			{
				xList.clear();
			}

			// Add a value to the input array
			inline void addInput(const Matrix<networkType> &x)
			{
				xList.emplace_back(x);
				if (xList.size() > lstmNodeList.size())
				{
					// Need to add a new LSTM node and create new memory state
					auto lstmState = LstmState<networkType>(lstmParam->memoryCellSize, lstmParam->inputSize);
					lstmNodeList.emplace_back(LstmNode<networkType>(lstmParam, lstmState));
				}

				// Get the most recent index of x
				auto idx = xList.size() - 1;
				if (idx == 0)
					lstmNodeList[idx].bottomDataIs(x);
				else
				{
					auto sPrev = lstmNodeList[idx - 1].state.state;
					auto hPrev = lstmNodeList[idx - 1].state.output;
					lstmNodeList[idx].bottomDataIs(x, sPrev, hPrev);
				}
			}
		};
	}

	template<typename layerType>
	class lstm : public Layer<layerType>
	{
	public:
		// Some useful data
		size_t inputSize = 0, outputSize = 0;
		layerType learningRate = 0.05;

		// The network data itself
		imp::LstmNetwork<layerType> network;
		std::shared_ptr<imp::LstmParam<layerType>> param;

		// Loss and bottom arrays
		std::vector<Matrix<layerType>> lossLayerLoss;
		std::vector<Matrix<layerType>> lossLayerBottomDiff;

		// Copies of the network data for normal feed forwards. These are reset every $recurrenceLevel$ backpropagation iterations
		imp::LstmNetwork<layerType> networkCopy;
		std::shared_ptr<imp::LstmParam<layerType>> paramCopy;

		// Every $n$ backpropagation calls, update the stored networks and erase the stored input data
		size_t recurrenceLevel = 1;

		// Number of backpropagation passes completed
		size_t backpropagationCount = 0;

		lstm(size_t networkOutputSize, size_t networkRecurrenceLevel, layerType networkLearningRate)
		{
			recurrenceLevel = networkRecurrenceLevel;
			outputSize = networkOutputSize;
			learningRate = networkLearningRate;
		}

		void compile(size_t nodesBefore, size_t nodesAfter) override
		{
			inputSize = nodesBefore;

			param = std::make_shared<rapid::imp::LstmParam<layerType>>(outputSize, inputSize);
			network = rapid::imp::LstmNetwork<layerType>(param);

			paramCopy = std::make_shared<rapid::imp::LstmParam<layerType>>(*param);
			networkCopy = rapid::imp::LstmNetwork<layerType>(paramCopy);
		}

		Matrix<layerType> feedForward(const Matrix<layerType> &input, bool saveLayerdata) override
		{
			// if (!saveLayerdata) // If not saving the data (i.e. called from parent feedForward call) use the copies of the network
			// {
			// 	if (input.size().rows != 1)
			// 		networkCopy.addInput(input);
			// 	else
			// 		networkCopy.addInput(input.transposed());
			// 
			// 	return networkCopy.lstmNodeList[networkCopy.lstmNodeList.size() - 1].state.output.transposed();
			// }

			// Function call is being used in a backpropagation call, so use the real network and parameters
			if (input.size().rows != 1)
				network.addInput(input);
			else
				network.addInput(input.transposed());

			return network.lstmNodeList[network.xList.size() - 1].state.output.transposed();
			// return network.lstmNodeList[network.lstmNodeList.size() - 1].state.output.transposed();
		}

		Matrix<layerType> backpropagate(const Matrix<layerType> &error) override
		{
			// Add input
			// Store resulting loss and diffs
			// If required, update the network
			//   - Update gates and state
			//   - Clear inputs

			// Input added during feed forward stage
			// Store losses

			// lossLayerLoss.emplace_back(error.transposed());
			// lossLayerBottomDiff.emplace_back(error.transposed());
			lossLayerLoss.emplace_back(error.pow(2).transposed());
			lossLayerBottomDiff.emplace_back((error * 2).transposed());

			// Do stuff
			auto loss = network.yErrorIs(lossLayerLoss, lossLayerBottomDiff);

			if (backpropagationCount % recurrenceLevel == 0)
			{
				param->applyDiff(-learningRate); // Update network
				network.clearX();				 // Clear stored inputs
				lossLayerLoss.clear();			 // Clear losses
				lossLayerBottomDiff.clear();	 // Clear bottom differences

				// Update the copies
				// paramCopy = std::make_shared<rapid::imp::LstmParam<layerType>>(*param);
				// networkCopy = rapid::imp::LstmNetwork<layerType>(paramCopy);
			}

			backpropagationCount++;

			// TODO: Fix this
			// Return something. This is not the correct error, and
			// means LSTMs only work as the very first layer
			return error;
		}

		size_t getSize() override
		{
			return outputSize;
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
