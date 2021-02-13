#pragma once

#include "../../internal.h"
#include "../../matrix.h"

namespace rapid
{
	template<typename layerType>
	class Layer
	{
	public:
		inline virtual void compile(size_t nodesBefore, size_t nodesAfter) = 0;
		inline virtual Matrix<layerType> feedForward(const Matrix<layerType> &input, bool saveLayerData) = 0;
		inline virtual Matrix<layerType> backpropagate(const Matrix<layerType> &output) = 0;

		inline virtual size_t getSize() = 0;
		inline virtual void setLearningRate(layerType newLr) = 0;
		inline virtual layerType getLearningRate() = 0;
	};
}
