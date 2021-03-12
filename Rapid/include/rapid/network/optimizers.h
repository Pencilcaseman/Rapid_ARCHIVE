#pragma once

#include "../internal.h"
#include "../math.h"
#include "../array.h"

namespace rapid
{
	namespace optim
	{
		template<typename t>
		struct LearningRate
		{
			int initialized = 0;
			t defaultValue;
			t value;

			inline void setValue(t val)
			{
				initialized = 1;
				value = val;
			}

			inline t getValue() const
			{
				return initialized ? value : defaultValue;
			}
		};

		template<typename t>
		struct Scalar
		{
			int initialized = 0;
			t defaultValue;
			t value;

			inline void setValue(t val)
			{
				initialized = 1;
				value = val;
			}

			inline t getValue() const
			{
				return initialized ? value : defaultValue;
			}
		};

		template<typename t>
		struct NDArray
		{
			int initialized = 0;
			Array<t> defaultValue;
			Array<t> value;

			inline void setValue(const Array<t> &val)
			{
				initialized = 1;
				value = val.copy();
			}

			inline Array<t> &getValue() const
			{
				return initialized ? value : defaultValue;
			}
		};

		template<typename t>
		struct Config
		{
			int initialized = 0;             // Is the context initialized

			LearningRate<t> learningRate;    // Scalar learning rate
			Scalar<t> momentum;              // Scalar between 0 and 1 representing momentum value
			NDArray<t> velocity;             // Array of same shape as w storing a moving average of gradients
			Scalar<t> decayRate;             // Scalar between 0 and 1 storing the decay rate for the squared gradient cache
			Scalar<t> epsilon;               // Small scalar used for smoothing to avoid division by zero
			NDArray<t> cache;                // Moving average of second moments of gradients
			Scalar<t> beta1;                 // Decay rate for moving average of first moment gradient
			Scalar<t> beta2;                 // Decay rate for moving average of second moment gradient
			NDArray<t> m;                    // Moving average of gradient
			NDArray<t> v;                    // Moving average of squared gradient
			uint64_t t;                      // Iteration number
		};

		template<typename t>
		struct OptimOutput
		{
			Array<t> weight;
			Config<t> config;
		};

		/// <summary>
		/// Compute vanilla stochastic gradient descent.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="arr"></param>
		/// <param name="config"></param>
		/// <returns></returns>
		template<typename t>
		inline OptimOutput<t> sdg(Array<t> &w, const Array<t> &dw, Config<t> &config)
		{
			if (config.initialized == 0)
			{
				config.initialized = 1;
				config.learningRate.initialized = 0;
				config.learningRate.defaultValue = 1e-2;
			}

			w -= config.learningRate.getValue() * dw;
			return {w, config};
		}

		/// <summary>
		/// Stochastic gradient descent with momentum
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="w"></param>
		/// <param name="dw"></param>
		/// <param name="config"></param>
		/// <returns></returns>
		template<typename t>
		inline OptimOutput<t> sdgMomentum(Array<t> &w, const Array<t> &dw, Config<t> &config)
		{
			if (config.initialized == 0)
			{
				config.initialized = 1;
				config.learningRate.defaultValue = 1e-2;
				config.momentum.defaultValue = 0.9;
				config.velocity.defaultValue = zerosLike(w);
			}

			// Momentum update formula -- also update velocity
			auto v = config.velocity.getValue();
			v = config.momentum.getValue() * v - config.learningRate.getValue() * dw;
			auto nextW = w + v;
			config.velocity.setValue(v);

			return {nextW, config};
		}

		/// <summary>
		/// Use the RMSProp update rule -- A moving average of squared gradients
		/// to set adaptive per-parameter learning rates
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="x"></param>
		/// <param name="dx"></param>
		/// <param name="config"></param>
		/// <returns></returns>
		template<typename t>
		inline OptimOutput<t> rmsprop(Array<t> &x, const Array<t> &dx, Config<t> &config)
		{
			if (config.initialized == 0)
			{
				config.initialized = 1;
				config.learningRate.defaultValue = 1e-2;
				config.decayRate.defaultValue = 0.99;
				config.epsilon = 1e-8;
				config.cache = zerosLike(x);
			}

			auto cache = config.cache.getValue();
			cache = config.decayRate.getValue() * cache + (1 - config.decayRate.getValue()) * (dx * dx);
			auto nextX = x - config.learningRate.getValue() * dx / (sqrt(cache) + config.epsilon.getValue());
			config.cache.setValue(cache);

			return {nextX, config};
		}

		/// <summary>
		/// Uses the Adam update rule, which incorporates moving averages of
		/// both the gradient and its square, and a bias correction term
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="x"></param>
		/// <param name="dx"></param>
		/// <param name="config"></param>
		/// <returns></returns>
		template<typename ty>
		inline OptimOutput<ty> adam(Array<ty> &x, const Array<ty> &dx, Config<ty> &config)
		{
			if (config.initialized == 0)
			{
				config.initialized = 0;
				config.learningRate.defaultValue = 1e-3;
				config.beta1.defaultValue = 0.9;
				config.beta2.defaultValue = 0.999;
				config.epsilon.defaultValue = 1e-8;
				config.m.defaultValue = zerosLike(x);
				config.v.defaultValue = zerosLike(x);
				config.t = 0;
			}

			auto &m = config.m.getValue();
			auto &v = config.v.getValue();
			auto t = config.t;

			t++;
			m = config.beta1.getValue() * m + (1 - config.beta1.getValue()) * dx;
			auto mCorr = m / (1 - std::pow(config.beta1.getValue(), (ty) t));
			v = config.beta2.getValue() * v + (1 - config.beta2.getValue()) * (dx * dx);
			auto vCorr = v / (1 - std::pow(config.beta2.getValue(), (ty) t));
			auto nextX = x - config.learningRate.getValue() * mCorr / (sqrt(vCorr) + config.epsilon.getValue());
			config.m.setValue(m);
			config.v.setValue(v);
			config.t = t;

			return {nextX, config};
		}
	}
}