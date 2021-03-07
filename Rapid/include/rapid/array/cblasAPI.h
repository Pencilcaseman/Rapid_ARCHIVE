#pragma once

#include "../internal.h"

namespace rapid
{
	namespace imp
	{
		template<typename t>
		inline t rapid_dot(uint64_t len, const t *__restrict a, const t *__restrict b)
		{
			static_assert(false, "Invalid type for vectorDot");
		}

		template<>
		inline double rapid_dot(uint64_t len, const double *__restrict a, const double *__restrict b)
		{
			return cblas_ddot(len, a, 1, b, 1);
		}

		template<>
		inline float rapid_dot(uint64_t len, const float *__restrict a, const float *__restrict b)
		{
			return cblas_sdot(len, a, 1, b, 1);
		}

		template<typename t>
		inline void rapid_gemm(uint64_t M, uint64_t N, uint64_t K, const t *__restrict a, const t *__restrict b, t *__restrict c)
		{
			static_assert(false, "Invalid type for vectorDot");
		}

		template<>
		inline void rapid_gemm(uint64_t M, uint64_t N, uint64_t K, const double *__restrict a, const double *__restrict b, double *__restrict c)
		{
			cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1., a, N, b, K, 0., c, K);
		}

		template<>
		inline void rapid_gemm(uint64_t M, uint64_t N, uint64_t K, const float *__restrict a, const float *__restrict b, float *__restrict c)
		{
			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1., a, N, b, K, 0., c, K);
		}
	}
}
