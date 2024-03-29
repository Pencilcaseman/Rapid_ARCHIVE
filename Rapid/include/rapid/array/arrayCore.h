#pragma once

#include "../internal.h"
#include "../math.h"

#include "fromData.h"

#ifndef RAPID_NO_BLAS
#include "cblasAPI.h"
#endif

namespace rapid
{
	namespace utils
	{
		struct strContainer
		{
			std::string str;
			size_t decimalPoint;
		};

		/// <summary>
		/// Format a numerical value and return it as a string
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="val"></param>
		/// <returns></returns>
		template<typename t>
		strContainer formatNumerical(const t &val)
		{
			std::stringstream stream;
			stream << val;

			auto lastDecimal = stream.str().find_last_of('.');

			if (std::is_floating_point<t>::value && lastDecimal == std::string::npos)
			{
				stream << ".";
				lastDecimal = stream.str().length() - 1;
			}

			auto lastZero = stream.str().find_last_of('0');

			// Value is integral
			if (lastDecimal == std::string::npos)
				return {stream.str(), stream.str().length() - 1};

			return {stream.str(), lastDecimal};
		}

		/// <summary>
		/// Convert an index and a given array shape and return the memory
		/// offset for the given position in contiguous memory
		/// </summary>
		/// <typeparam name="indexT"></typeparam>
		/// <typeparam name="shapeT"></typeparam>
		/// <param name="index"></param>
		/// <param name="shape"></param>
		/// <returns></returns>
		template<typename indexT, typename shapeT>
		inline indexT ndToScalar(const std::vector<indexT> &index, const std::vector<shapeT> &shape)
		{
			indexT sig = 1;
			indexT pos = 0;

			for (indexT i = shape.size(); i >= 1; i--)
			{
				pos += (i - 1 < index.size() ? index[i - 1] : 0) * sig;
				sig *= shape[i - 1];
			}

			return pos;
		}

		/// <summary>
		/// Convert an index and a given array shape and return the memory
		/// offset for the given position in contiguous memory
		/// </summary>
		/// <typeparam name="indexT"></typeparam>
		/// <typeparam name="shapeT"></typeparam>
		/// <param name="index"></param>
		/// <param name="shape"></param>
		/// <returns></returns>
		template<typename indexT, typename shapeT>
		inline indexT ndToScalar(const std::initializer_list<indexT> &index, const std::vector<shapeT> &shape)
		{
			indexT sig = 1;
			indexT pos = 0;

			for (indexT i = shape.size(); i > 0; i--)
			{
				pos += (i - 1 < index.size() ? (*(index.begin() + i - 1)) : 0) * sig;
				sig *= shape[i - 1];
			}

			return pos;
		}
	}

	namespace imp
	{
		/// <summary>
		/// Convert a set of dimensions into a memory location. Intended for internal use only
		/// </summary>
		/// <param name="dims"></param>
		/// <param name="pos"></param>
		/// <returns></returns>
		long int dimsToIndex(const std::vector<uint64_t> &dims, const std::vector<uint64_t> &pos)
		{
			uint64_t index = 0;
			for (long int i = 0; i < dims.size(); i++)
			{
				uint64_t sub = pos[i];
				for (uint64_t j = i; j < dims.size() - 1; j++)
					sub *= dims[j + 1];
				index += sub;
			}
			return index;
		}
	}

	enum class ExecutionType
	{
		SERIAL = 0b0001,
		PARALLEL = 0b0010,
		MASSIVE = 0b0100
	};

	/// <summary>
	/// A powerful and fast ndarray type, supporting a wide variety
	/// of optimized functions and routines. It also supports different
	/// arrayTypes, allowing for greater flexibility.
	/// </summary>
	/// <typeparam name="arrayType"></typeparam>
	template<typename arrayType>
	class Array
	{
	public:
		std::vector<size_t> shape;
		arrayType *dataOrigin;
		arrayType *dataStart;
		size_t *originCount;
		bool isZeroDim;

		/// <summary>
		/// Apply a lambda function to two arrays, storing the result in a third.
		/// Both arrays must be the same size, but this in not checked when running,
		/// so it is therefore the responsibility of the user to ensure this function
		/// is called safely
		/// </summary>
		/// <typeparam name="Lambda"></typeparam>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="mode"></param>
		/// <param name="func"></param>
		template<typename Lambda>
		inline static void binaryOpArrayArray(const Array<arrayType> &a, const Array<arrayType> &b, Array<arrayType> &c, ExecutionType mode, Lambda func)
		{
			size_t size = prod(a.shape);

			if (mode == ExecutionType::SERIAL)
			{
				// Serial execution on CPU
				size_t index = 0;

				if (size > 3)
				{
					for (index = 0; index < size - 3; index += 4)
					{
						c.dataStart[index + 0] = func(a.dataStart[index + 0], b.dataStart[index + 0]);
						c.dataStart[index + 1] = func(a.dataStart[index + 1], b.dataStart[index + 1]);
						c.dataStart[index + 2] = func(a.dataStart[index + 2], b.dataStart[index + 2]);
						c.dataStart[index + 3] = func(a.dataStart[index + 3], b.dataStart[index + 3]);
					}
				}

				for (; index < size; index++)
					c.dataStart[index] = func(a.dataStart[index], b.dataStart[index]);
			}
			else if (mode == ExecutionType::PARALLEL)
			{
				// Parallel execution on CPU
				long index = 0;

			#pragma omp parallel for shared(size, a, b, c) private(index) default(none)
				for (index = 0; index < size; ++index)
					c.dataStart[index] = func(a.dataStart[index], b.dataStart[index]);
			}
			else
			{
				RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
			}
		}

		/// <summary>
		/// Apply a lambda function to an array in the format
		/// func(array, scalar) and store the result. Both arrays
		/// must be the same size, but this in not checked when running,
		/// so it is therefore the responsibility of the user to ensure
		/// this function is called safely
		/// </summary>
		/// <typeparam name="Lambda"></typeparam>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="mode"></param>
		/// <param name="func"></param>
		template<typename Lambda>
		inline static void binaryOpArrayScalar(const Array<arrayType> &a, const arrayType &b, Array<arrayType> &c, ExecutionType mode, Lambda func)
		{
			size_t size = prod(a.shape);

			if (mode == ExecutionType::SERIAL)
			{
				// Serial execution on CPU
				size_t index = 0;

				if (size > 3)
				{
					for (index = 0; index < size - 3; index += 4)
					{
						c.dataStart[index + 0] = func(a.dataStart[index + 0], b);
						c.dataStart[index + 1] = func(a.dataStart[index + 1], b);
						c.dataStart[index + 2] = func(a.dataStart[index + 2], b);
						c.dataStart[index + 3] = func(a.dataStart[index + 3], b);
					}
				}

				for (; index < size; index++)
					c.dataStart[index] = func(a.dataStart[index], b);
			}
			else if (mode == ExecutionType::PARALLEL)
			{
				// Parallel execution on CPU
				long index = 0;

			#pragma omp parallel for shared(size, a, b, c) private(index) default(none)
				for (index = 0; index < size; ++index)
					c.dataStart[index] = func(a.dataStart[index], b);
			}
			else
			{
				RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
			}
		}

		/// <summary>
		/// Apply a lambda function to a scalar and an array in the format
		/// func(scalar, array) and store the result. Both arrays
		/// must be the same size, but this in not checked when running,
		/// so it is therefore the responsibility of the user to ensure
		/// this function is called safely
		/// </summary>
		/// <typeparam name="Lambda"></typeparam>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="c"></param>
		/// <param name="mode"></param>
		/// <param name="func"></param>
		template<typename Lambda>
		inline static void binaryOpScalarArray(const arrayType &a, const Array<arrayType> &b, Array<arrayType> &c, ExecutionType mode, Lambda func)
		{
			size_t size = prod(b.shape);

			if (mode == ExecutionType::SERIAL)
			{
				// Serial execution on CPU
				size_t index = 0;

				if (size > 3)
				{
					for (index = 0; index < size - 3; index += 4)
					{
						c.dataStart[index + 0] = func(a, b.dataStart[index + 0]);
						c.dataStart[index + 1] = func(a, b.dataStart[index + 1]);
						c.dataStart[index + 2] = func(a, b.dataStart[index + 2]);
						c.dataStart[index + 3] = func(a, b.dataStart[index + 3]);
					}
				}

				for (; index < size; index++)
					c.dataStart[index] = func(a, b.dataStart[index]);
			}
			else if (mode == ExecutionType::PARALLEL)
			{
				// Parallel execution on CPU
				long index = 0;

			#pragma omp parallel for shared(size, a, b, c) private(index) default(none)
				for (index = 0; index < size; ++index)
					c.dataStart[index] = func(a, b.dataStart[index]);
			}
			else
			{
				RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
			}
		}

		/// <summary>
		/// Apply a lambda function to an array in the format
		/// func(array) and store the result
		/// </summary>
		/// <typeparam name="Lambda"></typeparam>
		/// <param name="a"></param>
		/// <param name="b"></param>
		/// <param name="mode"></param>
		/// <param name="func"></param>
		template<typename Lambda>
		inline static void unaryOpArray(const Array<arrayType> &a, Array<arrayType> &b, ExecutionType mode, Lambda func)
		{
			size_t size = prod(a.shape);

			if (mode == ExecutionType::SERIAL)
			{
				// Serial execution on CPU
				size_t index = 0;

				if (size > 3)
				{
					for (index = 0; index < size - 3; index += 4)
					{
						b.dataStart[index + 0] = func(a.dataStart[index + 0]);
						b.dataStart[index + 1] = func(a.dataStart[index + 1]);
						b.dataStart[index + 2] = func(a.dataStart[index + 2]);
						b.dataStart[index + 3] = func(a.dataStart[index + 3]);
					}
				}

				for (; index < size; index++)
					b.dataStart[index] = func(a.dataStart[index]);
			}
			else if (mode == ExecutionType::PARALLEL)
			{
				// Parallel execution on CPU
				long index = 0;

			#pragma omp parallel for shared(size, a, b) private(index) default(none)
				for (index = 0; index < size; ++index)
					b.dataStart[index] = func(a.dataStart[index]);
			}
			else
			{
				RapidError("Mode Error", "Invalid mode for binary mapping. Must be SERIAL or PARALLEL").display();
			}
		}

		/// <summary>
		/// Resize an array to different dimensions and return the result.
		/// The data stored in the array is copied, so an update in the
		/// result array will not trigger an update in the parent.
		/// </summary>
		/// <param name="newShape"></param>
		/// <returns></returns>
		inline Array<arrayType> internal_resized(const std::vector<uint64_t> &newShape) const
		{
			rapidAssert(newShape.size() == 2, "Resizing currently only supports 2D array");

			Array<arrayType> res(newShape);
			auto resData = res.dataStart;
			auto thisData = dataStart;

			for (size_t i = 0; i < rapid::min(shape[0], newShape[0]); i++)
				memcpy(resData + i * newShape[1], thisData + i * shape[1], sizeof(arrayType) * rapid::min(shape[1], newShape[1]));

			return res;
		}

		/// <summary>
		/// Resize an array to different dimensions and return the result.
		/// The data stored in the array is copied, so an update in the
		/// result array will not trigger an update in the parent.
		/// </summary>
		/// <param name="newShape"></param>
		/// <returns></returns>
		inline void internal_resize(const std::vector<uint64_t> &newShape)
		{
			auto newThis = internal_resized(newShape);

			// Only delete data if originCount becomes zero
			(*originCount)--;

			if ((*originCount) == 0)
			{
				delete[] dataOrigin;
				delete originCount;
			}

			originCount = newThis.originCount;
			(*originCount)++;

			dataOrigin = newThis.dataOrigin;
			dataStart = newThis.dataStart;

			shape = newShape;
		}

	public:

		/// <summary>
		/// Default constructor
		/// </summary>
		Array()
		{
			isZeroDim = true;
			shape = {0};
			dataStart = nullptr;
			dataOrigin = nullptr;
			originCount = nullptr;
		}

		/// <summary>
		/// Create a new array from a given shape. This allocates entirely
		/// new data, and no existing arrays are modified in any way.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <typeparam name="type"></typeparam>
		/// <param name="arrShape"></param>
		Array(const std::vector<size_t> &arrShape)
		{
		#ifdef RAPID_DEBUG
			for (const auto &val : arrShape)
				if (val <= 0)
					rapidAssert(false, "Dimensions must be positive");
		#endif

			if (arrShape.empty() || prod(arrShape) == 0)
			{
				isZeroDim = true;
				shape = {1};
				dataStart = new arrayType[1];
				dataOrigin = dataStart;
				originCount = new size_t;
				*originCount = 1;
			}
			else
			{
				isZeroDim = false;
				shape = arrShape;
				dataStart = new arrayType[prod(arrShape)];
				dataOrigin = dataStart;
				originCount = new size_t;
				*originCount = 1;
			}
		}

		/// <summary>
		/// Create an array from an existing array. The array that is created
		/// will inherit the same data as the array it is created from, so an
		/// update in one will cause an update in the other.
		/// </summary>
		/// <param name="other"></param>
		Array(const Array<arrayType> &other)
		{
			isZeroDim = other.isZeroDim;
			shape = other.shape;
			dataOrigin = other.dataOrigin;
			dataStart = other.dataStart;
			originCount = other.originCount;
			(*originCount)++;
		}

		/// <summary>
		/// Set one array equal to another and copy the memory.
		/// This means an update in one array will not trigger
		/// an update in the other
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		Array<arrayType> &operator=(const Array<arrayType> &other)
		{
			rapidAssert(shape == other.shape, "Invalid shape for array setting");
			memcpy(dataStart, other.dataStart, prod(shape) * sizeof(arrayType));
			return *this;
		}

		/// <summary>
		/// Set an array equal to a scalar value. This fills
		/// the array with the value.
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		Array<arrayType> &operator=(const arrayType &other)
		{
			fill(other);
			return *this;
		}

		/// <summary>
		/// Create an array from the provided data, without creating a
		/// temporary one first. This fixes memory leaks and is intended
		/// for internal use only.
		/// </summary>
		/// <param name="arrDims"></param>
		/// <param name="newDataOrigin"></param>
		/// <param name="dataStart"></param>
		/// <param name="originCount"></param>
		/// <param name="isZeroDim"></param>
		/// <returns></returns>
		static inline Array<arrayType> fromData(const std::vector<size_t> &arrDims, arrayType *newDataOrigin, arrayType *dataStart, size_t *originCount, bool isZeroDim)
		{
			Array<arrayType> res;
			res.isZeroDim = isZeroDim;
			res.shape = arrDims;
			res.dataOrigin = newDataOrigin;
			res.dataStart = dataStart;
			res.originCount = originCount;
			return res;
		}

		/// <summary>
		/// Create a new array from an initializer_list. This supports creating
		/// arrays of up to 20-dimensions via nested initializer lists
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="data"></param>
		/// <returns></returns>
		template<typename t>
		static inline Array<arrayType> fromData(const std::initializer_list<t> &data)
		{
			auto res = Array<arrayType>({data.size()});
			std::vector<arrayType> values;
			for (const auto &val : data)
				values.emplace_back(val);
			memcpy(res.dataStart, values.data(), sizeof(arrayType) * data.size());
			return res;
		}


	#define imp_temp template<typename t>
	#define imp_func_def(x) static inline Array<arrayType> fromData(const x &data)
	#define imp_func_body	auto res = Array<arrayType>(imp::extractShape(data)); \
							uint64_t index = 0; \
							for (const auto &val : data) res[index++] = Array<arrayType>::fromData(val); \
							return res;
	#define L std::initializer_list

		// Up to 20-dimensional array setting from data
		imp_temp imp_func_def(L<L<t>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<t>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<t>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<t>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<t>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<t>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<t>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<t>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>) { imp_func_body }
		imp_temp imp_func_def(L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<L<t>>>>>>>>>>>>>>>>>>>>) { imp_func_body }

	#undef imp_temp
	#undef imp_func_def
	#undef imp_func_body
	#undef L

		~Array()
		{
			// Only delete data if originCount becomes zero
			(*originCount)--;

			if ((*originCount) == 0)
			{
				delete[] dataOrigin;
				delete originCount;
			}
		}

		/// <summary>
		/// Cast a zero-dimensional array to a scalar value
		/// </summary>
		/// <typeparam name="t"></typeparam>
		template<typename t>
		inline operator t() const
		{
			// if (std::is_integral<t>::value || std::is_floating_point<t>::value)
			if (!isZeroDim)
				rapidAssert(isZeroDim, "Cannot cast multidimensional array to scalar value");
			return (t) (dataStart[0]);
		}

		/// <summary>
		/// Access a subarray or value of an array. The result is linked
		/// to the parent array, so an update in one will trigger an update
		/// in the other.
		/// </summary>
		/// <param name="index"></param>
		/// <returns></returns>
		Array<arrayType> operator[](const size_t &index) const
		{
			rapidAssert(index < shape[0], "Index out of range for array subscript");

			(*originCount)++;

			if (shape.size() == 1)
				return Array<arrayType>::fromData({1}, dataOrigin, dataStart + utils::ndToScalar({index}, shape), originCount, true);

			std::vector<size_t> resShape(shape.begin() + 1, shape.end());
			return Array<arrayType>::fromData(resShape, dataOrigin, dataStart + utils::ndToScalar({index}, shape), originCount, isZeroDim);
		}

		/// <summary>
		/// Directly access an individual value in an array. This does
		/// not allow for changing the value, but is much faster than
		/// accessing it via repeated subscript operations
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="index"></param>
		/// <returns></returns>
		template<typename t>
		inline arrayType accessVal(const std::initializer_list<t> &index) const
		{
			rapidAssert(index.size() == shape.size(), "Invalid number of dimensions to access");
		#ifdef RAPID_DEBUG
			for (size_t i = 0; i < index.size(); i++)
			{
				if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i])
					RapidError("Index Error", "Index out of range or negative");
			}
		#endif

			return dataStart[utils::ndToScalar(index, shape)];
		}

		/// <summary>
		/// Set a scalar value in an array from a given
		/// index location
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="index"></param>
		/// <param name="val"></param>
		template<typename t>
		inline void setVal(const std::initializer_list<t> &index, const arrayType &val) const
		{
			rapidAssert(index.size() == shape.size(), "Invalid number of dimensions to access");
		#ifdef RAPID_DEBUG
			for (size_t i = 0; i < index.size(); i++)
			{
				if (*(index.begin() + i) < 0 || *(index.begin() + i) >= shape[i])
					RapidError("Index Error", "Index out of range or negative");
			}
		#endif

			dataStart[utils::ndToScalar(index, shape)] = val;
		}

		/// <summary>
		/// Array add Array
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		inline Array<arrayType> operator+(const Array<arrayType> &other)const
		{
			rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayArray(*this, other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x + y;
			});
			return res;
		}

		/// <summary>
		/// Array sub Array
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		inline Array<arrayType> operator-(const Array<arrayType> &other)const
		{
			rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayArray(*this, other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x - y;
			});
			return res;
		}

		/// <summary>
		/// Array mul Array
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		inline Array<arrayType> operator*(const Array<arrayType> &other)const
		{
			rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayArray(*this, other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x * y;
			});
			return res;
		}

		/// <summary>
		/// Array div Array
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		inline Array<arrayType> operator/(const Array<arrayType> &other)const
		{
			rapidAssert(shape == other.shape, "Shapes must be equal to perform array addition");
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayArray(*this, other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x / y;
			});
			return res;
		}

		/// <summary>
		/// Array add Scalar
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t>
		inline Array<arrayType> operator+(const t &other)const
		{
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayScalar(*this, (arrayType) other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x + y;
			});
			return res;
		}

		/// <summary>
		/// Array sub Scalar
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t>
		inline Array<arrayType> operator-(const t &other)const
		{
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayScalar(*this, (arrayType) other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x - y;
			});
			return res;
		}

		/// <summary>
		/// Array mul Scalar
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t>
		inline Array<arrayType> operator*(const t &other)const
		{
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayScalar(*this, (arrayType) other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x * y;
			});
			return res;
		}

		/// <summary>
		/// Array div Scalar
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t>
		inline Array<arrayType> operator/(const t &other)const
		{
			auto res = Array<arrayType>(shape);
			Array<arrayType>::binaryOpArrayScalar(*this, (arrayType) other, res, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](arrayType x, arrayType y)
			{
				return x / y;
			});
			return res;
		}

		/// <summary>
		/// Fill an array with a scalar value
		/// </summary>
		/// <param name="val"></param>
		inline void fill(const arrayType &val)
		{
			Array<arrayType>::unaryOpArray(*this, *this, prod(shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [=](arrayType x)
			{
				return val;
			});
		}

		/// <summary>
		/// Calculate the dot product with another array. If the
		/// arrays are single-dimensional vectors, the vector product
		/// is used and a scalar value is returned. If the arrays are
		/// matrices, the matrix product is calculated. Otherwise, the
		/// dot product of the final two dimensions of the array are
		/// calculated.
		/// </summary>
		/// <param name="other"></param>
		/// <returns></returns>
		inline Array<arrayType> dot(const Array<arrayType> &other) const
		{
			rapidAssert(shape.size() == other.shape.size(), "Invalid number of dimensions for array dot product");
			uint64_t dims = shape.size();

		#ifndef RAPID_NO_BLAS
			switch (dims)
			{
				case 1:
				{
					rapidAssert(shape[0] == other.shape[0], "Invalid shape for array product");
					rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array product");

					Array<arrayType> res(shape);
					res.isZeroDim = true;
					res.dataStart[0] = imp::rapid_dot(shape[0], dataStart, other.dataStart);

					return res;
				}
				case 2:
				{
					rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot product");

					Array<arrayType> res({shape[0], other.shape[1]});

					const size_t M = shape[0];
					const size_t N = shape[1];
					const size_t K = other.shape[1];

					const arrayType *__restrict a = dataStart;
					const arrayType *__restrict b = other.dataStart;
					arrayType *__restrict c = res.dataStart;

					imp::rapid_gemm(M, N, K, a, b, c);

					return res;
				}
				default:
				{
					std::vector<uint64_t> resShape = shape;
					resShape[resShape.size() - 2] = shape[shape.size() - 2];
					resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
					Array<arrayType> res(resShape);

					for (uint64_t i = 0; i < shape[0]; i++)
					{
						res[i] = (operator[](i).dot(other[i]));
					}

					return res;
				}
			}
		#else
			switch (dims)
			{
				case 1:
				{
					rapidAssert(shape[0] == other.shape[0], "Invalid shape for array product");
					rapidAssert(isZeroDim == other.isZeroDim, "Invalid value for array product");

					Array<arrayType> res({1});
					res.isZeroDim = true;
					res.dataStart[0] = 0;

					for (uint64_t i = 0; i < shape[0]; i++)
						res.dataStart[0] += dataStart[i] * other.dataStart[i];

					return res;
				}
				case 2:
				{
					rapidAssert(shape[1] == other.shape[0], "Columns of A must match rows of B for dot product");
					uint64_t mode;
					uint64_t size = shape[0] * shape[1] * other.shape[1];

					if (size < 8000) mode = 0;
					else if (size < 64000000) mode = 1;
				#ifndef RAPID_NO_AMP
					else mode = 2;
				#else
					else mode = 1;
				#endif

					Array<arrayType> res({shape[0], other.shape[1]});

					if (mode == 0)
					{
						// Serial

						size_t M = shape[0];
						size_t N = shape[1];
						size_t K = other.shape[1];

						const arrayType *__restrict a = dataStart;
						const arrayType *__restrict b = other.dataStart;
						arrayType *__restrict c = res.dataStart;

						size_t i, j, k;
						arrayType tmp;

						for (i = 0; i < M; ++i)
						{
							for (j = 0; j < K; ++j)
							{
								tmp = 0;

								for (k = 0; k < N; ++k)
									tmp += a[k + i * N] * b[j + k * K];

								c[j + i * K] = tmp;
							}
						}
					}
					else if (mode == 1)
					{
						// Parallel

						auto M = (long long) shape[0];
						auto N = (long long) shape[1];
						auto K = (long long) other.shape[1];

						const arrayType *__restrict a = dataStart;
						const arrayType *__restrict b = other.dataStart;
						arrayType *__restrict c = res.dataStart;

						long long i, j, k;
						arrayType tmp;

					#pragma omp parallel for shared(M, N, K, a, b, c) private(i, j, k, tmp) default(none)
						for (i = 0; i < M; ++i)
						{
							for (j = 0; j < K; ++j)
							{
								tmp = 0;

								for (k = 0; k < N; ++k)
									tmp += a[k + i * N] * b[j + k * K];

								c[j + i * K] = tmp;
							}
						}
					}
				#ifndef RAPID_NO_AMP
					else if (mode == 2)
					{
						// Massive parallel

						// Tile size
						static const int TS = 32;

						const auto resizedThis = internal_resized({rapid::roundUp(shape[0], (size_t) TS), rapid::roundUp(shape[1], (size_t) TS)});
						const auto resizedOther = internal_resized({rapid::roundUp(other.shape[0], (size_t) TS), rapid::roundUp(other.shape[1], (size_t) TS)});
						res.internal_resize({rapid::roundUp(shape[0], (size_t) TS), rapid::roundUp(other.shape[1], (size_t) TS)});

						auto M = (unsigned int) resizedThis.shape[0];
						auto N = (unsigned int) resizedThis.shape[1];
						auto K = (unsigned int) res.shape[1];

						array_view<const arrayType, 2> a(M, N, resizedThis.dataStart);
						array_view<const arrayType, 2> b(N, K, resizedOther.dataStart);
						array_view<arrayType, 2> product(M, K, res.dataStart);

						parallel_for_each(product.extent.tile<TS, TS>(), [=](tiled_index<TS, TS> t_idx) restrict(amp)
						{
							// Get the location of the thread relative to the tile (row, col)
							// and the entire array_view (rowGlobal, colGlobal).
							const int row = t_idx.local[0];
							const int col = t_idx.local[1];
							const int rowGlobal = t_idx.global[0];
							const int colGlobal = t_idx.global[1];
							arrayType sum = 0;

							for (int i = 0; i < M; i += TS)
							{
								tile_static arrayType locA[TS][TS];
								tile_static arrayType locB[TS][TS];
								locA[row][col] = a(rowGlobal, col + i);
								locB[row][col] = b(row + i, colGlobal);

								t_idx.barrier.wait();

								for (int k = 0; k < TS; k++)
									sum += locA[row][k] * locB[k][col];

								t_idx.barrier.wait();
							}

							product[t_idx.global] = sum;
						});

						product.synchronize();

						res.internal_resize({shape[0], other.shape[1]});
					}
				#endif

					return res;
				}
				default:
				{
					std::vector<uint64_t> resShape = shape;
					resShape[resShape.size() - 2] = shape[shape.size() - 2];
					resShape[resShape.size() - 1] = other.shape[other.shape.size() - 1];
					Array<arrayType> res(resShape);

					for (uint64_t i = 0; i < shape[0]; i++)
					{
						res[i] = (operator[](i).dot(other[i]));
					}

					return res;
				}
			}
		#endif
		}

		/// <summary>
		/// Transpose an array and return the result. If the
		/// array is one dimensional, a vector is returned. The
		/// order in which the transpose occurs can be set with
		/// the "axes" parameter
		/// </summary>
		/// <param name="axes"></param>
		/// <returns></returns>
		inline Array<arrayType> transposed(const std::vector<uint64_t> &axes = std::vector<uint64_t>()) const
		{
			if (axes.size() != shape.size())
				RapidError("Transpose Error", "Invalid number of axes for array transpose").display();
			for (uint64_t i = 0; i < axes.size(); i++)
				if (std::count(axes.begin(), axes.end(), i) != 1)
					RapidError("Transpose Error", "Dimension does not appear only once").display();

			std::vector<uint64_t> newDims(shape.size());
			if (axes.empty())
				for (uint64_t i = 0; i < shape.size(); i++)
					newDims[i] = shape[shape.size() - i - 1];
			else
				for (uint64_t i = 0; i < shape.size(); i++)
					newDims[i] = shape[axes[i]];

			// Edge case for 1D array
			if (shape.size() == 1 || (axes.size() == 1 && axes[0] == 0))
			{
				auto res = Array<arrayType>(newDims);
				memcpy(res.dataStart, dataStart, sizeof(arrayType) * prod(newDims));
				return res;
			}

			if (shape.size() == 2)
			{
				auto res = Array<arrayType>(newDims);

				uint64_t rows = shape[0];
				uint64_t cols = shape[1];

				if (rows * cols < 62500)
				{
					for (uint64_t i = 0; i < rows; i++)
					{
						for (uint64_t j = 0; j < cols; j++)
						{
							res.dataStart[i + j * rows] = dataStart[j + i * cols];
						}
					}
				}
				else
				{
					int64_t i = 0, j = 0;
					const arrayType *__restrict thisData = dataStart;
					arrayType *__restrict resData = res.dataStart;
					auto minCols = rapid::max(cols, 3) - 3;

				#pragma omp parallel for private(i, j) shared(resData, thisData, minCols) default(none) num_threads(8)
					for (i = 0; i < rows; i++)
					{
						for (j = 0; j < minCols; j++)
						{
							int64_t p1 = i + j * rows;
							int64_t p2 = j + i * cols;

							resData[p1 + 0] = thisData[p2 + 0];
							resData[p1 + 1] = thisData[p2 + 1];
							resData[p1 + 2] = thisData[p2 + 2];
							resData[p1 + 3] = thisData[p2 + 3];
						}

						for (; j < cols; j++)
							resData[i + j * rows] = thisData[+i * cols];
					}
				}

				return res;
			}

			auto res = Array<arrayType>(newDims);

			std::vector<uint64_t> indices(shape.size(), 0);
			std::vector<uint64_t> indicesRes(shape.size(), 0);

			if (prod(shape) < 62000)
			{
				for (uint64_t i = 0; i < prod(shape); i++)
				{
					if (axes.empty())
						for (uint64_t j = 0; j < shape.size(); j++)
							indicesRes[j] = indices[shape.size() - j - 1];
					else
						for (uint64_t j = 0; j < shape.size(); j++)
							indicesRes[j] = indices[axes[j]];

					res.dataStart[imp::dimsToIndex(newDims, indicesRes)] = dataStart[imp::dimsToIndex(shape, indices)];

					indices[shape.size() - 1]++;
					uint64_t index = shape.size() - 1;

					while (indices[index] >= shape[index] && index > 0)
					{
						indices[index] = 0;
						index--;
						indices[index]++;
					}
				}
			}
			else
			{
				auto tmpShape = shape;
				for (int64_t i = 0; i < prod(tmpShape); i++)
				{
					if (axes.empty())
						for (int64_t j = 0; j < tmpShape.size(); j++)
							indicesRes[j] = indices[tmpShape.size() - j - 1];
					else
						for (int64_t j = 0; j < tmpShape.size(); j++)
							indicesRes[j] = indices[axes[j]];

					res.dataStart[imp::dimsToIndex(newDims, indicesRes)] = dataStart[imp::dimsToIndex(tmpShape, indices)];

					indices[tmpShape.size() - 1]++;
					int64_t index = tmpShape.size() - 1;

					while (indices[index] >= tmpShape[index] && index > 0)
					{
						indices[index] = 0;
						index--;
						indices[index]++;
					}
				}
			}

			return res;
		}

		/// <summary>
		/// Resize an array and return the result. The resulting data
		/// is linked to the parent data, so updating values will
		/// trigger an update in the parent/child array
		/// </summary>
		/// <param name="newShape"></param>
		/// <returns></returns>
		inline Array<arrayType> resized(const std::vector<uint64_t> &newShape) const
		{
			if (prod(newShape) != prod(shape))
				RapidError("Invalid Shape", "Invalid reshape size. Number of elements differ").display();

			bool zeroDim = false;

			if (isZeroDim && newShape.size() == 1)
				zeroDim = true;
			else
				zeroDim = false;

			(*originCount)++;
			auto res = Array<arrayType>::fromData(newShape, dataOrigin, dataStart, originCount, zeroDim);

			return res;
		}

		/// <summary>
		/// Resize an array inplace
		/// </summary>
		/// <param name="newShape"></param>
		inline void resize(const std::vector<uint64_t> &newShape)
		{
			if (prod(newShape) != prod(shape))
				RapidError("Invalid Shape", "Invalid reshape size. Number of elements differ").display();

			if (isZeroDim && newShape.size() == 1)
				isZeroDim = true;
			else
				isZeroDim = false;

			shape = newShape;
		}

		/// <summary>
		/// Create an exact copy of an array. The resulting array
		/// is not linked to the parent in any way, so an 
		/// </summary>
		/// <returns></returns>
		inline Array<arrayType> copy() const
		{
			Array<arrayType> res;
			res.isZeroDim = isZeroDim;
			res.shape = shape;
			res.dataStart = new arrayType[prod(shape)];
			res.dataOrigin = res.dataStart;
			res.originCount = new size_t;
			*(res.originCount) = 1;
			memcpy(res.dataStart, dataStart, sizeof(arrayType) * prod(shape));

			return res;
		}

		/// <summary>
		/// Get a string representation of an array
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <returns></returns>
		std::string toString() const;
	};

	/// <summary>
	/// Create a new array of the same size and dimensions as
	/// another array, but fill it with zeros.
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="other"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> zerosLike(const Array<t> &other)
	{
		auto res = Array<t>(other.shape);
		res.fill((t) 0);
		return res;
	}

	/// <summary>
	/// Reverse multiplication
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="val"></param>
	/// <param name="other"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> operator*(t val, const Array<t> &other)
	{
		auto res = Array<t>(other.shape);
		Array<t>::binaryOpScalarArray(val, other, res, prod(other.shape) > 10000 ? ExecutionType::PARALLEL : ExecutionType::SERIAL, [](t x, t y)
		{
			return x * y;
		});
		return res;
	}

	/// <summary>
	/// Sum all of the elements of an array
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="arr"></param>
	/// <returns></returns>
	template<typename t>
	inline t sum(const Array<t> &arr)
	{
		t res = 0;

		for (size_t i = 0; i < prod(arr.shape); i++)
			res += arr.dataStart[i];

		return res;
	}

	/// <summary>
	/// Calculate the exponent of every value
	/// in an array, and return the result
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="arr"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> exp(const Array<t> &arr)
	{
		Array<t> result(arr.shape);

		ExecutionType mode;
		if (prod(arr.shape) > 10000)
			mode = ExecutionType::PARALLEL;
		else
			mode = ExecutionType::SERIAL;

		Array<t>::unaryOpArray(arr, result, mode, [](t x)
		{
			return std::exp(x);
		});

		return result;
	}

	/// <summary>
	/// Square every element in an array and return
	/// the result
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="arr"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> square(const Array<t> &arr)
	{
		Array<t> result(arr.shape);

		ExecutionType mode;
		if (prod(arr.shape) > 10000)
			mode = ExecutionType::PARALLEL;
		else
			mode = ExecutionType::SERIAL;

		Array<t>::unaryOpArray(arr, result, mode, [](t x)
		{
			return x * x;
		});

		return result;
	}

	/// <summary>
	/// Square root every element in an array
	/// and return the result
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="arr"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> sqrt(const Array<t> &arr)
	{
		Array<t> result(arr.shape);

		ExecutionType mode;
		if (prod(arr.shape) > 10000)
			mode = ExecutionType::PARALLEL;
		else
			mode = ExecutionType::SERIAL;

		Array<t>::unaryOpArray(arr, result, mode, [](t x)
		{
			return std::sqrt(x);
		});

		return result;
	}

	/// <summary>
	/// Raise an array to a power
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="arr"></param>
	/// <param name="power"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> pow(const Array<t> &arr, t power)
	{
		Array<t> result(arr.shape);

		ExecutionType mode;
		if (prod(arr.shape) > 10000)
			mode = ExecutionType::PARALLEL;
		else
			mode = ExecutionType::SERIAL;

		Array<t>::unaryOpArray(arr, result, mode, [=](t x)
		{
			return std::pow(x, power);
		});

		return result;
	}

	/// <summary>
	/// Create a vector of a given length where the first element
	/// is "start" and the final element is "end", increasing in
	/// regular increments
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="start"></param>
	/// <param name="end"></param>
	/// <param name="len"></param>
	/// <returns></returns>
	template<typename s, typename e>
	inline Array<typename std::common_type<s, e>::type> linspace(s start, e end, size_t len)
	{
		using ct = typename std::common_type<s, e>::type;

		Array<ct> result({len});
		result.isZeroDim = len <= 1;

		if (len == 0)
			return result;

		if (len == 1)
		{
			result.dataStart[0] = start;
			return result;
		}

		ct inc = ((ct) end - (ct) start) / (ct) (len - 1);
		for (size_t i = 0; i < len; i++)
			result.dataStart[i] = (ct) start + (ct) i * inc;

		return result;
	}

	/// <summary>
	/// Create a vector of a specified type, where the values
	/// increase/decrease linearly between a start and end
	/// point by an specified
	/// </summary>
	/// <typeparam name="s"></typeparam>
	/// <typeparam name="e"></typeparam>
	/// <typeparam name="t"></typeparam>
	/// <param name="start"></param>
	/// <param name="end"></param>
	/// <param name="inc"></param>
	/// <returns></returns>
	template<typename s, typename e, typename t>
	inline Array<typename std::common_type<s, e, t>::type> arange(s start, e end, t inc = 1)
	{
		using ct = typename std::common_type<s, e, t>::type;

		auto len = (uint64_t) ceil(abs((ct) end - (ct) start) / (ct) inc);
		auto res = Array<t>({len});
		for (uint64_t i = 0; i < len; i++)
			res[i] = (ct) start + (ct) inc * (ct) i;
		return res;
	}

	/// <summary>
	/// Create a vector of a specified type, where the values
	/// increase/decrease linearly between a start and end
	/// point by an specified
	/// </summary>
	/// <typeparam name="e"></typeparam>
	/// <param name="end"></param>
	/// <returns></returns>
	template<typename e>
	inline Array<e> arange(e end)
	{
		return arange((e) 0, end, (e) 1);
	}

	/// <summary>
	/// Create a 3D array from two vectors, where the first element
	/// is vector A in row format, and the second element is vector
	/// B in column format.
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="a"></param>
	/// <param name="b"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> meshgrid(const Array<t> &a, const Array<t> &b)
	{
		rapidAssert(a.shape.size() == 1 && b.shape.size() == 1, "Invalid size for meshgrid. Must be a 1D array");
		Array<t> result({2, b.shape[0], a.shape[0]});

		if (prod(result.shape) < 10000)
		{
			for (int64_t i = 0; i < b.shape[0]; i++)
				for (int64_t j = 0; j < a.shape[0]; j++)
					result.setVal<int64_t>({(int64_t) 0, i, j}, a.accessVal<int64_t>({j}));

			for (int64_t i = 0; i < b.shape[0]; i++)
				for (int64_t j = 0; j < a.shape[0]; j++)
					result.setVal<int64_t>({(int64_t) 1, i, j}, b.accessVal<int64_t>({i}));
		}
		else
		{
		#pragma omp parallel for
			for (int64_t i = 0; i < b.shape[0]; i++)
				for (int64_t j = 0; j < a.shape[0]; j++)
					result.setVal<int64_t>({(int64_t) 0, i, j}, a.accessVal<int64_t>({j}));

		#pragma omp parallel for
			for (int64_t i = 0; i < b.shape[0]; i++)
				for (int64_t j = 0; j < a.shape[0]; j++)
					result.setVal<int64_t>({(int64_t) 1, i, j}, b.accessVal<int64_t>({i}));
		}

		return result;
	}

	/// <summary>
	/// Return a gaussian matrix with the given rows, columns and
	/// standard deviation.
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="r"></param>
	/// <param name="c"></param>
	/// <param name="sigma"></param>
	/// <returns></returns>
	template<typename t>
	inline Array<t> gaussian(size_t r, size_t c, t sigma)
	{
		t rows = (t) r;
		t cols = (t) c;

		auto ax = linspace<t>(-(rows - 1) / 2., (rows - 1) / 2., r);
		auto ay = linspace<t>(-(cols - 1) / 2., (cols - 1) / 2., c);
		auto mesh = meshgrid(ay, ax);
		auto xx = mesh[0];
		auto yy = mesh[1];

		auto kernel = exp(-0.5 * (square(xx) + square(yy)) / (sigma * sigma));
		return kernel / sum(kernel);
	}

	/// <summary>
	/// Cast an array from one type to another. This makes a copy of the array,
	/// and therefore altering a value in one will not cause an update in the
	/// other.
	/// </summary>
	/// <typeparam name="res"></typeparam>
	/// <typeparam name="src"></typeparam>
	/// <param name="src"></param>
	/// <returns></returns>
	template<typename resT, typename srcT>
	inline Array<resT> cast(const Array<srcT> &src)
	{
		Array<resT> res(src.shape);

		if (prod(src.shape) < 10000)
		{
			for (int64_t i = 0; i < prod(src.shape); i++)
				res.dataStart[i] = (resT) src.dataStart[i];
		}
		else
		{
		#pragma omp parallel for
			for (int64_t i = 0; i < prod(src.shape); i++)
				res.dataStart[i] = (resT) src.dataStart[i];
		}

		return res;
	}
}
