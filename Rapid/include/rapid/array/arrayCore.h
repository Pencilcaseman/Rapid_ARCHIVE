#pragma once

#include "../internal.h"
#include "../math.h"

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

			auto lastDecimal = stream.str().find_last_of(".");

			if (std::is_floating_point<t>::value && lastDecimal == std::string::npos)
			{
				stream << ".";
				lastDecimal = stream.str().length() - 1;
			}

			auto lastZero = stream.str().find_last_of("0");

			// Value is integral
			if (lastDecimal == std::string::npos)
				return {stream.str(), stream.str().length() - 1};

			return {stream.str(), lastDecimal};
		}

		template<typename indexT, typename shapeT>
		inline indexT ndToScalar(const std::vector<indexT> &index, const std::vector<shapeT> &shape)
		{
			indexT sig = 1;
			indexT pos = 0;

			for (indexT i = shape.size() - 1; i >= 0; i--)
			{
				pos += (i < index.size() ? index[i] : 0) * sig;
				sig *= shape[i];
			}

			return pos;
		}

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

			if (arrShape.size() == 0 || prod(arrShape) == 0)
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

		// /// <summary>
		// /// Make one array equal to another. This will create an array
		// /// using exactly the same data as the provided array, meaning
		// /// a modification in one will result in a change in the other.
		// /// </summary>
		// /// <param name="other"></param>
		// /// <returns></returns>
		// Array<arrayType> operator=(const Array<arrayType> &other)
		// {
		// 	isZeroDim = other.isZeroDim;
		// 	shape = other.shape;
		// 	dataStart = other.dataStart;
		// 	dataOrigin = other.dataOrigin;
		// 	originCount = other.originCount;
		// 
		// 	if (this != &other)
		// 		(*originCount)++;
		// 
		// 	return *this;
		// }

		/// <summary>
		/// Set a zero-dimensional array object to a scalar value.
		/// This should be used for array setter functions.
		/// </summary>
		/// <typeparam name="t"></typeparam>
		/// <param name="other"></param>
		/// <returns></returns>
		template<typename t>
		Array<arrayType> &operator=(const t &other)
		{
			rapidAssert(isZeroDim, "Cannot set a multidimensional array to a scalar value");
			dataStart[0] = (arrayType) other;
			return *this;
		}

		Array<arrayType> &operator=(Array<arrayType> &other)
		{
			rapidAssert(shape == other.shape, "Invalid shape for array setting");

			memcpy(dataStart, other.dataStart, prod(shape) * sizeof(arrayType));

			// Only delete data if originCount becomes zero
			(*(other.originCount))--;

			if ((*(other.originCount)) == 0)
			{
				delete[] other.dataOrigin;
				delete other.originCount;
			}

			other.dataOrigin = dataOrigin;
			other.dataStart = dataStart;
			other.originCount = originCount;

			(*originCount)++;

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
			if (std::is_integral<t>::value || std::is_floating_point<t>::value)
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
	/// Reverse multiplication
	/// </summary>
	/// <typeparam name="t"></typeparam>
	/// <param name="val"></param>
	/// <param name="other"></param>
	/// <returns></returns>
	template<typename t>
	static inline Array<t> operator*(t val, const Array<t> &other)
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
	template<typename t>
	inline Array<t> linspace(t start, t end, size_t len)
	{
		Array<t> result({len});

		if (len == 0)
			return result;

		if (len == 1)
		{
			result.dataStart[0] = start;
			return result;
		}

		t inc = (end - start) / (t) (len - 1);
		for (size_t i = 0; i < len; i++)
			result.dataStart[i] = start + (t) i * inc;

		return result;
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
