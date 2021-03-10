// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

template<typename dtype>
inline std::vector<uint64_t> extractShape(const std::initializer_list<dtype> &data, std::vector<uint64_t> *shape = nullptr)
{
	if (shape == nullptr)
		shape = new std::vector<uint64_t>();

	shape->emplace_back(data.size());

	return *shape;
}

template<typename dtype>
inline std::vector<uint64_t> extractShape(const std::initializer_list<std::initializer_list<dtype>> &data, std::vector<uint64_t> *shape = nullptr)
{
	if (shape == nullptr)
		shape = new std::vector<uint64_t>();

	shape->emplace_back(data.size());
	extractShape<dtype>(*(data.begin()), shape);

	return *shape;
}

int main()
{
	auto a = rapid::Array<float>({100000});
	auto b = rapid::Array<float>({100000});
	a.fill(1);
	b.fill(1);

	START_TIMER(0, 10000);
	auto res = a.dot(b);
	END_TIMER(0);

	std::cout << a.toString() << "\n";
	std::cout << a.dot(b).toString() << "\n";

	START_TIMER(1, 100000);
	auto x = rapid::Array<double>::fromData({{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}});
	END_TIMER(1);

	auto x = rapid::Array<double>::fromData({{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}});
	std::cout << x.toString() << "\n";

	return 0;
}
