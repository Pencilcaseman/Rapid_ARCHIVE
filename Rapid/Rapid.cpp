// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	auto a = rapid::Array<double>({10000, 10000});
	auto b = rapid::Array<double>({10000, 10000});
	a.fill(1);
	b.fill(1);

	START_TIMER(0, 1);
	auto res = a.dot(b);
	END_TIMER(0);

	std::cout << a.toString() << "\n";
	std::cout << a.dot(b).toString() << "\n";

	return 0;
}
