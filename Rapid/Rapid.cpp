#define RAPID_NO_BLAS
//#define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	auto e = rapid::Array<double>({10000, 10000});
	auto f = rapid::Array<double>({10000, 10000});
	e.fill(1);
	f.fill(1);

	// std::cout << e.toString() << "\n\n";
	// std::cout << f.toString() << "\n\n";

	START_TIMER(0, 100);
	auto res = e.dot(f);
	END_TIMER(0);
	
	// std::cout << e.toString() << "\n";
	// std::cout << e.dot(f).toString() << "\n";

	return 0;
}
