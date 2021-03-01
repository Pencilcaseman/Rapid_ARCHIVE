// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	auto arr = rapid::Array<double>({2, 2, 2});
	arr.fill(3.14);

	auto arr2 = rapid::Array<double>({2});
	arr2.fill(123);

	arr[0][0] = arr2;

	arr[0][0][0] = 69;

	std::cout << arr.toString() << "\n\n";
	std::cout << arr[0].toString() << "\n\n";
	std::cout << arr2.toString() << "\n\n";

	START_TIMER(0, 100000);

	auto testArr = rapid::Array<double>({1000, 10000});
	// testArr.fill(3.14);

	auto testArr2 = rapid::Array<double>({10000});
	// testArr2.fill(123);

	testArr[0] = testArr2;
	testArr[0][0] = 69;

	END_TIMER(0);

	return 0;
}
