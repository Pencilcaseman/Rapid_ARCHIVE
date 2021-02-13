// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	std::string text("SET x TO 10+10/5");
	for (const auto &token : rapid::splitExpression(text, {" ", "+", "-", "*", "/"}))
		std::cout << "Token: " << token << "\n";

	return 0;
}
