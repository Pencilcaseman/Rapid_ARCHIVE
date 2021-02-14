// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

int main()
{
	std::string text("(5+5)");
	rapid::ExpressionSolver solver(text);
	solver.expressionToInfix();
	solver.infixToPostfix();
	
	for (const auto &token : solver.infix)
		std::cout << token << " ";
	std::cout << "\n";
	
	for (const auto &token : solver.postfix)
		std::cout << token << " ";
	std::cout << "\n";

	return 0;
}
