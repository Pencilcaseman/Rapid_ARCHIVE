// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

class Window : public rapid::RapidGraphics
{
public:
	rapid::ExpressionSolver solver;

	Window()
	{
		std::string text("(3*x^5+7*x^4+2*x^3-3*x^2-3*x-1)/2");
		solver = rapid::ExpressionSolver(text);
		solver.compile();

		create(500, 500, "Graphs for Dayz");
	}

	bool setup() override
	{
		limitFrameRate = false;
		transparent(0.4);

		return true;
	}

	bool draw() override
	{
		background(0);

		strokeWeight(1.5);
		stroke(100);

		line((double) width / 2, 0, (double) width / 2, height);
		line(0, (double) height / 2, width, (double) height / 2);

		strokeWeight(5);
		stroke(255, 0, 255);

		for (double x = 0; x < width; x += 0.1)
		{
			solver.variables["x"] = rapid::map(x, 0, width, -2, 2);
			solver.variables["y"] = sin((double) frameCount / 500);
			auto y = solver.eval();
		
			point(x, rapid::map(y, -2, 2, height, 0));
		}

		std::cout << frameRate() << "\n";

		return true;
	}
};

int main()
{
	Window window;
	window.start();

	std::cout << std::fixed;

	auto start = TIME;
	std::string text("(3*x^5+7*x^4+2*x^3-3*x^2-3*x-1)/2");
	rapid::ExpressionSolver solver(text);

	solver.registerFunction("sin", [](double a)
	{
		return sin(a);
	});

	solver.expressionToInfix();
	solver.infixToPostfix();
	solver.postfixProcess();
	auto end = TIME;

	std::cout << "Calculated in " << end - start << " seconds\n";
	
	for (const auto &token : solver.infix)
		std::cout << token << " ";
	std::cout << "\n";
	
	for (const auto &token : solver.postfix)
		std::cout << token << " ";
	std::cout << "\n";

	for (const auto &token : solver.processed)
		if (token.second.length() == 0)
			std::cout << token.first << " ";
		else
			std::cout << token.second << " ";
	std::cout << "\n";

	solver.variables["x"] = -1;

	start = TIME;
	auto res = solver.postfixEval();
	end = TIME;

	std::cout << "Result: " << solver.postfixEval() << "\n";
	std::cout << "Calcualated in " << end - start << " seconds\n";

	return 0;
}
