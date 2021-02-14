#pragma once

#include "../internal.h"
#include "utils.h"

namespace rapid
{
	class Term
	{
	public:

	};

	class ExpressionSolver
	{
	public:
		std::string expression;
		std::vector<std::string> infix;
		std::vector<std::string> postfix;

		std::vector<std::string> splitBy = {" ", "(", ")", "+", "-", "*", "/", "^"};
		std::vector<std::string> precedence = {"sin", "cos", "tan", "^", "/", "*", "+", "-"};
		std::unordered_map<std::string, Term> variables;

	public:

		ExpressionSolver(const std::string &expr) : expression(expr)
		{}

		inline void expressionToInfix()
		{
			for (const auto &term : splitString(expression, splitBy))
			{
				if (term != " " && term != "")
					infix.emplace_back(term);
			}
		}

		inline void infixToPostfix()
		{
			std::stack<std::string> stack;

			for (const auto &token : infix)
			{
				auto it = std::find(precedence.begin(), precedence.end(), token);

				if (it == precedence.end())
				{
					stack.push(token);
				}
				else
				{
					if (isalphanum(token))
						postfix.emplace_back(token);
					else if (token == "(" || token == "^")
						stack.push(token);
					else if (token == ")")
					{
						while (stack.size() > 0 && stack.top() != "(")
						{
							postfix.emplace_back(stack.top());
							stack.pop();
						}
						stack.pop();
					}
					else
					{
						while (stack.size() > 0 && std::find(precedence.begin(), precedence.end(), token) >= std::find(precedence.begin(), precedence.end(), stack.top()))
						{
							postfix.emplace_back(stack.top());
							stack.pop();
						}
						stack.push(token);
					}
				}
			}

			while (stack.size() > 0)
			{
				postfix.emplace_back(stack.top());
				stack.pop();
			}
		}
	};
}