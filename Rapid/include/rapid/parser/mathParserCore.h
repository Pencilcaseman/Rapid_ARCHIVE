#pragma once

#include "../internal.h"

namespace rapid
{
	std::vector<std::string> splitExpression(const std::string &expression, const std::vector<std::string> delimiters)
	{
		std::vector<std::string> res;

		uint64_t start = 0;
		uint64_t end = 0;

		while (end != std::string::npos)
		{
			// Find the nearest delimiter
			uint64_t nearest = -1;
			uint64_t index = 0;

			for (uint64_t i = 0; i < delimiters.size(); i++)
			{
				auto pos = expression.find(delimiters[i], start);
				if (pos != std::string::npos && pos < nearest)
				{
					nearest = pos;
					index = i;
				}
			}

			if (nearest == (uint64_t) -1) // Nothing else was found
				break;

			end = nearest;
			res.emplace_back(std::string(expression.begin() + start, expression.begin() + end));
			start = end + 1;
		}

		res.emplace_back(std::string(expression.begin() + start, expression.end()));

		return res;
	}
}