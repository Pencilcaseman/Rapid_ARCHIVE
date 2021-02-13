// #define RAPID_NO_BLAS
// #define RAPID_NO_AMP
// #define RAPID_NO_GRAPHICS
// #define RAPID_CHECK_NAN

#include <iostream>
#include "include/rapid/rapid.h"

class PixelWindow : public rapid::PixelEngine
{
public:
	PixelWindow()
	{}

	bool setup() override
	{
		create(400, 400, 4, 4, "This is a window");

		pixel(10, 10, {255, 255, 255});

		limitFrameRate = false;

		return true;
	}

	bool draw() override
	{
		if (frameCount % 100 == 0)
			std::cout << "FPS: " << frameRate() << "\n";

		return true;
	}
};

int main()
{
	auto x = rapid::Array<double>({3, 4, 2});
	std::cout << x.toString() << "\n";

	return 0;
}
