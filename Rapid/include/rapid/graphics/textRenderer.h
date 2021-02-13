#pragma once

#include "../internal.h"
#include "messageBoxCore.h"

#include <ft2build.h>
#include FT_FREETYPE_H

namespace rapid
{
	FT_Library freeTypeLibrary;

	void initializeFreeType()
	{
		if (FT_Init_FreeType(&freeTypeLibrary))
		{
			RapidError("Font Error", "Could not initialize FreeType").display();
		}
	}

	void loadFont(std::string font)
	{
		FT_Face face;
		if (FT_New_Face(freeTypeLibrary, (std::string("C:/Windows/Fonts/") + font + ".ttf").c_str(), 0, &face))
		{
			RapidError("Font Error", "Could not load font " + font).display();
		}

		FT_Set_Pixel_Sizes(face, 0, 48);
	}
}
