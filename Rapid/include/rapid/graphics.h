#pragma once

#ifndef RAPID_NO_GRAPHICS			// Optional graphics include as libs might not work on some operating systems
#include "graphics/graphicsCore.h"
#include "graphics/CImg/CImg.h"
#endif

#define OLC_PGE_APPLICATION
#include "graphics/olcPGE/olcPixelGameEngine.h" // olcPGE is awesome, and works cross-platform, so include it regardless
