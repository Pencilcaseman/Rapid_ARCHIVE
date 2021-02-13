#pragma once

#define GLEW_STATIC

#include "graphicsCore.h"

namespace rapid
{
	void keyPressCallback(GLFWwindow *window_, int key, int scancode, int action, int mods);

	struct Color
	{
		uint8_t r = 0;
		uint8_t g = 0;
		uint8_t b = 0;
	};

	class PixelEngine
	{
	private:
		GLFWwindow *window = nullptr;
		double aspectRatio = 1;
		double msPerFrame = 0;
		double prevFrameTime = 0;
		double timeCurrent = 0;
		double timeStart = 0;
		bool windowShouldClose = false;
		std::string title = "PixelEngine Window";

		bool shapeStarted = false;
		bool shapeEnded = true;

		unsigned char fillR = 255, fillG = 255, fillB = 255;
		unsigned char strokeR = 255, strokeG = 255, strokeB = 255;
		double strokeWeightVal;

		void internalCloseWindow()
		{
			glfwTerminate();
			windowShouldClose = true;
		}

	public:
		int width = 100, height = 100;
		int pixelWidth = 100, pixelHeight = 100;
		int pixelsX;
		int pixelsY;

		double mouseX = 0, mouseY = 0;
		int windowX = 0, windowY = 0;
		int screenWidth = 0, screenHeight = 0;
		int screenX = 0, screenY = 0;
		unsigned long long frameCount = 0;
		double targetFrameRate = 60;
		bool limitFrameRate = true;
		pollMode eventPollMode = POLL;
		bool callbackWhenNotFocused = false;

		Color *pixels;

		KeyDefinition internal_keyDef{};
		bool keyDefSet = false;

		PixelEngine() = default;

		void create(int windowWidth = 100, int windowHeight = 100, int windowPixelWidth = 5, int windowPixelHeight = 5, const std::string &windowTitle = "PixelEngine Window")
		{
			width = windowWidth;
			height = windowHeight;
			pixelWidth = windowPixelWidth;
			pixelHeight = windowPixelHeight;

			pixelsX = roundUp(width, pixelWidth) / pixelWidth;
			pixelsY = roundUp(height, pixelHeight) / pixelHeight;

			title = windowTitle;
			aspectRatio = double(width) / double(height);

			pixels = new Color[windowWidth * windowHeight];
		}

		~PixelEngine()
		{
			glfwTerminate();
		}

		bool initialize()
		{
			rapidValidate(glfwInit(), "Unable to initialize GLFW");

			glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

			window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);

			if (!window)
			{
				internalCloseWindow();

				std::cerr << "Unable to initialize GLFW window\n";
				exit(1);
			}

			glfwMakeContextCurrent(window);
			glfwSwapInterval(0);

			rapidValidate(glewInit() == GLEW_OK, "Unable to initialize glew");

			aspectRatio = double(width) / double(height);
			timeStart = TIME;
			prevFrameTime = timeStart;

			glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), &screenX, &screenY, &screenWidth, &screenHeight);

			glfwSetKeyCallback(window, keyPressCallback);

			windowShouldClose = false;

			return true;
		}

		void setWindowTitle(const std::string &title)
		{
			glfwSetWindowTitle(window, title.c_str());
		}

		bool update()
		{
			if (windowShouldClose)
				return false;

			glfwMakeContextCurrent(window);

			switch (eventPollMode)
			{
				case POLL:
					glfwPollEvents();
					break;
				case WAIT:
					glfwWaitEvents();
					break;
				case TIMEOUT:
					glfwWaitEventsTimeout(1. / targetFrameRate);
					break;
				default:
					rapidValidate(false, "Invalid poll method");
					break;
			}

			// Update internal variables
			glfwGetCursorPos(window, &mouseX, &mouseY);
			glfwGetWindowPos(window, &windowX, &windowY);

			if (!draw())
			{
				internalCloseWindow();
				return false;
			}

			glBegin(GL_TRIANGLES);

			for (int y = 0; y < pixelsY; y++)
			{
				for (int x = 0; x < pixelsX; x++)
				{
					// Set the fill color
					auto pix = pixels[x + y * pixelsX];

					// glColor3ub(pix.r, pix.g, pix.b);
					glColor3ub(255, 0, 0);

					// Draw two triangles for the rectangle
					double x1, x2;
					double y1, y2;
					x1 = (double) x * (double) pixelWidth;
					x2 = (double) (x + 1) * (double) pixelWidth;

					y1 = (double) (y + 1) * (double) pixelHeight;
					y2 = (double) (y) * (double) pixelHeight;

					x1 = map(x1, 0, width, -1, 1);
					x2 = map(x2, 0, width, -1, 1);

					y1 = map(y1, 0, height, -1, 1);
					y2 = map(y2, 0, height, -1, 1);

					glVertex2d(x1, y1);
					glVertex2d(x2, y2);
					glVertex2d(x1, y2);

					glColor3ub(0, 255, 0);

					glVertex2d(x1, y1);
					glVertex2d(x2, y2);
					glVertex2d(x2, y1);
				}
			}

			glEnd();

			if (keyDefSet)
			{
				onKeyPress(internal_keyDef.key, internal_keyDef.action, internal_keyDef.mod);
				keyDefSet = false;
			}

			glfwSwapBuffers(window);

			if (limitFrameRate)
				while (TIME - timeCurrent < (1. / (targetFrameRate * 1.00001)))
				{
				}

			timeCurrent = TIME;
			auto deltaTime = timeCurrent - prevFrameTime;
			msPerFrame = (deltaTime) / 1000;
			prevFrameTime = timeCurrent;

			frameCount++;

			if (glfwWindowShouldClose(window))
			{
				internalCloseWindow();
				return false;
			}

			return true;
		}

		inline void close()
		{
			internalCloseWindow();
		}

		bool start()
		{
			// Run the setup script
			if (!setup())
			{
				internalCloseWindow();
				return false;
			}

			// This should never fail, but who knows
			if (!initialize())
			{
				internalCloseWindow();
				return false;
			}

			// Run the update loop
			while (isOpen())
			{
				if (!update())
				{
					internalCloseWindow();
					return false;
				}
			}

			internalCloseWindow();

			return true;
		}

		inline virtual bool onKeyPress(keyType key, keyAction action, const std::vector<keyModifier> &mods)
		{
			return true;
		}

		inline virtual bool setup()
		{
			return true;
		}

		inline virtual bool draw()
		{
			return true;
		}

		inline virtual bool final()
		{
			return true;
		}

		inline bool isOpen() const
		{
			return !windowShouldClose;
		}

		inline bool isFocused() const
		{
			return glfwGetWindowAttrib(window, GLFW_FOCUSED);
		}

		inline bool isPressed(keyType key) const
		{
			return glfwGetKey(window, key) == ACTION_PRESS;
		}

		inline double frameRate() const
		{
			return 1. / (msPerFrame * 1000);
		}

		inline double avgFrameRate() const
		{
			return double(frameCount) / (timeCurrent - timeStart);
		}

		inline double time() const
		{
			return timeCurrent - timeStart;
		}

		inline static void background(unsigned short b)
		{
			glClearColor(float(b) / 255.f, float(b) / 255.f, float(b) / 255.f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT);
		}

		inline static void background(unsigned short r, unsigned short g, unsigned short b)
		{
			glClearColor(float(r) / 255.f, float(g) / 255.f, float(b) / 255.f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT);
		}

		inline void pixel(long x, long y, Color p)
		{
			if (x < 0 || x > pixelsX)
				return;

			if (y < 0 || y > pixelsY)
				return;

			pixels[x + y * pixelsX] = p;
		}

		inline void transparent(float opacity) const
		{
			glfwSetWindowOpacity(window, opacity);
		}

		inline void moveTo(const int x, const int y)
		{
			glfwSetWindowPos(window, x, y);

			// Update internal variables
			glfwGetCursorPos(window, &mouseX, &mouseY);
			glfwGetWindowPos(window, &windowX, &windowY);
		}

		inline void showCursor() const
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}

		inline void hideCursor() const
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
		}

		inline void lockCursor() const
		{
			glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}

		inline bool mousePressed(int button = 0) const
		{
			return glfwGetMouseButton(window, button);
		}

		inline GLFWwindow *internal_window_() const
		{
			return window;
		}
	};
}
