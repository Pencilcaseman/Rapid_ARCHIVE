#pragma once

#include "../internal.h"

#define GLEW_STATIC

#include "glew/include/GL/glew.h"
#include "GLFW/glfw3.h"
#include "../math.h"

namespace rapid
{
	void keyPressCallback(GLFWwindow *window_, int key, int scancode, int action, int mods);

	class RapidGraphics;

	std::vector<RapidGraphics *> graphicsInstances;

	unsigned long long rapidGraphicsWindowsExposed = 0;

	enum pollMode
	{
		POLL = 0, WAIT = 1, TIMEOUT = 2
	};

	enum shapeTypes
	{
		TRIANGLES = GL_TRIANGLES,
		TRIANGLE_STRIP = GL_TRIANGLE_STRIP,
		TRIANGLE_FAN = GL_TRIANGLE_FAN,
		POINTS = GL_POINTS,
		QUADS = GL_QUADS,
		QUAD_STRIP = GL_QUAD_STRIP,
		POLYGON = GL_POLYGON
	};

	enum keyType
	{
		KEY_SPACE = GLFW_KEY_SPACE,
		KEY_APOSTROPHE = GLFW_KEY_APOSTROPHE,
		KEY_COMMA = GLFW_KEY_COMMA,
		KEY_MINUS = GLFW_KEY_MINUS,
		KEY_PERIOD = GLFW_KEY_PERIOD,
		KEY_SLASH = GLFW_KEY_SLASH,
		KEY_0 = GLFW_KEY_0,
		KEY_1 = GLFW_KEY_1,
		KEY_2 = GLFW_KEY_2,
		KEY_3 = GLFW_KEY_3,
		KEY_4 = GLFW_KEY_4,
		KEY_5 = GLFW_KEY_5,
		KEY_6 = GLFW_KEY_6,
		KEY_7 = GLFW_KEY_7,
		KEY_8 = GLFW_KEY_8,
		KEY_9 = GLFW_KEY_9,
		KEY_SEMICOLON = GLFW_KEY_SEMICOLON,
		KEY_EQUAL = GLFW_KEY_EQUAL,
		KEY_A = GLFW_KEY_A,
		KEY_B = GLFW_KEY_B,
		KEY_C = GLFW_KEY_C,
		KEY_D = GLFW_KEY_D,
		KEY_E = GLFW_KEY_E,
		KEY_F = GLFW_KEY_F,
		KEY_G = GLFW_KEY_G,
		KEY_H = GLFW_KEY_H,
		KEY_I = GLFW_KEY_I,
		KEY_J = GLFW_KEY_J,
		KEY_K = GLFW_KEY_K,
		KEY_L = GLFW_KEY_L,
		KEY_M = GLFW_KEY_M,
		KEY_N = GLFW_KEY_N,
		KEY_O = GLFW_KEY_O,
		KEY_P = GLFW_KEY_P,
		KEY_Q = GLFW_KEY_Q,
		KEY_R = GLFW_KEY_R,
		KEY_S = GLFW_KEY_S,
		KEY_T = GLFW_KEY_T,
		KEY_U = GLFW_KEY_U,
		KEY_V = GLFW_KEY_V,
		KEY_W = GLFW_KEY_W,
		KEY_X = GLFW_KEY_X,
		KEY_Y = GLFW_KEY_Y,
		KEY_Z = GLFW_KEY_Z,
		KEY_LEFT_BRACKET = GLFW_KEY_LEFT_BRACKET,
		KEY_BACKSLASH = GLFW_KEY_BACKSLASH,
		KEY_RIGHT_BRACKET = GLFW_KEY_RIGHT_BRACKET,
		KEY_GRAVE_ACCENT = GLFW_KEY_GRAVE_ACCENT,
		KEY_ESCAPE = GLFW_KEY_ESCAPE,
		KEY_ENTER = GLFW_KEY_ENTER,
		KEY_TAB = GLFW_KEY_TAB,
		KEY_BACKSPACE = GLFW_KEY_BACKSPACE,
		KEY_INSERT = GLFW_KEY_INSERT,
		KEY_DELETE = GLFW_KEY_DELETE,
		KEY_RIGHT = GLFW_KEY_RIGHT,
		KEY_LEFT = GLFW_KEY_LEFT,
		KEY_DOWN = GLFW_KEY_DOWN,
		KEY_UP = GLFW_KEY_UP,
		KEY_PAGE_UP = GLFW_KEY_PAGE_UP,
		KEY_PAGE_DOWN = GLFW_KEY_PAGE_DOWN,
		KEY_HOME = GLFW_KEY_HOME,
		KEY_END = GLFW_KEY_END,
		KEY_CAPS_LOCK = GLFW_KEY_CAPS_LOCK,
		KEY_SCROLL_LOCK = GLFW_KEY_SCROLL_LOCK,
		KEY_NUM_LOCK = GLFW_KEY_NUM_LOCK,
		KEY_PRINT_SCREEN = GLFW_KEY_PRINT_SCREEN,
		KEY_PAUSE = GLFW_KEY_PAUSE,
		KEY_F1 = GLFW_KEY_F1,
		KEY_F2 = GLFW_KEY_F2,
		KEY_F3 = GLFW_KEY_F3,
		KEY_F4 = GLFW_KEY_F4,
		KEY_F5 = GLFW_KEY_F5,
		KEY_F6 = GLFW_KEY_F6,
		KEY_F7 = GLFW_KEY_F7,
		KEY_F8 = GLFW_KEY_F8,
		KEY_F9 = GLFW_KEY_F9,
		KEY_F10 = GLFW_KEY_F10,
		KEY_F11 = GLFW_KEY_F11,
		KEY_F12 = GLFW_KEY_F12,
		KEY_F13 = GLFW_KEY_F13,
		KEY_F14 = GLFW_KEY_F14,
		KEY_F15 = GLFW_KEY_F15,
		KEY_F16 = GLFW_KEY_F16,
		KEY_F17 = GLFW_KEY_F17,
		KEY_F18 = GLFW_KEY_F18,
		KEY_F19 = GLFW_KEY_F19,
		KEY_F20 = GLFW_KEY_F20,
		KEY_F21 = GLFW_KEY_F21,
		KEY_F22 = GLFW_KEY_F22,
		KEY_F23 = GLFW_KEY_F23,
		KEY_F24 = GLFW_KEY_F24,
		KEY_F25 = GLFW_KEY_F25,
		KEY_NUMPAD_0 = GLFW_KEY_KP_0,
		KEY_NUMPAD_1 = GLFW_KEY_KP_1,
		KEY_NUMPAD_2 = GLFW_KEY_KP_2,
		KEY_NUMPAD_3 = GLFW_KEY_KP_3,
		KEY_NUMPAD_4 = GLFW_KEY_KP_4,
		KEY_NUMPAD_5 = GLFW_KEY_KP_5,
		KEY_NUMPAD_6 = GLFW_KEY_KP_6,
		KEY_NUMPAD_7 = GLFW_KEY_KP_7,
		KEY_NUMPAD_8 = GLFW_KEY_KP_8,
		KEY_NUMPAD_9 = GLFW_KEY_KP_9,
		KEY_NUMPAD_DECIMAL = GLFW_KEY_KP_DECIMAL,
		KEY_NUMPAD_DIVIDE = GLFW_KEY_KP_DIVIDE,
		KEY_NUMPAD_MULTIPLY = GLFW_KEY_KP_MULTIPLY,
		KEY_NUMPAD_SUBTRACT = GLFW_KEY_KP_SUBTRACT,
		KEY_NUMPAD_ADD = GLFW_KEY_KP_ADD,
		KEY_NUMPAD_ENTER = GLFW_KEY_KP_ENTER,
		KEY_NUMPAD_EQUAL = GLFW_KEY_KP_EQUAL,
		KEY_LEFT_SHIFT = GLFW_KEY_LEFT_SHIFT,
		KEY_LEFT_CONTROL = GLFW_KEY_LEFT_CONTROL,
		KEY_LEFT_ALT = GLFW_KEY_LEFT_ALT,
		KEY_LEFT_SUPER = GLFW_KEY_LEFT_SUPER,
		KEY_RIGHT_SHIFT = GLFW_KEY_RIGHT_SHIFT,
		KEY_RIGHT_CONTROL = GLFW_KEY_RIGHT_CONTROL,
		KEY_RIGHT_ALT = GLFW_KEY_RIGHT_ALT,
		KEY_RIGHT_SUPER = GLFW_KEY_RIGHT_SUPER,
		KEY_MENU = GLFW_KEY_MENU
	};

	enum keyAction
	{
		ACTION_PRESS = GLFW_PRESS,
		ACTION_RELEASE = GLFW_RELEASE
	};

	enum keyModifier
	{
		MODIFIER_ALT = GLFW_MOD_ALT,
		MODIFIER_CAPS = GLFW_MOD_CAPS_LOCK,
		MODIFIER_CONTROL = GLFW_MOD_CAPS_LOCK,
		MODIFIER_NUM_LOCK = GLFW_MOD_NUM_LOCK,
		MODIFIER_SHIFT = GLFW_MOD_SHIFT
	};

	struct KeyDefinition
	{
		keyType key;
		keyAction action;
		std::vector<keyModifier> mod;
	};

	class RapidGraphics
	{
	private:
		GLFWwindow *window = nullptr;
		double aspectRatio = 1;
		double msPerFrame = 0;
		double prevFrameTime = 0;
		double timeCurrent = 0;
		double timeStart = 0;
		bool windowShouldClose = false;
		std::string title = "RapidGraphics Window";

		bool shapeStarted = false;
		bool shapeEnded = true;

		unsigned char fillR = 255, fillG = 255, fillB = 255;
		unsigned char strokeR = 255, strokeG = 255, strokeB = 255;
		double strokeWeightVal;

		void internalCloseWindow()
		{
			rapidGraphicsWindowsExposed--;

			if (rapidGraphicsWindowsExposed == 0)
				glfwTerminate();
			else
				glfwDestroyWindow(window);

			windowShouldClose = true;
		}

	public:
		int width = 100, height = 100;
		double mouseX = 0, mouseY = 0;
		int windowX = 0, windowY = 0;
		int screenWidth = 0, screenHeight = 0;
		int screenX = 0, screenY = 0;
		unsigned long long frameCount = 0;
		double targetFrameRate = 60;
		bool limitFrameRate = true;
		pollMode eventPollMode = POLL;
		bool callbackWhenNotFocused = false;

		KeyDefinition internal_keyDef{};
		bool keyDefSet = false;

		RapidGraphics() = default;

		void create(int windowWidth = 100, int windowHeight = 100, const std::string &windowTitle = "RapidGraphics Window")
		{
			width = windowWidth;
			height = windowHeight;
			title = windowTitle;
			aspectRatio = double(width) / double(height);
		}

		~RapidGraphics()
		{
			if (rapidGraphicsWindowsExposed == 0)
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

			if (!setup())
			{
				windowShouldClose = true;
			}
			else
			{
				windowShouldClose = false;
			}

			aspectRatio = double(width) / double(height);
			timeStart = TIME;
			prevFrameTime = timeStart;

			glfwGetMonitorWorkarea(glfwGetPrimaryMonitor(), &screenX, &screenY, &screenWidth, &screenHeight);

			glfwSetKeyCallback(window, keyPressCallback);

			graphicsInstances.push_back(this);

			rapidGraphicsWindowsExposed++;

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

		template<typename t>
		inline void fill(const t &r, const t &g, const t &b)
		{
			if (std::is_floating_point<t>::value)
			{
				// Fill in range (0, 1)
				fillR = (unsigned char) (r * 255);
				fillG = (unsigned char) (g * 255);
				fillB = (unsigned char) (b * 255);
			}
			else
			{
				// Fill in range (0, 255)
				fillR = (unsigned char) r;
				fillG = (unsigned char) g;
				fillB = (unsigned char) b;
			}
		}

		template<typename t>
		inline void fill(const t &b)
		{
			fill(b, b, b);
		}

		template<typename t>
		inline void stroke(const t &r, const t &g, const t &b)
		{
			if (std::is_floating_point<t>::value)
			{
				// Stroke in range (0, 1)
				strokeR = (unsigned char) (r * 255);
				strokeG = (unsigned char) (g * 255);
				strokeB = (unsigned char) (b * 255);
			}
			else
			{
				// Stroke in range (0, 255)
				strokeR = (unsigned char) r;
				strokeG = (unsigned char) g;
				strokeB = (unsigned char) b;
			}
		}

		template<typename t>
		inline void stroke(const t &b)
		{
			stroke(b, b, b);
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

		inline void begin(const shapeTypes mode)
		{
			rapidValidate(!shapeStarted, "Cannot start a new shape because there is an unfinished one already created");

			shapeStarted = true;
			shapeEnded = false;

			glBegin(mode);
		}

		inline void end()
		{
			rapidValidate(!shapeEnded, "Cannot end a shape because nothing was started");

			shapeStarted = false;
			shapeEnded = true;

			glEnd();
		}

		inline void strokeWeight(double s)
		{
			strokeWeightVal = s;
		}

		inline void triangle(double x1, double y1, double x2, double y2, double x3, double y3) const
		{
			// Set the current color to the fill color
			glColor3ub(fillR, fillG, fillB);

			double screenSpaceX1, screenSpaceX2, screenSpaceX3, screenSpaceY1, screenSpaceY2, screenSpaceY3;

			screenSpaceX1 = map(x1, 0, width, -1, 1);
			screenSpaceX2 = map(x2, 0, width, -1, 1);
			screenSpaceX3 = map(x3, 0, width, -1, 1);
			screenSpaceY1 = map(y1, 0, height, 1, -1);
			screenSpaceY2 = map(y2, 0, height, 1, -1);
			screenSpaceY3 = map(y3, 0, height, 1, -1);

			if (!shapeStarted)
				glBegin(GL_TRIANGLES);
			else
				RapidError("Graphics Error", "'begin' has already been called. Call 'end' before drawing a triangle");

			glVertex2d(screenSpaceX1, screenSpaceY1);
			glVertex2d(screenSpaceX2, screenSpaceY2);
			glVertex2d(screenSpaceX3, screenSpaceY3);

			glEnd();

			glLineWidth(strokeWeightVal);
			glColor3ub(strokeR, strokeG, strokeB);

			// P1 to P2
			line(screenSpaceX1, screenSpaceY1, screenSpaceX2, screenSpaceY2);

			// P2 to P3
			line(screenSpaceX2, screenSpaceY2, screenSpaceX3, screenSpaceY3);

			// P1 to P3
			line(screenSpaceX1, screenSpaceY1, screenSpaceX3, screenSpaceY3);
		}

		inline void rect(double x, double y, double w, double h)
		{
			// Set the fill color
			glColor3ub(fillR, fillG, fillB);

			// Draw two triangles for the rectangle
			triangle(x, y + h, x, y, x + w, y);
			triangle(x, y + h, x + w, y + h, x + w, y);

			// Draw the outline

			glColor3ub(strokeR, strokeG, strokeB);

			glLineWidth(strokeWeightVal);

			// Top
			line(x, y, x + w, y);

			// Bottom
			line(x, y + h, x + w, y + h);

			// Left
			line(x, y, x, y + h);

			// Right
			line(x + w, y, x + w, y + h);
		}

		inline void line(double x1, double y1, double x2, double y2) const
		{
			glColor3ub(strokeR, strokeG, strokeB);

			if (strokeWeightVal != 0)
			{
				glLineWidth(strokeWeightVal);

				double screenSpaceX1 = map(x1, 0, width, -1, 1);
				double screenSpaceY1 = map(y1, 0, height, 1, -1);

				double screenSpaceX2 = map(x2, 0, width, -1, 1);
				double screenSpaceY2 = map(y2, 0, height, 1, -1);

				glBegin(GL_LINES);

				glVertex2d(screenSpaceX1, screenSpaceY1);
				glVertex2d(screenSpaceX2, screenSpaceY2);

				glEnd();
			}
		}

		inline void point(double x, double y) const
		{
			double screenSpaceX, screenSpaceY;

			glPointSize(strokeWeightVal);
			glColor3ub(strokeR, strokeG, strokeB);

			screenSpaceX = map(x, 0, width, -1, 1);
			screenSpaceY = map(y, 0, height, 1, -1);

			if (!shapeStarted)
				glBegin(GL_POINTS);

			glVertex2d(screenSpaceX, screenSpaceY);

			if (shapeEnded)
				glEnd();
		}

		template<typename t>
		inline void vertex(t x, t y) const
		{
			double screenSpaceX, screenSpaceY;

			screenSpaceX = map((double) x, 0, width, -1, 1);
			screenSpaceY = map((double) y, 0, height, 1, -1);

			glColor3ub(fillR, fillG, fillB);

			glVertex2d(screenSpaceX, screenSpaceY);
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

	void keyPressCallback(GLFWwindow *window_, int key, int scancode, int action, int mods)
	{
		for (const auto &window : graphicsInstances)
		{
			if (window->callbackWhenNotFocused || window_ == window->internal_window_())
			{
				auto mods_ = std::vector<keyModifier>();
				if (mods & MODIFIER_ALT)
					mods_.emplace_back(MODIFIER_ALT);
				if (mods & MODIFIER_CAPS)
					mods_.emplace_back(MODIFIER_CAPS);
				if (mods & MODIFIER_CONTROL)
					mods_.emplace_back(MODIFIER_CONTROL);
				if (mods & MODIFIER_NUM_LOCK)
					mods_.emplace_back(MODIFIER_NUM_LOCK);
				if (mods & MODIFIER_SHIFT)
					mods_.emplace_back(MODIFIER_SHIFT);

				window->internal_keyDef = {
						(keyType) key,
						(keyAction) action,
						mods_
				};
				window->keyDefSet = true;
			}
		}
	}
}
