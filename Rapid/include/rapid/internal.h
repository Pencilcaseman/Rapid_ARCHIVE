#pragma once

#include <cstdlib>

#include <iostream>
#include <fstream>
#include <sstream>

#include <array>
#include <string>
#include <vector>
#include <functional>

#include <type_traits>

#include <random>
#include <chrono>
#include <ctime>

#include <algorithm>
#include <iterator>
#include <cassert>
#include <memory>
#include <stdexcept>

#include <conio.h>
#include <sys/stat.h>

#include <intrin.h>

#include <omp.h>

#ifndef RAPID_NO_AMP
#include <amp.h>		// Optional AMP include
#endif

#ifndef RAPID_NO_BLAS
#pragma comment(lib, "libopenblas.lib")
#include <cblas.h>		// Optional OpenBLAS include
#endif

#if !(defined(RAPID_NO_AMP) || defined(RAPID_NO_BLAS))
#define RAPID_SET_THREADS(x) omp_set_num_threads((x));
#else
#define RAPID_SET_THREADS(x) (x)
#endif

#undef min
#undef max

// Rapid definitions
#if defined(NDEBUG)
#define RAPID_RELEASE
#else
#ifndef RAPID_DEBUG
#define RAPID_DEBUG
#endif
#endif

#ifdef RAPID_DEBUG
#ifndef RAPID_NO_CHECK_NAN
#define RAPID_CHECK_NAN
#endif
#endif

#if defined(_M_IX86)
#define RAPID_X86
#elif defined(_M_X64)
#define RAPID_X64
#else
#define RAPID_BUILD_UNKNOWN
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#if defined(_WIN32)
#define RAPID_OS_WINDOWS // Windows
#define RAPID_OS "windows"
#elif defined(_WIN64)
#define RAPID_OS_WINDOWS // Windows
#define RAPID_OS "windows"
#elif defined(__CYGWIN__) && !defined(_WIN32)
#define RAPID_OS_WINDOWS  // Windows (Cygwin POSIX under Microsoft Window)
#define RAPID_OS "windows"
#elif defined(__ANDROID__)
#define RAPID_OS_ANDROID // Android (implies Linux, so it must come first)
#define RAPID_OS "android"
#elif defined(__linux__)
#define RAPID_OS_LINUX // Debian, Ubuntu, Gentoo, Fedora, openSUSE, RedHat, Centos and other
#define RAPID_OS "linux"
#elif defined(__unix__) || !defined(__APPLE__) && defined(__MACH__)
#include <sys/param.h>
#if defined(BSD)
#define RAPID_OS_BSD // FreeBSD, NetBSD, OpenBSD, DragonFly BSD
#define RAPID_OS "bsd"
#endif
#elif defined(__hpux)
#define RAPID_OS_HP_UX// HP-UX
#define RAPID_OS "hp-ux" 
#elif defined(_AIX)
#define RAPID_OS_AIX // IBM AIX
#define RAPID_OS "aix"
#elif defined(__APPLE__) && defined(__MACH__) // Apple OSX and iOS (Darwin)
#include <TargetConditionals.h>
#if TARGET_IPHONE_SIMULATOR == 1
#define RAPID_OS_IOS // Apple iOS
#define RAPID_OS "ios"
#elif TARGET_OS_IPHONE == 1
#define RAPID_OS_IOS // Apple iOS
#define RAPID_OS "ios"
#elif TARGET_OS_MAC == 1
#define RAPID_OS_OSX // Apple OSX
#define RAPID_OS "osx"
#endif
#elif defined(__sun) && defined(__SVR4)
#define RAPID_OS_SOLARIS // Oracle Solaris, Open Indiana
#define RAPID_OS "solaris"
#else
#define RAPID_OS_UNKNOWN 
#define RAPID_OS "unknown"
#endif

#ifdef RAPID_OS_WINDOWS
#include <direct.h>

std::string workingDirectory()
{
	char buff[255];
	_getcwd(buff, 255);
	std::string current_working_dir(buff);

	return current_working_dir;
}

#define RAPID_WORKING_DIR workingDirectory()
#endif

#ifndef RAPID_NO_GRAPHICS
// Link the required libraries for graphics
#pragma comment(lib, "glew32s.lib")
#pragma comment(lib, "glfw3.lib")
#pragma comment(lib, "opengl32.lib")
#endif

namespace rapid
{
#ifdef RAPID_DEBUG
#define rapidAssert(cond, err) { if (!(cond)) {rapid::RapidError("Assertion Failed", err).display(); }}

#else
#define rapidAssert(cond, err) cond
#endif

	inline void rapidValidate(bool condition, const std::string &err = "Error", const int code = 1)
	{
		if (!condition)
		{
			std::cerr << err << "\n";
			exit(code);
		}
	}
}

//**************//
// CURRENT TIME //
//**************//
// A routine to give access to a high precision timer on most systems.
#if defined(_WIN32) || defined(_WIN64) || defined(MSC_VER) || defined(WIN32) || defined(__CYGWIN__) || defined(CUDAAPI)
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif

#include <Windows.h>

double seconds()
{
	LARGE_INTEGER t;
	static double oofreq;
	static int checkedForHighResTimer = 0;
	static BOOL hasHighResTimer;

	if (!checkedForHighResTimer)
	{
		hasHighResTimer = QueryPerformanceFrequency(&t);
		oofreq = 1.0 / (double) t.QuadPart;
		checkedForHighResTimer = 1;
	}

	if (hasHighResTimer)
	{
		QueryPerformanceCounter(&t);
		return (double) t.QuadPart * oofreq;
	}
	else
	{
		return (double) GetTickCount64() * 1.0e-3;
	}
}

#define TIME (seconds())

#else
#define TIME (omp_get_wtime())
#endif

namespace rapid
{
	class RapidTimer
	{
	public:
		std::chrono::time_point<std::chrono::steady_clock> start, end;

		RapidTimer()
		{
			start = std::chrono::high_resolution_clock::now();
		}

		~RapidTimer()
		{
			end = std::chrono::high_resolution_clock::now();

			auto delta = (double) std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
			std::string unit = "ns";

			if (delta >= 1000)
			{
				unit = "us";
				delta /= 1000.;
			}

			if (delta >= 1000)
			{
				unit = "ms";
				delta /= 1000.;
			}

			if (delta >= 1000.)
			{
				unit = "s";
				delta /= 1000.;
			}

			std::cout << std::fixed;
			std::cout << "Block finished in: " << delta << " " << unit << "\n";
		}
	};

	template<typename T, typename U>
	inline T rapidCast(const U &in, const std::string &name = "C")
	{
		std::stringstream istr;
		istr.imbue(std::locale(name));

		if (std::is_floating_point<U>::value)
			istr.precision(std::numeric_limits<U>::digits10);

		istr << in;

		std::string str = istr.str(); // save string in case of exception

		T val;
		istr >> val;

		if (istr.fail())
			throw std::invalid_argument(str);

		return val;
	}

	template<>
	inline std::string rapidCast(const std::string &in, const std::string &name)
	{
		return in;
	}

	template<>
	inline unsigned char rapidCast(const std::string &in, const std::string &name)
	{
		return (unsigned char) atoi(in.c_str());
	}

	template<>
	inline char rapidCast(const std::string &in, const std::string &name)
	{
		return (char) atoi(in.c_str());
	}

	template<>
	inline int rapidCast(const std::string &in, const std::string &name)
	{
		return (int) atoi(in.c_str());
	}

	template<>
	inline float rapidCast(const std::string &in, const std::string &name)
	{
		return (float) atof(in.c_str());
	}

	template<>
	inline double rapidCast(const std::string &in, const std::string &name)
	{
		return (double) atof(in.c_str());
	}
}
