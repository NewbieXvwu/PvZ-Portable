cmake_minimum_required(VERSION 3.12...4.0)

project(pvz-portable LANGUAGES C CXX)

# One game on every machine: arm64 clang fuses a*b+c into a single FMA by default and x86-64 builds have no FMA,
# so Mac archives drifted from the desktop's (2026-10-19: a pole vaulter at x 599.843 vs 599.882, frame 413).
# Off on Mac, a no-op for the desktop's GCC.
if(NOT MSVC)
	add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-ffp-contract=off>)
endif()

# The engine process is the whole cost of collection: every decision spends its time stepping
# ticks in here.  Link-time optimisation is what lets its small inline-heavy update functions
# fold into their callers.  -ffp-contract=off above keeps the game bit-identical: LTO never
# reassociates float arithmetic, and the rollout leaf digests in scripts/bench_engine.py are
# the gate (2026-10-23: 64/64 roots identical).
#
# -march=native was measured and dropped: with LTO it added nothing (87.2k -> 87.2k ticks/CPU-s,
# scripts/bench_engine.py, 3 runs each) and it would make the binary machine-specific, which the
# identical-game guarantee across Mac and desktop does not want.
if(NOT MSVC AND NOT EMSCRIPTEN)
	# Line tables for the sampler in scripts/prof_sample.c: without them a profile can only say
	# "inside Reanimation::Update", not which line.  -g1 does not change code generation.
	add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-g1>)
	include(CheckIPOSupported)
	check_ipo_supported(RESULT PVZ_IPO_SUPPORTED OUTPUT PVZ_IPO_ERROR LANGUAGES C CXX)
	if(PVZ_IPO_SUPPORTED)
		set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
	else()
		message(STATUS "LTO unavailable: ${PVZ_IPO_ERROR}")
	endif()
	# Profile-guided optimisation.  scripts/build_pgo.sh writes the profile for this same
	# build directory (GCC looks the .gcda up by object path, so it cannot be moved); once it
	# is there every build uses it.  A machine without a profile still builds normally.
	set(PVZ_PGO_DATA "${PROJECT_SOURCE_DIR}/artifacts/perf/pgo" CACHE PATH "PGO profile directory")
	if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
		if(EXISTS "${PVZ_PGO_DATA}/engine.profdata")
			message(STATUS "PGO: using profile in ${PVZ_PGO_DATA}/engine.profdata")
			add_compile_options(-fprofile-instr-use=${PVZ_PGO_DATA}/engine.profdata)
			add_link_options(-fprofile-instr-use=${PVZ_PGO_DATA}/engine.profdata)
		endif()
	else()
	file(GLOB PVZ_PGO_FILES "${PVZ_PGO_DATA}/*.gcda")
	if(PVZ_PGO_FILES)
		message(STATUS "PGO: using profile in ${PVZ_PGO_DATA}")
		# -Wno-error=coverage-mismatch: after a source change the old .gcda no longer matches;
		# that must degrade to "no profile for this function", not fail the build.  The profile is
		# regenerated with scripts/build_pgo.sh at the end of every optimisation round.
		add_compile_options(-fprofile-use=${PVZ_PGO_DATA} -fprofile-correction -Wno-missing-profile
			-Wno-error=coverage-mismatch)
		add_link_options(-fprofile-use=${PVZ_PGO_DATA})
	endif()
	endif()
endif()
include(CMake/ProjectVersion.cmake)

option(BUILD_STATIC "Statically link dependencies for a standalone exe (Windows MinGW, Linux)" OFF)
if(BUILD_STATIC)
	if(WIN32 AND NOT MSVC)
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
		set(BUILD_SHARED_LIBS OFF)
		add_link_options(-static)
	elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
		set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
		set(BUILD_SHARED_LIBS OFF)
	elseif(MSVC)
		message(WARNING "BUILD_STATIC is not supported for MSVC. Use a vcpkg -static triplet instead.")
		set(BUILD_STATIC OFF)
	else()
		message(WARNING "BUILD_STATIC is only supported on Windows with MinGW-based toolchains and on Linux.")
		set(BUILD_STATIC OFF)
	endif()
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (CMAKE_GENERATOR MATCHES "Ninja")
	set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
endif()

if(MSVC)
	if(DEFINED VCPKG_TARGET_TRIPLET AND VCPKG_TARGET_TRIPLET MATCHES "-static")
		set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
	elseif(DEFINED VCPKG_TARGET_TRIPLET)
		set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
	endif()
endif()

file(GLOB SOURCES CONFIGURE_DEPENDS
	src/*.cpp
	src/SexyAppFramework/*.cpp
	src/SexyAppFramework/graphics/*.cpp
	src/SexyAppFramework/imagelib/*.cpp
	src/SexyAppFramework/misc/*.cpp
	src/SexyAppFramework/paklib/*.cpp
	src/SexyAppFramework/sound/*.cpp
	src/SexyAppFramework/widget/*.cpp
	src/SexyAppFramework/fcaseopen/*.c
	src/PvzpLib/*.cpp
	src/Lawn/*.cpp
	src/Lawn/Widget/*.cpp
	src/Lawn/System/*.cpp
)
# Inference backends have independent compile flags/dependencies. Switching an optional
# backend must not rebuild every game source on the production host.
list(FILTER SOURCES EXCLUDE REGEX "/GenericForward(Sparse|Dnnl|Token)?\\.cpp$")

option(LOW_MEMORY "Enable low memory mode" OFF)
if(NINTENDO_SWITCH)
	list(APPEND SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/switch/Window.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/switch/Input.cpp
	)
	list(APPEND PLAT_INCLUDES
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/switch
	)
elseif(EMSCRIPTEN)
	# libopenmpt has no Emscripten port; must be pre-built with emmake.
	find_library(OPENMPT_LIBRARY NAMES openmpt libopenmpt REQUIRED)
	set(OPENMPT_LIB ${OPENMPT_LIBRARY})

	list(APPEND SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/emscripten/Window.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/default/Input.cpp
	)
	list(APPEND PLAT_INCLUDES
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/emscripten
	)
else()
	if(VCPKG_TOOLCHAIN)
		find_package(libopenmpt CONFIG REQUIRED)
		set(OPENMPT_LIB libopenmpt::libopenmpt)
	else()
		find_library(OPENMPT_LIBRARY NAMES openmpt libopenmpt REQUIRED)
		set(OPENMPT_LIB ${OPENMPT_LIBRARY})
		if(OPENMPT_LIBRARY MATCHES "\\${CMAKE_STATIC_LIBRARY_SUFFIX}$")
			find_path(OPENMPT_INCLUDE_DIR libopenmpt/libopenmpt.h REQUIRED)
			include(CheckCXXSourceCompiles)
			include(CMakePushCheckState)
			cmake_push_check_state()
			set(CMAKE_REQUIRED_INCLUDES ${OPENMPT_INCLUDE_DIR})
			set(CMAKE_REQUIRED_LIBRARIES ${OPENMPT_LIBRARY})
			check_cxx_source_compiles("#include <libopenmpt/libopenmpt.h>
int main() {
	return openmpt_module_create_from_memory2(0, 0, 0, 0, 0, 0, 0, 0, 0) != 0;
}" OPENMPT_LINKS_STANDALONE)
			cmake_pop_check_state()
			if(NOT OPENMPT_LINKS_STANDALONE)
				find_library(MPG123_LIBRARY NAMES mpg123 libmpg123)
				find_library(VORBISFILE_LIBRARY NAMES vorbisfile libvorbisfile)
				find_library(VORBIS_LIBRARY NAMES vorbis libvorbis)
				find_library(OGG_LIBRARY NAMES ogg libogg)
				if(NOT MPG123_LIBRARY OR NOT VORBISFILE_LIBRARY OR NOT VORBIS_LIBRARY OR NOT OGG_LIBRARY)
					message(FATAL_ERROR "Static libopenmpt does not link standalone and some of its dependencies were not found (see the configure log for details)")
				endif()
				list(APPEND OPENMPT_LIB ${MPG123_LIBRARY} ${VORBISFILE_LIBRARY} ${VORBIS_LIBRARY} ${OGG_LIBRARY})
			endif()
		endif()
	endif()
	list(APPEND SOURCES
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/default/Window.cpp
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/default/Input.cpp
	)
	list(APPEND PLAT_INCLUDES
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/platform/default
	)
endif()

if(CMAKE_BUILD_TYPE AND CMAKE_BUILD_TYPE MATCHES "^[Dd][Ee][Bb][Uu][Gg]$")
	set(WIN_CONSOLE_DEFAULT ON)
	set(PVZ_DEBUG_DEFAULT ON)
else()
	set(WIN_CONSOLE_DEFAULT OFF)
	set(PVZ_DEBUG_DEFAULT OFF)
endif()

option(PVZ_DEBUG "Enable PVZ_DEBUG macro to enable cheats, hidden features, etc." ${PVZ_DEBUG_DEFAULT})
option(CONSOLE "Show console on Windows" ${WIN_CONSOLE_DEFAULT})
option(DO_FIX_BUGS "Define DO_FIX_BUGS macro (Community fixes for original game bugs of 1.2.0.1073 GOTY Edition)" OFF)

if(EMSCRIPTEN)
	list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -sUSE_SDL=2 -sUSE_ZLIB=1 -sUSE_LIBPNG=1 -sUSE_LIBJPEG=1")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -sUSE_SDL=2 -sUSE_ZLIB=1 -sUSE_LIBPNG=1 -sUSE_LIBJPEG=1 -fexceptions")
else()
	find_package(Threads REQUIRED)
	find_package(ZLIB REQUIRED)
	find_package(JPEG REQUIRED)
	find_package(PNG REQUIRED)
	find_package(SDL2 REQUIRED)
endif()

function(_pvz_patch_sdl2_iface_libs target)
	get_target_property(_libs ${target} INTERFACE_LINK_LIBRARIES)
	if(NOT _libs)
		return()
	endif()
	set(_patched)
	foreach(_lib IN LISTS _libs)
		if(_lib STREQUAL "SDL2::SDL2")
			list(APPEND _patched SDL2::SDL2-static)
		else()
			list(APPEND _patched "${_lib}")
		endif()
	endforeach()
	set_target_properties(${target} PROPERTIES INTERFACE_LINK_LIBRARIES "${_patched}")
endfunction()

if(BUILD_STATIC AND TARGET SDL2::SDL2-static)
	if(TARGET SDL2::SDL2main)
		_pvz_patch_sdl2_iface_libs(SDL2::SDL2main)
	endif()
	set(SDL2_LINK_TARGET SDL2::SDL2-static)
else()
	set(SDL2_LINK_TARGET SDL2::SDL2)
endif()

set(SDL_MIXER_X_STATIC ON CACHE BOOL "Build static library of SDL Mixer X")
set(SDL_MIXER_X_SHARED OFF CACHE BOOL "Build shared library of SDL Mixer X")
set(USE_MIDI OFF CACHE BOOL "SDL Mixer X midi")
set(USE_MODPLUG OFF CACHE BOOL "SDL Mixer X modplug")
set(USE_XMP OFF CACHE BOOL "SDL Mixer X xmp")
set(USE_OPUS OFF CACHE BOOL "SDL Mixer X opus")
set(USE_DRFLAC OFF CACHE BOOL "SDL Mixer X flac")
set(USE_PXTONE OFF CACHE BOOL "SDL Mixer X pxtone")
add_subdirectory(src/SexyAppFramework/sound/SDL-Mixer-X)

# Patch SDL-Mixer-X's transitive SDL2 dep without touching vendored CMake files
if(BUILD_STATIC AND TARGET SDL2::SDL2-static)
	_pvz_patch_sdl2_iface_libs(SDL2_mixer_ext_Static)
endif()

# Android requires a shared library loaded by SDLActivity; other platforms use an executable.
if(ANDROID)
	add_library(pvz-portable SHARED ${SOURCES})
elseif(IOS)
	add_executable(pvz-portable MACOSX_BUNDLE ${SOURCES})
else()
	add_executable(pvz-portable ${SOURCES})
endif()

target_include_directories(pvz-portable PRIVATE
	${PLAT_INCLUDES}
	${PROJECT_BINARY_DIR}
	${PROJECT_SOURCE_DIR}/src
	${PROJECT_SOURCE_DIR}/src/SexyAppFramework
	${PROJECT_SOURCE_DIR}/src/SexyAppFramework/sound/SDL-Mixer-X/include
	${SDL2_INCLUDE_DIRS}
)
target_compile_definitions(pvz-portable PRIVATE
	$<$<BOOL:${PVZ_DEBUG}>:PVZ_DEBUG>
	$<$<BOOL:${DO_FIX_BUGS}>:DO_FIX_BUGS>
)
target_compile_features(pvz-portable PRIVATE cxx_std_20)
set_target_properties(pvz-portable PROPERTIES CXX_EXTENSIONS OFF)

# Desktop (Linux, OpenBLAS) 2026-10-17: teacher 98.3 -> 72.4 s/game with identical games; without an OpenBLAS the
# hand-written kernels are used (the reference BLAS is not faster than them).
option(PVZ_ROLLOUT_BLAS "Use native/system BLAS for student convolutions when available" ON)
if(PVZ_ROLLOUT_BLAS)
	if(APPLE)
		find_library(ROLLOUT_BLAS Accelerate REQUIRED)
	else()
		if(NOT DEFINED BLA_VENDOR)
			set(BLA_VENDOR OpenBLAS)
		endif()
		find_package(BLAS QUIET)
		set(ROLLOUT_BLAS ${BLAS_LIBRARIES})
	endif()
	if(ROLLOUT_BLAS)
		target_link_libraries(pvz-portable PRIVATE ${ROLLOUT_BLAS})
		target_compile_definitions(pvz-portable PRIVATE PVZ_ROLLOUT_BLAS)
	endif()
endif()

# Value inference always uses im2col + CBLAS, independently of student-kernel selection.
if(NOT ROLLOUT_BLAS)
	if(APPLE)
		find_library(VALUE_BLAS Accelerate REQUIRED)
	else()
		find_package(BLAS REQUIRED)
		set(VALUE_BLAS ${BLAS_LIBRARIES})
	endif()
	target_link_libraries(pvz-portable PRIVATE ${VALUE_BLAS})
endif()

option(PVZ_GENERIC_DNNL "Use optional oneDNN float32 convolutions in generic graphs" OFF)
add_library(pvz-forward STATIC src/GenericForward.cpp src/GenericForwardDnnl.cpp src/GenericForwardSparse.cpp src/GenericForwardToken.cpp)
target_include_directories(pvz-forward PUBLIC src)
target_compile_features(pvz-forward PUBLIC cxx_std_20)
target_link_libraries(pvz-forward PUBLIC ${ROLLOUT_BLAS} ${VALUE_BLAS})
target_link_libraries(pvz-portable PRIVATE pvz-forward)
if(PVZ_GENERIC_DNNL)
	find_package(dnnl CONFIG REQUIRED)
	find_package(OpenMP REQUIRED)
	target_compile_definitions(pvz-forward PRIVATE PVZ_GENERIC_DNNL)
	target_link_libraries(pvz-forward PRIVATE DNNL::dnnl OpenMP::OpenMP_CXX)
endif()

if (LOW_MEMORY)
	target_compile_definitions(pvz-portable PRIVATE LOW_MEMORY)
endif()

# Allocator.  The engine allocates and frees board objects, reanimation tracks and particles
# constantly; the system malloc was ~7% of the rollout CPU and mimalloc removes most of it
# (2026-10-23: 103.2k -> 111.1k ticks/CPU-s, 3 runs each, scripts/bench_engine.py).  It
# overrides malloc process-wide, so it is linked statically and needs no runtime switch.
# The first configure downloads it; later ones reuse build/_deps.
if(NOT MSVC AND NOT EMSCRIPTEN AND NOT APPLE)
	include(FetchContent)
	set(MI_OVERRIDE ON CACHE BOOL "" FORCE)
	set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
	set(MI_BUILD_OBJECT OFF CACHE BOOL "" FORCE)
	FetchContent_Declare(mimalloc
		GIT_REPOSITORY https://github.com/microsoft/mimalloc
		GIT_TAG v2.1.7
		GIT_SHALLOW TRUE)
	FetchContent_MakeAvailable(mimalloc)
	target_link_libraries(pvz-portable PRIVATE mimalloc-static)
	if(APPLE)
		target_link_options(pvz-portable PRIVATE "-Wl,-force_load,$<TARGET_FILE:mimalloc-static>")
	endif()
endif()

if(EMSCRIPTEN)
	set_property(TARGET pvz-portable APPEND PROPERTY LINK_DEPENDS
		"${CMAKE_CURRENT_SOURCE_DIR}/wasm/shell.html"
	)
	target_link_libraries(pvz-portable PRIVATE
		SDL2_mixer_ext_Static
		${OPENMPT_LIB}
	)
	target_link_options(pvz-portable PRIVATE
		"SHELL:-sUSE_SDL=2"
		"SHELL:-sUSE_ZLIB=1"
		"SHELL:-sUSE_LIBPNG=1"
		"SHELL:-sUSE_LIBJPEG=1"
		"SHELL:-fexceptions"
		"SHELL:-sASYNCIFY"
		"SHELL:-sALLOW_MEMORY_GROWTH=1"
		"SHELL:-sMAXIMUM_MEMORY=1073741824"
		"SHELL:-sGROWABLE_ARRAYBUFFERS=0"
		"SHELL:-sFORCE_FILESYSTEM=1"
		"SHELL:-sEXPORTED_RUNTIME_METHODS=[\"FS\",\"callMain\"]"
		"SHELL:-sINVOKE_RUN=0"
		"SHELL:-lidbfs.js"
		"SHELL:--shell-file ${CMAKE_CURRENT_SOURCE_DIR}/wasm/shell.html"
	)
	set_target_properties(pvz-portable PROPERTIES SUFFIX ".html")
else()
	target_link_libraries(pvz-portable PRIVATE
		SDL2_mixer_ext_Static 
		${OPENMPT_LIB} 
		PNG::PNG
		JPEG::JPEG
		ZLIB::ZLIB
		SDL2::SDL2main
		${SDL2_LINK_TARGET}
		Threads::Threads
	)
endif()

if (WIN32)
	if(MSVC)
		target_compile_options(pvz-portable PRIVATE /utf-8)
	endif()

	target_sources(pvz-portable PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/LawnProject.rc)

	set_source_files_properties(
		${CMAKE_CURRENT_SOURCE_DIR}/src/SexyAppFramework/LawnProject.rc
		PROPERTIES
			COMPILE_DEFINITIONS "APP_ICON_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/icon.ico\""
			OBJECT_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/icon.ico"
	)

	target_link_libraries(pvz-portable PRIVATE ws2_32 user32 gdi32 winmm imm32 shlwapi)
	target_compile_definitions(pvz-portable PRIVATE WINDOWS)

	if(NOT CONSOLE)
		if(MSVC)
			set_target_properties(pvz-portable PROPERTIES WIN32_EXECUTABLE TRUE)
		else()
			target_link_options(pvz-portable PRIVATE -mwindows)
		endif()
	endif()
elseif (NINTENDO_SWITCH)
	target_link_libraries(pvz-portable PRIVATE GLESv2 EGL glapi drm_nouveau)

	nx_generate_nacp(pvz-portable.nacp
		NAME "PvZ Portable"
		AUTHOR "wszqkzqk, etc., a community-driven port"
		VERSION "${PVZP_VERSION_PLAIN}"
	)

	nx_create_nro(pvz-portable
		NACP pvz-portable.nacp
		ICON ${CMAKE_CURRENT_SOURCE_DIR}/icon-switch.jpg
	)
elseif (ANDROID)
	target_link_libraries(pvz-portable PRIVATE GLESv2 EGL android log)
	set_target_properties(pvz-portable PROPERTIES OUTPUT_NAME "main")
elseif (IOS)
	target_link_libraries(pvz-portable PRIVATE
		"-framework Foundation"
		"-framework UIKit"
		"-framework OpenGLES"
		"-framework QuartzCore"
		"-framework CoreGraphics"
		"-framework CoreMotion"
		"-framework GameController"
		"-framework AVFoundation"
		"-framework AudioToolbox"
		"-framework Metal"
	)

	configure_file(ios/Info.plist.in "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
	set_target_properties(pvz-portable PROPERTIES
		MACOSX_BUNDLE TRUE
		MACOSX_BUNDLE_GUI_IDENTIFIER "io.github.wszqkzqk.pvz-portable"
		MACOSX_BUNDLE_BUNDLE_NAME "PvZ Portable"
		MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
		XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
		XCODE_ATTRIBUTE_IPHONEOS_DEPLOYMENT_TARGET "16.4"
		XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "io.github.wszqkzqk.pvz-portable"
		XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY "-"
		XCODE_ATTRIBUTE_ASSETCATALOG_COMPILER_APPICON_NAME "AppIcon"
	)

	# Add launch storyboard to the bundle
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ios/LaunchScreen.storyboard")
		target_sources(pvz-portable PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ios/LaunchScreen.storyboard")
		set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/ios/LaunchScreen.storyboard"
			PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
	endif()

	# Add asset catalog (app icon)
	if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ios/Assets.xcassets")
		target_sources(pvz-portable PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/ios/Assets.xcassets")
		set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/ios/Assets.xcassets"
			PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
	endif()
endif()
