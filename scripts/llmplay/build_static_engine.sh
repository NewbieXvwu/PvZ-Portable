#!/usr/bin/env bash
# GPT package: static x86-64 Linux, headless only, no runtime system libraries.
set -euo pipefail
SRC=$(realpath "${1:?engine source dir}")
OUT=$(realpath -m "${2:?build dir}")
W=$HOME/pvz-logs/llmplay-build
SDL=$W/SDL2-2.32.10
PREFIX=$W/sdl-headless-install
mkdir -p "$W"
if [ ! -d "$SDL" ]; then
  curl --fail -L -o "$W/SDL2-2.32.10.tar.gz" https://github.com/libsdl-org/SDL/releases/download/release-2.32.10/SDL2-2.32.10.tar.gz
  tar xf "$W/SDL2-2.32.10.tar.gz" -C "$W"
fi
# SDL_Init(TIMER) also initializes DBus in a desktop-enabled SDL build.
# Disable loadso as well, so future SDL callers cannot dlopen a system library.
# SDL's separate dynamic API also needs dlopen marked unavailable.
cmake -S "$SDL" -B "$W/sdl-headless-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DSDL_SHARED=OFF -DSDL_STATIC=ON -DSDL_TEST=OFF \
  -DSDL_LOADSO=OFF -DSDL_DBUS=OFF -DSDL_IBUS=OFF -DSDL_LIBUDEV=OFF \
  -DHAVE_DLOPEN=OFF -DHAVE_DLOPEN_IN_LIBC=OFF -DHAVE_DLOPEN_IN_LIBDL=OFF \
  -DSDL_JOYSTICK=OFF -DSDL_HAPTIC=OFF -DSDL_SENSOR=OFF -DSDL_HIDAPI=OFF \
  -DSDL_X11=OFF -DSDL_WAYLAND=OFF -DSDL_KMSDRM=OFF -DSDL_DIRECTFB=OFF \
  -DSDL_RPI=OFF -DSDL_VIVANTE=OFF -DSDL_OFFSCREEN=OFF -DSDL_DUMMYVIDEO=ON \
  -DSDL_OPENGL=OFF -DSDL_OPENGLES=OFF -DSDL_VULKAN=OFF \
  -DSDL_AUDIO=OFF -DSDL_ALSA=OFF -DSDL_OSS=OFF -DSDL_JACK=OFF -DSDL_ESD=OFF \
  -DSDL_PIPEWIRE=OFF -DSDL_PULSEAUDIO=OFF -DSDL_ARTS=OFF -DSDL_NAS=OFF \
  -DSDL_SNDIO=OFF -DSDL_FUSIONSOUND=OFF -DSDL_LIBSAMPLERATE=OFF \
  -DSDL_DISKAUDIO=OFF -DSDL_DUMMYAUDIO=OFF
nice -n 19 cmake --build "$W/sdl-headless-build" -j 4
cmake --install "$W/sdl-headless-build"
cmake -S "$SRC" -B "$OUT" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_STATIC=ON -DCMAKE_EXE_LINKER_FLAGS=-static -DPVZ_ROLLOUT_BLAS=OFF \
  -DSDL2_DIR="$PREFIX/lib/cmake/SDL2" -DCMAKE_PREFIX_PATH="$PREFIX"
nice -n 19 cmake --build "$OUT" -j 4 --target pvz-portable
strip -o "$OUT/pvz-portable.static" "$OUT/pvz-portable"
file "$OUT/pvz-portable.static"
