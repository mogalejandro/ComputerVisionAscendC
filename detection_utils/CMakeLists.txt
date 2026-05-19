# ==============================================================================
# CMakeLists.txt  —  detection_utils/
# ==============================================================================
# Compiler : GCC 11.4.0  (Ubuntu 11.4.0-1ubuntu1~22.04.2)
# Platform : Ubuntu 22.04 LTS
#
# Builds the `detection_utils` STATIC library — the shared inference pipeline
# consumed exclusively by the `infer` binary.
#
# What this library provides:
#   • ModelConfig / LoadModelConfig   — JSON config loader (paths, resize dims, threshold)
#   • StreamWorker                    — per-channel DVPP decoder + FFmpeg reader thread
#   • OnFrameDecodedMux               — MxBase callback: routes decoded frames round-robin
#   • RunPipeline                     — batch-inference loop (resize → infer → post-process)
#   • RunPostProcessYolov3/Yolov11    — per-model post-process dispatch + bbox drawing
#   • CsvLog                          — thread-safe CSV result writer
#   • MkdirP / Timestamp              — misc utilities
#
# Why STATIC (not SHARED):
#   detection_utils is internal glue consumed only by `infer`.  A static library
#   is linked directly into the binary at build time, so there is no loose .so
#   that would need to be on LD_LIBRARY_PATH at runtime.
#   The yolov3/yolov11 post-processors must remain SHARED because MxBase
#   may dlopen them as runtime plugins.
# ==============================================================================

cmake_minimum_required(VERSION 3.22)   # Ubuntu 22.04 default
project(detection_utils LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
# GCC 11 has full C++17 support.  C++17 is required for:
#   • if-init statements used in detection_utils.cpp (if (auto v = ...; !v.empty()))
#   • inline variables in the cfg:: namespace (detection_utils.h)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)    # -std=c++17 not -std=gnu++17

# ── ABI and namespace definitions ─────────────────────────────────────────────
# Must be identical to the root CMakeLists.  Sub-directory add_definitions do
# not inherit automatically across all CMake versions, so we repeat them here.
#
# _GLIBCXX_USE_CXX11_ABI=0
#   Ubuntu 22.04 / GCC 11 defaults to new ABI (=1).  MxBase was built with
#   old ABI (=0).  Mismatch causes linker failures on any std::string symbol
#   crossing the MxBase API boundary (model paths, label paths, etc.).
#
# Dgoogle=mindxsdk_private
#   Renames the protobuf google:: namespace vendored inside the MxBase SDK to
#   avoid conflicts with any system-installed protobuf (common on Ubuntu 22.04
#   which ships libprotobuf-dev 3.12).
#
# DENABLE_DVPP_INTERFACE
#   Gates the MxBase DVPP hardware decode/encode declarations.  Required to
#   compile VideoDecoder, ImageProcessor and their associated types.
add_definitions(
    -D_GLIBCXX_USE_CXX11_ABI=0
    -Dgoogle=mindxsdk_private
    -DENABLE_DVPP_INTERFACE
)

# ── SDK roots ─────────────────────────────────────────────────────────────────
# Re-read from environment (exported by set_env.sh before cmake runs).
set(MX_SDK_HOME      $ENV{MX_SDK_HOME})
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})

# ── Library name and output path ──────────────────────────────────────────────
set(TARGET_LIBRARY detection_utils)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)  # .a goes to build/lib/

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}                           # detection_utils.h  (self)
    ${CMAKE_SOURCE_DIR}                                   # resolves "yolov3/Yolov3PostProcessNew.h"
                                                          #      and "yolov11/Yolov11PostProcess.h"
    ${MX_SDK_HOME}/include                                # MxBase.h, TensorBase.h, VideoDecoder.h
    ${MX_SDK_HOME}/opensource/include                     # third-party headers in SDK bundle
    ${MX_SDK_HOME}/opensource/include/opencv4             # cv::Mat, cv::imread, cv::rectangle
    ${MX_SDK_HOME}/opensource/include/gstreamer-1.0       # GStreamer — required by plugintoolkit
    ${MX_SDK_HOME}/opensource/include/glib-2.0            # GLib base types (gboolean, gint…)
    ${MX_SDK_HOME}/opensource/lib/glib-2.0/include        # glibconfig.h (generated, per-arch)
    ${ASCEND_HOME_PATH}/include                           # CANN / ACL runtime headers
    /usr/include/aarch64-linux-gnu                        # AArch64 system headers
)

# ── Link-directory search paths ───────────────────────────────────────────────
link_directories(
    ${MX_SDK_HOME}/lib             # mxbase, plugintoolkit, mxpidatatype
    ${MX_SDK_HOME}/opensource/lib  # glib, gstreamer, opencv_world, avformat…
    ${ASCEND_HOME_PATH}/lib64      # CANN / ACL runtime
    /usr/lib/aarch64-linux-gnu     # system libs
)

# ── Static library target ─────────────────────────────────────────────────────
add_library(${TARGET_LIBRARY} STATIC
    detection_utils.cpp   # full pipeline implementation
    detection_utils.h     # public API (included by src/main.cpp)
)

# ── Link dependencies ─────────────────────────────────────────────────────────
target_link_libraries(${TARGET_LIBRARY}
    # GStreamer stack ─────────────────────────────────────────────────────────
    # Not called directly, but plugintoolkit has a hard link-time dependency on
    # these shared libraries.  GCC 11 / ld 2.38 enforce --as-needed by default
    # on Ubuntu 22.04, so all transitive deps must be listed explicitly.
    glib-2.0        # fundamental type system and main loop
    gstreamer-1.0   # plugin pipeline framework
    gobject-2.0     # GObject type system
    gstbase-1.0     # GStreamer base classes
    gmodule-2.0     # dynamic module loading (GLib)

    plugintoolkit   # MxBase plugin registration / lifecycle management
    mxpidatatype    # shared data types: ObjectInfo, TensorBase, ResizedImageInfo
    mxbase          # MxBase core: Model, ImageProcessor, VideoDecoder, BatchConcat

    opencv_world    # OpenCV all-in-one: frame encode/decode, drawing
    avformat        # FFmpeg: container demux  (mp4, RTSP, mkv)
    avcodec         # FFmpeg: bitstream filter (h264_mp4toannexb, hevc_mp4toannexb)
    avutil          # FFmpeg: utility types (AVRational, av_dict_set, etc.)

    pthread         # POSIX threads — backend for std::thread / std::mutex / std::cv
)