# ==============================================================================
# CMakeLists.txt  —  ROOT  (project root)
# ==============================================================================
# Compiler : GCC 11.4.0  (Ubuntu 11.4.0-1ubuntu1~22.04.2)
# Platform : Ubuntu 22.04 LTS  (x86_64 or aarch64)
# CMake    : 3.22+  (Ubuntu 22.04 ships cmake 3.22.1)
#
# This is the top-level build file.  It sets global compiler flags, resolves
# SDK environment variables, registers every sub-library, and defines the final
# `infer` binary that links everything together.
#
# Build graph:
#   infer  (binary, src/main.cpp)
#     ├── detection_utils       [STATIC]   pipeline glue, FFmpeg, OpenCV, MxBase
#     ├── yolov3postprocessnew  [SHARED]   YOLOv3 post-processor plugin
#     └── yolov11postprocess    [SHARED]   YOLOv11 post-processor plugin
#
# Prerequisite: run build.sh, which sources set_env.sh to export
#   MX_SDK_HOME       →  MxVision SDK root  (e.g. .../mxVision-7.3.0)
#   ASCEND_HOME_PATH  →  CANN toolkit root  (e.g. .../ascend-toolkit/latest)
# before invoking cmake.
# ==============================================================================

cmake_minimum_required(VERSION 3.22)   # Ubuntu 22.04 default is cmake 3.22.1
project(AscendInfer LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
# C++17 is fully supported by GCC 11.  We use it for:
#   • inline namespace variables  (cfg:: knobs in detection_utils.h)
#   • if-init statements          (if (auto v = FlagVal(...); !v.empty()))
#   • structured bindings         (auto& [k, v] : map)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)          # use -std=c++17, not -std=gnu++17

# ── Build type default ────────────────────────────────────────────────────────
# Release enables -O2 and strips assert().  Override with:
#   cmake -DCMAKE_BUILD_TYPE=Debug ..
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Compiler flags (applied to every target in this project) ──────────────────
# -fPIE / -pie          position-independent executable (hardening)
# -fPIC                 position-independent code (required by linked SHARED libs)
# -fstack-protector-all insert stack canaries in every function (hardening)
# -Wl,-z,relro,...      linker hardening: read-only relocations after startup,
#                       immediate PLT binding, non-executable stack segment
# -s                    strip debug symbols from the final binary
# -Wall                 standard GCC warning set
#
# NOTE: -Wl,... flags passed to add_compile_options are forwarded to the
# compiler driver which passes them to ld.  This is the same approach the
# original build used and is compatible with GCC 11 / ld 2.38 (Ubuntu 22.04).
add_compile_options(
    -fPIE
    -fPIC
    -fstack-protector-all
    -Wl,-z,relro,-z,now,-z,noexecstack
    -s
    -pie
    -Wall
)

# ── Preprocessor definitions (applied project-wide) ───────────────────────────
# _GLIBCXX_USE_CXX11_ABI=0
#   GCC 5+ introduced a new std::string / std::list ABI.  Ubuntu 22.04 / GCC 11
#   defaults to ABI=1, but MxBase and the MindX SDK were compiled with ABI=0
#   (old ABI).  Without this flag linking fails with undefined symbol errors
#   for anything that passes std::string across the MxBase boundary.
#
# Dgoogle=mindxsdk_private
#   MxBase vendors a copy of protobuf compiled into the SDK.  Its google::
#   namespace would clash with any system protobuf headers.  This macro renames
#   the internal namespace to mindxsdk_private:: at the preprocessor level.
#
# DENABLE_DVPP_INTERFACE
#   Unlocks the MxBase DVPP hardware video decode/encode API surface.
#   Without this flag VideoDecoder and ImageProcessor are not declared.
add_definitions(
    -D_GLIBCXX_USE_CXX11_ABI=0
    -Dgoogle=mindxsdk_private
    -DENABLE_DVPP_INTERFACE
)

# ── SDK root paths ─────────────────────────────────────────────────────────────
# Both variables are exported by set_env.sh before cmake is invoked (build.sh).
# Capturing them here makes them available as CMake variables in every
# sub-directory without re-reading $ENV{} in each sub-CMakeLists.
set(MX_SDK_HOME      $ENV{MX_SDK_HOME})       # MxVision SDK (headers, libs, plugins)
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})  # CANN toolkit (ACL headers, lib64)

# ── Output location ───────────────────────────────────────────────────────────
# Place the compiled `infer` binary directly in the project root so it can be
# run as ./infer without any install step.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# ── Sub-library directories ───────────────────────────────────────────────────
# Each add_subdirectory call processes that folder's CMakeLists.txt and makes
# its library target available for target_link_libraries below.
add_subdirectory(yolov3)           # → target: yolov3postprocessnew  (SHARED)
add_subdirectory(yolov11)          # → target: yolov11postprocess     (SHARED)
add_subdirectory(detection_utils)  # → target: detection_utils        (STATIC)

# ── Global include paths ──────────────────────────────────────────────────────
# Added to the search path for every target including src/main.cpp.
include_directories(
    ${CMAKE_SOURCE_DIR}                       # resolves "yolov3/...", "yolov11/..."
    ${CMAKE_SOURCE_DIR}/detection_utils        # resolves bare "detection_utils.h"
    ${MX_SDK_HOME}/include                        # MxBase core headers (MxBase.h, etc.)
    ${MX_SDK_HOME}/opensource/include             # third-party headers bundled with SDK
    ${MX_SDK_HOME}/opensource/include/opencv4     # OpenCV 4.x headers from SDK bundle
    ${ASCEND_HOME_PATH}/include                   # CANN / ACL runtime headers
    /usr/include/aarch64-linux-gnu                # AArch64 system headers (if cross-building)
)

# ── Global link-directory search paths ────────────────────────────────────────
# The linker resolves -l<name> flags by searching these directories in order.
link_directories(
    ${MX_SDK_HOME}/lib                              # mxbase, plugintoolkit, mxpidatatype
    ${MX_SDK_HOME}/opensource/lib                   # glib, gstreamer, opencv_world, ffmpeg
    ${MX_SDK_HOME}/lib/modelpostprocessors          # post-processor .so plugins (runtime)
    /usr/local/Ascend/ascend-toolkit/latest/lib64  # CANN runtime (ascendcl, etc.)
    /usr/local/Ascend/driver/lib64/                # Ascend driver interface library
    ${ASCEND_HOME_PATH}/lib64                      # ACL and other CANN libs
    /usr/lib/aarch64-linux-gnu                     # system libs for AArch64
)

# ── Final binary: infer ───────────────────────────────────────────────────────
# src/main.cpp is the only translation unit for the binary.  It contains all
# three runtime modes: --video, --rtsp, and --sweep (max channel sweep).
add_executable(infer src/main.cpp)

target_link_libraries(infer PRIVATE
    detection_utils        # pipeline glue: workers, FFmpeg reader, post-process dispatch
    yolov3postprocessnew   # YOLOv3 anchor-based post-processor  (draws blue boxes)
    yolov11postprocess     # YOLOv11 anchor-free post-processor   (draws green boxes)
    mxbase                 # MxBase runtime: Model::Infer, ImageProcessor, VideoDecoder
    opencv_world           # OpenCV all-in-one (imread, imwrite, rectangle, putText)
    avformat               # FFmpeg: container demux  (mp4, RTSP)
    avcodec                # FFmpeg: codec parsing and bitstream filtering (h264_mp4toannexb)
    avutil                 # FFmpeg: utility types (AVRational, AVDictionary, etc.)
    pthread                # POSIX threads — backend for std::thread on Linux
)