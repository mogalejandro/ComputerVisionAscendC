# ==============================================================================
# CMakeLists.txt  —  yolov3/
# ==============================================================================
# Compiler : GCC 11.4.0  (Ubuntu 11.4.0-1ubuntu1~22.04.2)
# Platform : Ubuntu 22.04 LTS
#
# Builds the `yolov3postprocessnew` SHARED library — the YOLOv3 post-processor
# plugin for MxBase on Ascend 910B.
#
# What this library does:
#   Receives three raw feature-map tensors from the YOLOv3 .om model:
#     13×13  (large objects)
#     26×26  (medium objects)
#     52×52  (small objects)
#   Applies anchor box decoding, sigmoid activations, objectness filtering,
#   and Non-Maximum Suppression (NMS), then returns ObjectInfo structs with
#   pixel-space bounding boxes.  Boxes are drawn in BLUE by the caller.
#
# Input tensor layout (ATC-compiled .om, NHWC or NCHW — set by MODEL_TYPE in .cfg):
#   [1, H, W, anchors * (5 + classes)]
#   Anchor values are read from the BIASES= field in the .cfg file.
#
# Why SHARED (not STATIC):
#   MxBase's plugin system resolves post-processors as .so files at runtime.
#   The root CMakeLists adds lib/modelpostprocessors/ to the linker search path
#   so the binary can find this .so without LD_LIBRARY_PATH.
#
# Config file: model/yolov3_tf_bs1_fp16.cfg
#   CLASS_NUM, BIASES_NUM, BIASES, SCORE_THRESH, OBJECTNESS_THRESH,
#   IOU_THRESH, YOLO_TYPE, ANCHOR_DIM, MODEL_TYPE
# ==============================================================================

cmake_minimum_required(VERSION 3.22)   # Ubuntu 22.04 default
project(Yolov3PostProcessNew LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
# C++14 would suffice for this post-processor, but we align with the rest of
# the project (C++17 / GCC 11) to avoid ABI mismatches from mixed standards.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── ABI and namespace definitions ─────────────────────────────────────────────
# _GLIBCXX_USE_CXX11_ABI=0
#   GCC 11 on Ubuntu 22.04 defaults to new ABI (=1).  MxBase was built with
#   old ABI (=0).  All targets in the project must use the same value or the
#   linker will fail to resolve std::string symbols across the MxBase boundary.
add_definitions(
    -D_GLIBCXX_USE_CXX11_ABI=0
    -Dgoogle=mindxsdk_private      # avoid collision with system protobuf google::
    -DENABLE_DVPP_INTERFACE        # expose DVPP-related MxBase declarations
)

# ── SDK root ──────────────────────────────────────────────────────────────────
set(MX_SDK_HOME $ENV{MX_SDK_HOME})

# ── Library name and output path ──────────────────────────────────────────────
set(PLUGIN_NAME "yolov3postprocessnew")
set(TARGET_LIBRARY ${PLUGIN_NAME})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)   # .so goes to build/lib/

# ── Include paths ─────────────────────────────────────────────────────────────
include_directories(
    ${MX_SDK_HOME}/include                          # ObjectPostProcessBase.h, NmsSort, etc.
    ${MX_SDK_HOME}/opensource/include               # third-party headers in SDK
    ${MX_SDK_HOME}/opensource/include/gstreamer-1.0 # GStreamer — plugintoolkit hard dep
    ${MX_SDK_HOME}/opensource/include/glib-2.0      # GLib base types
    ${MX_SDK_HOME}/opensource/lib/glib-2.0/include  # glibconfig.h (generated per-arch)
)

# ── Link-directory search paths ───────────────────────────────────────────────
link_directories(
    ${MX_SDK_HOME}/opensource/lib  # glib-2.0, gstreamer-1.0, etc.
    ${MX_SDK_HOME}/lib             # mxbase, plugintoolkit, mxpidatatype
)

# ── Shared library target ─────────────────────────────────────────────────────
add_library(${TARGET_LIBRARY} SHARED
    Yolov3PostProcessNew.cpp   # anchor decode, sigmoid, NMS, coordinate mapping
    Yolov3PostProcessNew.h     # class declaration — extends ObjectPostProcessBase
)

# ── Link dependencies ─────────────────────────────────────────────────────────
# GCC 11 / ld 2.38 on Ubuntu 22.04 enables --as-needed by default, so every
# library that is actually used at link time must be listed explicitly.
target_link_libraries(${TARGET_LIBRARY}
    # GStreamer stack — required by plugintoolkit at link time
    glib-2.0
    gstreamer-1.0
    gobject-2.0
    gstbase-1.0
    gmodule-2.0

    plugintoolkit   # registers GetObjectInstance() as an MxBase plugin entry point
    mxpidatatype    # ObjectInfo, TensorBase, ResizedImageInfo type definitions
    mxbase          # NmsSort(), ObjectPostProcessBase, fastmath sigmoid/exp helpers
)