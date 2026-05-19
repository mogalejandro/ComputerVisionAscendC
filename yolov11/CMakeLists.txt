# ==============================================================================
# CMakeLists.txt  —  yolov11/
# ==============================================================================
# Compiler : GCC 11.4.0  (Ubuntu 11.4.0-1ubuntu1~22.04.2)
# Platform : Ubuntu 22.04 LTS
#
# Builds the `yolov11postprocess` SHARED library — the YOLOv11 post-processor
# plugin for MxBase on Ascend 910B.
#
# What this library does:
#   Receives the single output tensor produced by a YOLOv11 model exported
#   via Ultralytics ONNX export and compiled with ATC.
#
#   Tensor shape (after ATC, batch=1, COCO 80-class example):
#     [1, 84, 8400]   →   [batch, 4 + num_classes, num_anchors]
#
#   Rows 0–3  : cx, cy, w, h  (already in pixel space — no anchor multiply needed)
#   Rows 4–83 : class logit scores (sigmoid applied here before thresholding)
#
#   After confidence filtering and Non-Maximum Suppression, returns ObjectInfo
#   structs with pixel-space bounding boxes.  Boxes are drawn in GREEN by the
#   caller to distinguish them visually from YOLOv3 output (blue).
#
# Key differences from YOLOv3 post-processor:
#   • Single output tensor instead of three multi-scale feature maps.
#   • Anchor-free: box coordinates are predicted directly, no anchor multiply.
#   • No separate objectness score: per-class confidence is the only threshold.
#   • Coordinate layout is transposed relative to classic YOLO exports.
#
# Why SHARED (not STATIC):
#   MxBase's plugin system resolves post-processors as .so files at runtime.
#   The root CMakeLists adds lib/modelpostprocessors/ to the linker search path
#   so the binary can find this .so without LD_LIBRARY_PATH.
#
# Config file: model/yolov11_bs1_fp16.cfg
#   CLASS_NUM, SCORE_THRESH, IOU_THRESH, LABEL_PATH
# ==============================================================================

cmake_minimum_required(VERSION 3.22)   # Ubuntu 22.04 default
project(Yolov11PostProcess LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
# Aligned with the rest of the project (C++17 / GCC 11).
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
set(PLUGIN_NAME "yolov11postprocess")
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
    Yolov11PostProcess.cpp   # tensor decode, sigmoid, NMS, coordinate un-scaling
    Yolov11PostProcess.h     # class declaration — extends ObjectPostProcessBase
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

    plugintoolkit   # registers GetYolov11Instance() as an MxBase plugin entry point
    mxpidatatype    # ObjectInfo, TensorBase, ResizedImageInfo type definitions
    mxbase          # NmsSort(), ObjectPostProcessBase, GetBuffer() helper
)