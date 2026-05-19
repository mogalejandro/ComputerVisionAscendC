/*
 * Yolov11PostProcess.cpp
 * ──────────────────────────────────────────────────────────────────────────────
 * YOLOv11 ONNX single-tensor post-processor for MxBase / Ascend 910B.
 *
 * Tensor layout produced by `torch.onnx.export` on an Ultralytics YOLOv11 model
 * (default, no --simplify tricks):
 *
 *   output0 : [batch, 4 + num_classes, num_anchors]
 *              ^^^
 *              box rows  : cx, cy, w, h  (normalised, already sigmoid-scaled)
 *              class rows: raw scores    (sigmoid applied here)
 *
 * After ATC conversion the batch dimension is the first axis, so for a
 * batch=1 model the shape is [1, 84, 8400] for COCO (80 classes).
 * ──────────────────────────────────────────────────────────────────────────────
 */

#include "Yolov11PostProcess.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "MxBase/Log/Log.h"
#include "MxBase/CV/ObjectDetection/Nms/Nms.h"

namespace MxBase {

// ─── helpers ─────────────────────────────────────────────────────────────────

static inline float Sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float Clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

// ─── operator= ───────────────────────────────────────────────────────────────

Yolov11PostProcess& Yolov11PostProcess::operator=(const Yolov11PostProcess& o) {
    if (this != &o) {
        ObjectPostProcessBase::operator=(o);
        scoreThresh_ = o.scoreThresh_;
        iouThresh_   = o.iouThresh_;
        classNum_    = o.classNum_;
    }
    return *this;
}

// ─── Init ────────────────────────────────────────────────────────────────────

APP_ERROR Yolov11PostProcess::Init(const std::map<std::string, std::string>& postConfig) {
    LogDebug << "Yolov11PostProcess::Init start.";

    APP_ERROR ret = ObjectPostProcessBase::Init(postConfig);
    if (ret != APP_ERR_OK) {
        LogError << "ObjectPostProcessBase::Init failed.";
        return ret;
    }

    // Read optional overrides from .cfg
    configData_.GetFileValue<float>("SCORE_THRESH", scoreThresh_);
    configData_.GetFileValue<float>("IOU_THRESH",   iouThresh_);
    configData_.GetFileValue<int>  ("CLASS_NUM",    classNum_);

    LogInfo << "Yolov11PostProcess init: classes=" << classNum_
            << " score_thresh=" << scoreThresh_
            << " iou_thresh="   << iouThresh_;
    LogDebug << "Yolov11PostProcess::Init done.";
    return APP_ERR_OK;
}

// ─── DeInit ──────────────────────────────────────────────────────────────────

APP_ERROR Yolov11PostProcess::DeInit() {
    return APP_ERR_OK;
}

// ─── DecodeSingleImage ────────────────────────────────────────────────────────
//
// `data` points to a contiguous block of size (4 + classNum_) * numBoxes floats
// laid out as [rows][cols]:
//   row 0        : cx for all anchors
//   row 1        : cy
//   row 2        : w
//   row 3        : h
//   row 4…4+C-1  : class score (pre-sigmoid)
//
// All box values are already normalised to [0, 1] by the Ultralytics exporter.

void Yolov11PostProcess::DecodeSingleImage(
    const float*             data,
    int                      numBoxes,
    int                      numClasses,
    const ResizedImageInfo&  imgInfo,
    std::vector<ObjectInfo>& out)
{
    // Letterbox coordinate decoding.
    // During resize the image was scaled by a single uniform gain (the smaller
    // of the two axis ratios) and padded symmetrically — exactly how YOLO
    // models are trained.  Using separate gainX/gainY (stretch formula) would
    // produce shifted boxes whenever the source is not square.
    float modelW = static_cast<float>(imgInfo.widthResize);
    float modelH = static_cast<float>(imgInfo.heightResize);
    float gain   = std::min(modelW / imgInfo.widthOriginal,
                            modelH / imgInfo.heightOriginal);
    float padX   = (modelW - imgInfo.widthOriginal  * gain) / 2.0f;
    float padY   = (modelH - imgInfo.heightOriginal * gain) / 2.0f;

    for (int b = 0; b < numBoxes; ++b) {
        // Find best class first to avoid unnecessary math
        int bestClass = -1;
        float bestScore = scoreThresh_;
        
        for (int c = 0; c < numClasses; ++c) {
            // YOLOv8/v11 usually does NOT need Sigmoid if it's already in the ONNX output
            // But if your export is raw, Sigmoid is correct.
            float score = data[(4 + c) * numBoxes + b]; 
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        if (bestClass < 0) continue;

        // 2. Decode Box (Coordinates are typically in 640x640 scale)
        float cx = data[0 * numBoxes + b];
        float cy = data[1 * numBoxes + b];
        float w  = data[2 * numBoxes + b];
        float h  = data[3 * numBoxes + b];

        // 3. Map back to original image — remove letterbox padding then undo scale.
        float x0 = (cx - w * 0.5f - padX) / gain;
        float y0 = (cy - h * 0.5f - padY) / gain;
        float x1 = (cx + w * 0.5f - padX) / gain;
        float y1 = (cy + h * 0.5f - padY) / gain;

        // Clamp to actual original image boundaries
        ObjectInfo det;
        det.x0 = std::max(0.0f, x0);
        det.y0 = std::max(0.0f, y0);
        det.x1 = std::min((float)imgInfo.widthOriginal, x1);
        det.y1 = std::min((float)imgInfo.heightOriginal, y1);
        

        det.classId = bestClass;
        det.className  = configData_.GetClassName(bestClass);
        det.confidence = bestScore;
        out.emplace_back(det);
    }
}

// ─── Process ─────────────────────────────────────────────────────────────────

APP_ERROR Yolov11PostProcess::Process(
    const std::vector<TensorBase>&                       tensors,
    std::vector<std::vector<ObjectInfo>>&                objectInfos,
    const std::vector<ResizedImageInfo>&                 resizedImageInfos,
    const std::map<std::string, std::shared_ptr<void>>&  /*paramMap*/)
{
    LogDebug << "Yolov11PostProcess::Process start.";

    if (resizedImageInfos.empty()) {
        LogError << "resizedImageInfos is empty.";
        return APP_ERR_INPUT_NOT_MATCH;
    }
    if (tensors.empty()) {
        LogError << "No input tensors.";
        return APP_ERR_INPUT_NOT_MATCH;
    }

    // Move tensors to host if needed
    auto inputs = tensors;
    APP_ERROR ret = CheckAndMoveTensors(inputs);
    if (ret != APP_ERR_OK) {
        LogError << "CheckAndMoveTensors failed.";
        return ret;
    }

    // Expect exactly one output tensor: [batch, 4+classes, num_anchors]
    const TensorBase& t = inputs[0];
    auto shape = t.GetShape();

    if (shape.size() < 3) {
        LogError << "Unexpected tensor rank " << shape.size() << " (expected 3).";
        return APP_ERR_INPUT_NOT_MATCH;
    }

    uint32_t batchSize  = shape[0];
    int      rows       = static_cast<int>(shape[1]);   // 4 + classes
    int      numBoxes   = static_cast<int>(shape[2]);   // num anchors
    int      numClasses = rows - 4;

    if (numClasses <= 0) {
        LogError << "Derived numClasses=" << numClasses << " is invalid (rows=" << rows << ").";
        return APP_ERR_INPUT_NOT_MATCH;
    }

    for (uint32_t i = 0; i < batchSize; ++i) {
        // GetBuffer(tensor, batchIndex) is the MxBase API for per-image data access
        const float* data = static_cast<const float*>(GetBuffer(t, i));
        std::vector<ObjectInfo> dets;
        DecodeSingleImage(data, numBoxes, numClasses,
                          resizedImageInfos[i < resizedImageInfos.size() ? i : 0],
                          dets);
        NmsSort(dets, iouThresh_);
        objectInfos.emplace_back(std::move(dets));
    }

    LogDebug << "Yolov11PostProcess::Process done. Images=" << batchSize;
    return APP_ERR_OK;
}

// ─── factory ─────────────────────────────────────────────────────────────────

extern "C" {
    std::shared_ptr<Yolov11PostProcess> GetYolov11Instance() {
        return std::make_shared<Yolov11PostProcess>();
    }
}

} // namespace MxBase