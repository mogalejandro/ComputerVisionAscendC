/*
 * Yolov11PostProcess.h
 * ──────────────────────────────────────────────────────────────────────────────
 * Post-processor for YOLOv11 models exported via ONNX (Ultralytics default).
 *
 * Expected output tensor layout (after ATC conversion):
 *   Shape : [batch, 4 + num_classes, num_boxes]   (transposed export)
 *   Box   : cx, cy, w, h  (normalised 0–1)
 *   Scores: raw logits or post-sigmoid floats — both are handled
 *
 * Config keys (same .cfg style as YOLOv3):
 *   CLASS_NUM          number of classes  (default 80)
 *   SCORE_THRESH       confidence threshold (default 0.25)
 *   IOU_THRESH         NMS IoU threshold   (default 0.45)
 *   LABEL_PATH         path to .names file
 * ──────────────────────────────────────────────────────────────────────────────
 */

#ifndef YOLOV11_POST_PROCESS_H
#define YOLOV11_POST_PROCESS_H

#include <map>
#include <string>
#include <vector>
#include <memory>

#include "MxBase/PostProcessBases/ObjectPostProcessBase.h"

namespace MxBase {

class Yolov11PostProcess : public ObjectPostProcessBase
{
public:
    Yolov11PostProcess()  = default;
    ~Yolov11PostProcess() = default;
    Yolov11PostProcess(const Yolov11PostProcess&)            = default;
    Yolov11PostProcess& operator=(const Yolov11PostProcess&);

    APP_ERROR Init(const std::map<std::string, std::string>& postConfig) override;
    APP_ERROR DeInit() override;

    APP_ERROR Process(
        const std::vector<TensorBase>&              tensors,
        std::vector<std::vector<ObjectInfo>>&       objectInfos,
        const std::vector<ResizedImageInfo>&        resizedImageInfos = {},
        const std::map<std::string, std::shared_ptr<void>>& paramMap = {}) override;

private:
    // Decode one image from the batch tensor
    void DecodeSingleImage(
        const float*                    data,        // pointer to this image's slice
        int                             numBoxes,
        int                             numClasses,
        const ResizedImageInfo&         imgInfo,
        std::vector<ObjectInfo>&        out);

    float scoreThresh_ = 0.25f;
    float iouThresh_   = 0.45f;
    int   classNum_    = 80;
};

extern "C" {
    std::shared_ptr<Yolov11PostProcess> GetYolov11Instance();
}

} // namespace MxBase

#endif // YOLOV11_POST_PROCESS_H