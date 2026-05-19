#pragma once
/**
 * detection_utils.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Shared declarations for the YOLOv3 / YOLOv11 multi-channel inference
 * pipeline running on Ascend 910B (MxBase + FFmpeg + OpenCV).
 *
 * Consumers: main.cpp  ·  main_stress.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "opencv2/opencv.hpp"
#include "MxBase/Log/Log.h"
#include "MxBase/MxBase.h"
#include "MxBase/E2eInfer/VideoDecoder/VideoDecoder.h"
#include "yolov3/Yolov3PostProcessNew.h"
#include "yolov11/Yolov11PostProcess.h"
#include "tracker/SimpleTracker.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
}

using namespace MxBase;
using Clock = std::chrono::steady_clock;

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Runtime knobs (set once from CLI / config before any pipeline call)
// ─────────────────────────────────────────────────────────────────────────────
namespace cfg {
    // Hard limits
    constexpr uint32_t MAX_VDEC_CHANNELS = 32;   ///< DVPP slots per device

    // Tunable defaults (all mutable so CLI/JSON can override them)
    inline uint32_t BATCH_SIZE           = 32;
    inline uint32_t MAX_QUEUE            = BATCH_SIZE * 2;
    inline uint32_t LOG_INTERVAL_MS      = 5'000;
    inline uint32_t DECODE_SKIP_INTERVAL = 0;
    inline uint32_t NUM_DEVICES          = 1;
    inline bool     WRITE_FRAMES         = false;
    inline bool     WRITE_TRACK_FRAMES   = false;
    inline bool     LOOP_VIDEO           = false;
    inline std::string RESULT_CSV        = "infer_results.csv";

    inline double   MIN_BATCH_FILL_GOOD  = 0.75;
} // namespace cfg

// ─────────────────────────────────────────────────────────────────────────────
// § 2  Model / post-process selection
// ─────────────────────────────────────────────────────────────────────────────
enum class ModelType { YOLOV3, YOLOV11 };

ModelType ModelTypeFromString(const std::string& s);

// ─────────────────────────────────────────────────────────────────────────────
// § 3  ModelConfig  — loaded from config.json
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Holds every parameter that describes one model variant.
 * Populated by LoadModelConfig() from a JSON file.
 */
struct ModelConfig {
    // Model identity
    ModelType   modelType  = ModelType::YOLOV3;

    // File paths
    std::string modelPath  = "./model/yolov3_tf_bsd_rgb.om";
    std::string configPath = "./model/yolov3_tf_bs1_fp16.cfg";
    std::string labelPath  = "./model/yolov3.names";

    // Inference geometry
    uint32_t    resizeWidth        = 416;
    uint32_t    resizeHeight       = 416;

    // Detection filtering
    float                    confidenceThreshold = 0.30f; ///< post-NMS score filter (also passed as SCORE_THRESH to post-processor)
    float                    nmsIouThreshold     = 0.45f; ///< NMS IoU threshold passed to post-processor
    std::set<std::string>    classFilter;                 ///< keep only these class names; empty = keep all

    // Tracker hyperparameters
    float   trackerIouThresh = 0.30f; ///< min IoU to match a detection to an existing track
    int     trackerMaxMisses = 30;    ///< frames without a match before a track is deleted
    int     trackerMaxTraj   = 50;    ///< max centroid history points per track

    // Device assignment (can be overridden at runtime)
    uint32_t    deviceId   = 0;
    uint32_t    numDevices = 1;   ///< total NPU devices to use

    // Batch sizing
    uint32_t    batchSize  = 32;
};

/**
 * Parse a config.json file and return a populated ModelConfig.
 * Falls back to defaults for any missing key and prints a warning.
 *
 * JSON schema (all fields optional – defaults shown):
 * {
 *   "model_type"            : "yolov3",
 *   "model_path"            : "./model/yolov3_tf_bsd_rgb.om",
 *   "config_path"           : "./model/yolov3_tf_bs1_fp16.cfg",
 *   "label_path"            : "./model/yolov3.names",
 *   "resize_width"          : 416,
 *   "resize_height"         : 416,
 *   "confidence_threshold"  : 0.30,
 *   "num_devices"           : 1,
 *   "batch_size"            : 32
 * }
 */
ModelConfig LoadModelConfig(const std::string& jsonPath);

// ─────────────────────────────────────────────────────────────────────────────
// § 4  Per-frame payload
// ─────────────────────────────────────────────────────────────────────────────
struct DecodedFrame {
    MxBase::Image     image;
    uint32_t          frameId   = 0;
    Clock::time_point decodeTime;
};

// ─────────────────────────────────────────────────────────────────────────────
// § 5  Per-stream statistics
// ─────────────────────────────────────────────────────────────────────────────
struct StreamStats {
    std::atomic<uint64_t> framesDecoded {0};
    std::atomic<uint64_t> framesInferred{0};
    std::atomic<uint64_t> framesDropped {0};
    std::atomic<uint64_t> totalLagUs    {0};
    std::atomic<uint64_t> lagSamples    {0};
};

// ─────────────────────────────────────────────────────────────────────────────
// § 6  Per-stream state
// ─────────────────────────────────────────────────────────────────────────────
struct StreamState {
    std::queue<DecodedFrame> frameQueue;
    std::mutex               queueMutex;
    std::condition_variable  queueCv;
    std::atomic<bool>        decodeFinished{false};
    std::atomic<uint32_t>    callbackCount {0};
    std::atomic<uint32_t>    packetsSent   {0};
    StreamStats              stats;
};

// ─────────────────────────────────────────────────────────────────────────────
// § 7  Multiplexing callback context
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Passed as userData to MxBase::VideoDecoder::Decode().
 * Allows N logical streams to share one physical DVPP slot.
 */
struct MuxCallbackCtx {
    std::vector<StreamState*> logicalStates;
    std::atomic<uint32_t>     roundRobin{0};
};

/// MxBase decoder callback — routes each decoded frame round-robin to a logical stream.
APP_ERROR OnFrameDecodedMux(MxBase::Image& img, uint32_t channelId,
                             uint32_t frameId, void* userData);

// ─────────────────────────────────────────────────────────────────────────────
// § 8  FFmpeg reader thread
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Demuxes @p url in a dedicated thread, converts packets to Annex-B, and
 * feeds them to @p decoder.  Loops if @p isFile && cfg::LOOP_VIDEO.
 */
void FFmpegReaderThread(const std::string& url, bool isFile,
                        MxBase::VideoDecoder* decoder,
                        MuxCallbackCtx* ctx);

// ─────────────────────────────────────────────────────────────────────────────
// § 9  Stream probing  (must precede StreamWorker)
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Dimensions and frame-rate of a video source, as reported by FFmpeg.
 * width/height are the coded (not display) dimensions used to configure
 * the DVPP VideoDecoder buffer allocation.
 */
struct StreamInfo {
    uint32_t width  = 0;   ///< coded frame width  in pixels
    uint32_t height = 0;   ///< coded frame height in pixels
    double   fps    = 0.0; ///< average frame-rate (0 if unknown)
    std::string codec = "h264"; 
};

/**
 * Open @p url with FFmpeg, find the first video stream, read its coded
 * dimensions and frame-rate, then close immediately.
 * Returns a zeroed StreamInfo on any failure.
 */
StreamInfo ProbeStream(const std::string& url, bool isFile);

// ─────────────────────────────────────────────────────────────────────────────
// § 10  StreamWorker
// ─────────────────────────────────────────────────────────────────────────────
/**
 * Owns or borrows a DVPP decoder slot for one logical stream.
 *
 * Lifecycle:
 *   Init()   — allocate / join DVPP slot
 *   Start()  — launch FFmpegReaderThread
 *   Stop()   — signal finish and join thread
 */
class StreamWorker {
public:
    StreamWorker(uint32_t id, const std::string& url, bool isFile,
                 uint32_t deviceId, uint32_t physicalSlot);

    /**
     * @param muxCtx    nullptr → this worker creates and owns the DVPP decoder.
     *                  non-null → this worker is a secondary sharing an existing slot.
     * @param probed    Stream dimensions probed once by RunPipeline before the
     *                  worker loop.  Only used when muxCtx == nullptr (slot owner).
     *                  Pass a zeroed StreamInfo to have Init fall back to defaults.
     */
    APP_ERROR Init(MuxCallbackCtx* muxCtx = nullptr,
                   StreamInfo      probed = {});
    void      Start();
    void      Stop();

    /// Called by RunPipeline to hand secondary workers the shared decoder pointer.
    void SetSharedDecoder(MxBase::VideoDecoder* dec);

    bool         PopFrame(DecodedFrame& df);
    bool         IsActive()   const;
    uint32_t     GetId()      const;
    size_t       QueueDepth() const;
    StreamStats& GetStats();

    MxBase::VideoDecoder* GetDecoder()        const;
    MuxCallbackCtx*       GetMuxCtx();
    const StreamInfo&     GetStreamInfo()     const;

private:
    uint32_t    id_;
    std::string url_;
    bool        isFile_;
    uint32_t    deviceId_;
    uint32_t    physicalSlot_;
    StreamInfo  streamInfo_;   ///< probed once in Init() for the slot-owner worker

    StreamState                           state_;
    MuxCallbackCtx*                       muxCtx_        = nullptr;
    std::unique_ptr<MuxCallbackCtx>       ownMuxCtx_;        ///< non-null only for slot owner
    std::unique_ptr<MxBase::VideoDecoder> decoder_;           ///< non-null only for slot owner
    MxBase::VideoDecoder*                 sharedDecoder_ = nullptr;
    std::thread                           readerThread_;
};

// ─────────────────────────────────────────────────────────────────────────────
// § 11  Post-processing helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Post-process a single frame from a YOLOv3 inference batch, drawing
 * bounding boxes onto @p resultMat.
 */
APP_ERROR RunPostProcessYolov3(Yolov3PostProcessNew&         pp,
                                MxBase::Image&               decodedImage,
                                std::vector<MxBase::Tensor>& outputs,
                                uint32_t                     batchIndex,
                                cv::Mat&                     resultMat,
                                const ModelConfig&           mcfg);

/**
 * Post-process a single frame from a YOLOv11 inference batch, drawing
 * bounding boxes onto @p resultMat.
 */
APP_ERROR RunPostProcessYolov11(MxBase::Yolov11PostProcess&  pp,
                                 MxBase::Image&              decodedImage,
                                 std::vector<MxBase::Tensor>& outputs,
                                 uint32_t                    batchIndex,
                                 cv::Mat&                    resultMat,
                                 const ModelConfig&          mcfg);

/**
 * Run post-processing and return raw detections without drawing.
 * Used internally by RunPipeline when tracker or separate draw control is needed.
 */
APP_ERROR GetDetectionsYolov3(Yolov3PostProcessNew&         pp,
                               MxBase::Image&               decodedImage,
                               std::vector<MxBase::Tensor>& outputs,
                               uint32_t                     batchIndex,
                               const ModelConfig&           mcfg,
                               std::vector<ObjectInfo>&     out);

APP_ERROR GetDetectionsYolov11(MxBase::Yolov11PostProcess&  pp,
                                MxBase::Image&              decodedImage,
                                std::vector<MxBase::Tensor>& outputs,
                                uint32_t                    batchIndex,
                                const ModelConfig&          mcfg,
                                std::vector<ObjectInfo>&    out);

// ─────────────────────────────────────────────────────────────────────────────
// § 12  CSV logger
// ─────────────────────────────────────────────────────────────────────────────
class CsvLog {
public:
    explicit CsvLog(const std::string& path);
    void Write(int channels, uint64_t inferred, double fps,
               double fill, uint32_t skip, double lagMs, double inferMs);
private:
    std::ofstream file_;
    std::mutex    mu_;
};

// ─────────────────────────────────────────────────────────────────────────────
// § 13  Core pipeline
// ─────────────────────────────────────────────────────────────────────────────
struct PipelineResult {
    double   avgFps        = 0;   ///< total inference throughput across all channels
    double   fpsPerChannel = 0;   ///< avgFps / numChannels — the real-time signal
    double   avgFill       = 0;
    double   avgLagMs      = 0;
    uint64_t totalInferred = 0;
    uint32_t numChannels   = 0;
};

/**
 * Build workers, run the batch-inference loop for @p runSeconds
 * (0 = run until all streams end), write stats to @p csv.
 *
 * @param srcInfo  Stream geometry pre-probed by the caller.  If non-zero,
 *                 RunPipeline skips its internal ProbeStream call entirely.
 *                 Pass a zeroed StreamInfo (the default) to probe internally.
 */
PipelineResult RunPipeline(const std::vector<std::string>& urls,
                            bool isFile,
                            const ModelConfig& mcfg,
                            uint32_t   runSeconds = 0,
                            CsvLog*    csv        = nullptr,
                            int        channels   = 1,
                            StreamInfo srcInfo    = {});

// ─────────────────────────────────────────────────────────────────────────────
// § 14  Misc utilities
// ─────────────────────────────────────────────────────────────────────────────
void        MkdirP(const std::string& path);
std::string Timestamp();