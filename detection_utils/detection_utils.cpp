/**
 * detection_utils.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Implementation of the shared inference pipeline for YOLOv3 / YOLOv11 on
 * Ascend 910B (MxBase + FFmpeg + OpenCV).
 *
 * Included by: main.cpp  (via detection_utils.h)
 *
 * Pipeline overview
 * ─────────────────
 *   FFmpegReaderThread  →  raw frameQueue  (per worker, YUV frames)
 *   ResizeThread        →  readyQueue      (host-resident tensors, one per worker)
 *   InferenceLoop       →  batch from readyQueue → NPU → post-process
 *
 * Key design decisions
 * ────────────────────
 *  1. Per-worker resize thread pool
 *     DVPP resize + host DMA are completely removed from the inference hot-path.
 *     The NPU is never starved waiting for pre-processing.
 *
 *  2. Triple-stage NPU inference loop (assembly overlaps with NPU)
 *     OLD order (serial):
 *       [get/post-process 35ms] → [assemble 40ms] → [launch] → repeat
 *       NPU was idle during the entire assembly window.
 *
 *     NEW order (parallel):
 *       [assemble batch N+1]  ← runs while NPU processes batch N
 *       [get batch N + post-process]  ← NPU likely already done by now
 *       [launch batch N+1]
 *       → repeat
 *
 *     The NPU idle gap between batches is eliminated.  batchTimeout is set
 *     to the rolling-average NPU duration × 1.2 so the assembler deadline
 *     matches actual NPU cadence.
 *
 *  3. Raw-queue depth proportional to channel count  ← new
 *     cfg::MAX_QUEUE = max(4, (BATCH_SIZE * 4) / numLogical)
 *
 *     Problem: at 32 channels / batch=16 the old formula (batchSize*4 = 64
 *     per queue) allowed 32×64 = 2048 total buffered raw frames.  Resize
 *     threads raced ahead, the ready queue pinned at its ceiling, and every
 *     inferred frame had been sitting for 2+ seconds — making fps=735 and
 *     lag=2300 ms meaningless numbers.
 *
 *     Fix: distribute the 4-batch total budget evenly across workers.  With
 *     32 channels / batch=16: (16*4)/32 = 2 frames/queue → total budget 64.
 *     With 8 channels / batch=16: (16*4)/8 = 8 frames/queue → more headroom
 *     where it's safe.
 *
 *  4. Ready-queue tight absolute ceiling
 *     READY_QUEUE_MAX = BATCH_SIZE * 4
 *     Keeps back-pressure on resize threads so tensors never age long enough
 *     for their ACL device-memory backing to be freed (error 507899).
 *
 *  5. Tensors moved to host before queuing
 *     ResizeThread calls t.ToHost() immediately after ConvertToTensor().
 *     This means every tensor in the ready queue lives in CPU RAM — stable,
 *     copy-safe, and immune to ACL device-memory reclaim.
 *
 *  6. Safe batch padding
 *     Pad frames copy the host-resident tensor from slot 0.  Since host
 *     MxBase::Tensor copies are deep (they clone the CPU buffer), no aliasing
 *     of ACL pointers occurs and BatchConcat never races a freed buffer.
 *
 *  7. Adaptive batchTimeout
 *     Initial timeout derived from source FPS and batch size.  After every
 *     inferred batch the actual NPU duration is measured and a 16-batch
 *     rolling average is maintained.  batchTimeout is set to
 *     avg_infer_ms × 1.2, clamped [10 ms, 150 ms].
 *
 *     Problem: a fixed timeout derived from source FPS and channel count is
 *     not correlated with real NPU latency.  At 48 channels / batch=32 the
 *     formula gave ~40 ms but actual NPU batches took ~80 ms — the assembler
 *     timed out too early, padded every batch, and fill stayed at 0.84.
 *
 *     Fix: adapt to measured NPU speed.  Once the rolling average stabilises
 *     (after ~16 batches) the timeout matches NPU cadence and the assembler
 *     can always collect a full batch before the deadline.
 *
 *  8. Decoded / resized counters in log
 *     framesDecoded (DVPP callback) and framesInferred (resize thread) are
 *     now printed each log interval alongside inferred, making it easy to
 *     pinpoint whether a gap opens at the decode or resize stage.
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "detection_utils.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <filesystem>
#include <sys/stat.h>
#include <thread>
#include <vector>

// ─── tiny JSON helpers (no external dependency) ──────────────────────────────
namespace {

void Trim(std::string& s) {
    const char* ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    auto end = s.find_last_not_of(ws);
    if (end != std::string::npos) s.erase(end + 1);
}

std::string ExtractQuoted(const std::string& s) {
    auto a = s.find('"');
    if (a == std::string::npos) return "";
    auto b = s.find('"', a + 1);
    if (b == std::string::npos) return "";
    return s.substr(a + 1, b - a - 1);
}

std::string ExtractValue(const std::string& line) {
    auto colon = line.find(':');
    if (colon == std::string::npos) return "";
    std::string val = line.substr(colon + 1);
    auto comma = val.rfind(',');
    if (comma != std::string::npos) val = val.substr(0, comma);
    Trim(val);
    return val;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// § 1  Misc utilities
// ─────────────────────────────────────────────────────────────────────────────
void MkdirP(const std::string& path) {
    std::string tmp = path;
    for (size_t i = 1; i < tmp.size(); ++i) {
        if (tmp[i] == '/') {
            tmp[i] = '\0';
            ::mkdir(tmp.c_str(), 0755);
            tmp[i] = '/';
        }
    }
    ::mkdir(tmp.c_str(), 0755);
}

std::string Timestamp() {
    auto now = std::chrono::system_clock::now();
    auto ms  = std::chrono::duration_cast<std::chrono::milliseconds>(
                   now.time_since_epoch()) % 1000;
    auto t   = std::chrono::system_clock::to_time_t(now);
    std::tm bt{};
    localtime_r(&t, &bt);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%H:%M:%S")
        << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 1b  ProbeStream
// ─────────────────────────────────────────────────────────────────────────────
StreamInfo ProbeStream(const std::string& url, bool isFile) {
    StreamInfo info;

    AVDictionary* opts = nullptr;
    if (!isFile) {
        av_dict_set(&opts, "rtsp_transport", "tcp",      0);
        av_dict_set(&opts, "stimeout",        "5000000", 0);
    }

    AVFormatContext* fmtCtx = nullptr;
    if (avformat_open_input(&fmtCtx, url.c_str(), nullptr, &opts) != 0) {
        av_dict_free(&opts);
        std::cerr << "[WARN] ProbeStream: cannot open '" << url << "'\n";
        return info;
    }
    av_dict_free(&opts);
    avformat_find_stream_info(fmtCtx, nullptr);

    for (unsigned i = 0; i < fmtCtx->nb_streams; ++i) {
        AVStream* st = fmtCtx->streams[i];
        if (st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            info.width  = static_cast<uint32_t>(st->codecpar->width);
            info.height = static_cast<uint32_t>(st->codecpar->height);
            if (st->avg_frame_rate.den > 0)
                info.fps = static_cast<double>(st->avg_frame_rate.num)
                           / st->avg_frame_rate.den;
            
            // Detectar el codec del video
            if (st->codecpar->codec_id == AV_CODEC_ID_H264) {
                info.codec = "h264";
            } else if (st->codecpar->codec_id == AV_CODEC_ID_H265) {
                info.codec = "h265";
            } else {
                info.codec = "unknown";
            }
            
            break;
        }
    }

    avformat_close_input(&fmtCtx);

    if (info.width == 0 || info.height == 0)
        std::cerr << "[WARN] ProbeStream: no video stream found in '"
                  << url << "'\n";
    else
        std::cout << "[PROBE] " << url
                  << "  " << info.width << "×" << info.height
                  << "  " << std::fixed << std::setprecision(2)
                  << info.fps << " fps"
                  << "  codec=" << info.codec << "\n";  // <-- muestra el codec

    return info;
}

ModelType ModelTypeFromString(const std::string& s) {
    if (s == "yolov11" || s == "YOLOV11") return ModelType::YOLOV11;
    return ModelType::YOLOV3;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 3  LoadModelConfig  (JSON parser)
// ─────────────────────────────────────────────────────────────────────────────
ModelConfig LoadModelConfig(const std::string& jsonPath) {
    ModelConfig mc;

    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        std::cerr << "[WARN] Cannot open model config '" << jsonPath
                  << "' — using built-in defaults.\n";
        return mc;
    }

    std::string line;
    while (std::getline(f, line)) {
        Trim(line);
        if (line.empty() || line.substr(0, 2) == "//") continue;
        if (line.find(':') == std::string::npos) continue;

        std::string key    = ExtractQuoted(line);
        if (key.empty()) continue;
        std::string rawVal = ExtractValue(line);

        if (key == "model_type") {
            std::string v = ExtractQuoted(rawVal.empty()
                                ? line.substr(line.find(':') + 1) : rawVal);
            if (v.empty()) v = rawVal;
            mc.modelType = ModelTypeFromString(v);

        } else if (key == "model_path") {
            mc.modelPath = ExtractQuoted(rawVal.empty() ? line : rawVal);
            if (mc.modelPath.empty()) mc.modelPath = rawVal;

        } else if (key == "config_path") {
            mc.configPath = ExtractQuoted(rawVal.empty() ? line : rawVal);
            if (mc.configPath.empty()) mc.configPath = rawVal;

        } else if (key == "label_path") {
            mc.labelPath = ExtractQuoted(rawVal.empty() ? line : rawVal);
            if (mc.labelPath.empty()) mc.labelPath = rawVal;

        } else if (key == "resize_width") {
            mc.resizeWidth = static_cast<uint32_t>(std::stoul(rawVal));

        } else if (key == "resize_height") {
            mc.resizeHeight = static_cast<uint32_t>(std::stoul(rawVal));

        } else if (key == "confidence_threshold") {
            mc.confidenceThreshold = std::stof(rawVal);

        } else if (key == "nms_iou_threshold") {
            mc.nmsIouThreshold = std::stof(rawVal);

        } else if (key == "tracker_iou_thresh") {
            mc.trackerIouThresh = std::stof(rawVal);

        } else if (key == "tracker_max_misses") {
            mc.trackerMaxMisses = std::stoi(rawVal);

        } else if (key == "tracker_max_traj") {
            mc.trackerMaxTraj = std::stoi(rawVal);

        } else if (key == "num_devices") {
            mc.numDevices    = static_cast<uint32_t>(std::stoul(rawVal));
            cfg::NUM_DEVICES = mc.numDevices;

        } else if (key == "batch_size") {
            mc.batchSize    = static_cast<uint32_t>(std::stoul(rawVal));
            cfg::BATCH_SIZE = mc.batchSize;
            // MAX_QUEUE is recomputed per RunPipeline call based on channel
            // count.  Store a conservative base here for pre-pipeline reads.
            cfg::MAX_QUEUE = mc.batchSize * 4;

        } else if (key == "class_filter") {
            // Value is a JSON array, possibly spanning multiple lines.
            // Collect everything up to (and including) the closing ']'.
            std::string arrBuf = rawVal;
            if (arrBuf.find('[') != std::string::npos) {
                while (arrBuf.find(']') == std::string::npos) {
                    std::string next;
                    if (!std::getline(f, next)) break;
                    arrBuf += next;
                }
                size_t pos = 0;
                while (true) {
                    auto a = arrBuf.find('"', pos);
                    if (a == std::string::npos) break;
                    auto b = arrBuf.find('"', a + 1);
                    if (b == std::string::npos) break;
                    mc.classFilter.insert(arrBuf.substr(a + 1, b - a - 1));
                    pos = b + 1;
                }
            }
        }
    }

    std::cout << "[CONFIG] "
              << "model_type="      << (mc.modelType == ModelType::YOLOV11
                                        ? "yolov11" : "yolov3")
              << "  model_path="    << mc.modelPath
              << "  config_path="   << mc.configPath
              << "  label_path="    << mc.labelPath
              << "  resize="        << mc.resizeWidth << "x" << mc.resizeHeight
              << "  conf_thresh="   << mc.confidenceThreshold
              << "  nms_iou="       << mc.nmsIouThreshold
              << "  num_devices="   << mc.numDevices
              << "  batch_size="    << mc.batchSize
              << "  tracker_iou="   << mc.trackerIouThresh
              << "  tracker_miss="  << mc.trackerMaxMisses
              << "  tracker_traj="  << mc.trackerMaxTraj;
    if (!mc.classFilter.empty()) {
        std::cout << "  class_filter=[";
        bool first = true;
        for (const auto& c : mc.classFilter) {
            if (!first) std::cout << ",";
            std::cout << c;
            first = false;
        }
        std::cout << "]";
    }
    std::cout << "\n";

    return mc;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 4  CsvLog
// ─────────────────────────────────────────────────────────────────────────────
CsvLog::CsvLog(const std::string& path) : file_(path, std::ios::app) {
    if (file_.tellp() == 0)
        file_ << "timestamp,channels,inferred,avg_fps,batch_fill,"
                 "skip_interval,avg_lag_ms,avg_infer_ms\n";
}

void CsvLog::Write(int channels, uint64_t inferred, double fps,
                   double fill, uint32_t skip, double lagMs, double inferMs) {
    std::lock_guard<std::mutex> g(mu_);
    file_ << Timestamp() << ","
          << channels    << ","
          << inferred    << ","
          << std::fixed  << std::setprecision(2) << fps     << ","
          << std::fixed  << std::setprecision(3) << fill    << ","
          << skip        << ","
          << std::fixed  << std::setprecision(1) << lagMs   << ","
          << std::fixed  << std::setprecision(1) << inferMs << "\n";
    file_.flush();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 5  MuxCallbackCtx  /  OnFrameDecodedMux
// ─────────────────────────────────────────────────────────────────────────────
APP_ERROR OnFrameDecodedMux(MxBase::Image& img, uint32_t /*channelId*/,
                             uint32_t frameId, void* userData)
{
    if (!userData) return APP_ERR_COMM_INVALID_POINTER;

    auto*    ctx   = static_cast<MuxCallbackCtx*>(userData);
    uint32_t idx   = ctx->roundRobin.fetch_add(1) % ctx->logicalStates.size();
    StreamState* state = ctx->logicalStates[idx];

    ++state->callbackCount;
    ++state->stats.framesDecoded;

    std::unique_lock<std::mutex> lock(state->queueMutex);
    if (state->frameQueue.size() >= cfg::MAX_QUEUE) {
        state->frameQueue.pop();
        ++state->stats.framesDropped;
    }
    if (state->decodeFinished) return APP_ERR_OK;

    DecodedFrame df;
    df.image      = img;
    df.frameId    = frameId;
    df.decodeTime = Clock::now();
    state->frameQueue.push(std::move(df));
    state->queueCv.notify_one();
    return APP_ERR_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 6  FFmpegReaderThread
// ─────────────────────────────────────────────────────────────────────────────
void FFmpegReaderThread(const std::string& url, bool isFile,
                        MxBase::VideoDecoder* decoder,
                        MuxCallbackCtx* ctx)
{
    StreamState* state = ctx->logicalStates[0];

    auto openInput = [&](AVFormatContext*& fmtCtx) -> bool {
        fmtCtx = nullptr;
        AVDictionary* opts = nullptr;
        if (!isFile) {
            av_dict_set(&opts, "rtsp_transport", "tcp",      0);
            av_dict_set(&opts, "stimeout",        "5000000", 0);
            av_dict_set(&opts, "max_delay",        "0",      0);
        }
        int r = avformat_open_input(&fmtCtx, url.c_str(), nullptr, &opts);
        av_dict_free(&opts);
        if (r < 0) {
            std::cerr << "FFmpeg: cannot open: " << url << "\n";
            return false;
        }
        avformat_find_stream_info(fmtCtx, nullptr);
        return true;
    };

    auto findVideoStream = [](AVFormatContext* fCtx) -> int {
        for (unsigned i = 0; i < fCtx->nb_streams; i++)
            if (fCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                return static_cast<int>(i);
        return -1;
    };

    auto setupBsf = [](AVCodecParameters* par, AVBSFContext*& bsfCtx) -> bool {
        const char* name = (par->codec_id == AV_CODEC_ID_H264)
                            ? "h264_mp4toannexb" : "hevc_mp4toannexb";
        const AVBitStreamFilter* bsf = av_bsf_get_by_name(name);
        if (!bsf) return false;
        if (av_bsf_alloc(bsf, &bsfCtx) != 0) return false;
        avcodec_parameters_copy(bsfCtx->par_in, par);
        return av_bsf_init(bsfCtx) == 0;
    };

    AVFormatContext* fmtCtx = nullptr;
    if (!openInput(fmtCtx)) {
        state->decodeFinished = true;
        state->queueCv.notify_all();
        return;
    }

    int vidIdx = findVideoStream(fmtCtx);
    if (vidIdx < 0) {
        std::cerr << "FFmpeg: no video stream in " << url << "\n";
        avformat_close_input(&fmtCtx);
        state->decodeFinished = true;
        state->queueCv.notify_all();
        return;
    }

    AVBSFContext* bsfCtx = nullptr;
    bool useBsf = setupBsf(fmtCtx->streams[vidIdx]->codecpar, bsfCtx);

    AVPacket* pkt    = av_packet_alloc();
    AVPacket* bsfPkt = av_packet_alloc();
    uint32_t  frameId = 0;

    auto sendToDecoder = [&](AVPacket* p) {
        auto data = std::shared_ptr<uint8_t>(
            new uint8_t[p->size], [](uint8_t* ptr){ delete[] ptr; });
        std::memcpy(data.get(), p->data, p->size);
        APP_ERROR ret = decoder->Decode(data,
                                        static_cast<uint32_t>(p->size),
                                        frameId, ctx);
        if (ret != APP_ERR_OK)
            std::cout << "[WARN] Decode ret=" << ret
                      << " frame=" << frameId << "\n";
        ++frameId;
        ++state->packetsSent;
    };

    uint32_t loopCount = 0;
    do {
        if (loopCount > 0) {
            if (isFile) {
                av_seek_frame(fmtCtx, vidIdx, 0, AVSEEK_FLAG_BACKWARD);
            } else {
                avformat_close_input(&fmtCtx);
                if (!openInput(fmtCtx)) break;
                vidIdx = findVideoStream(fmtCtx);
                if (vidIdx < 0) break;
            }
        }
        ++loopCount;

        while (av_read_frame(fmtCtx, pkt) >= 0) {
            if (pkt->stream_index == vidIdx && pkt->data && pkt->size > 0) {
                if (useBsf) {
                    av_bsf_send_packet(bsfCtx, pkt);
                    while (av_bsf_receive_packet(bsfCtx, bsfPkt) == 0) {
                        sendToDecoder(bsfPkt);
                        av_packet_unref(bsfPkt);
                    }
                } else {
                    sendToDecoder(pkt);
                }
            }
            av_packet_unref(pkt);
            if (state->decodeFinished) goto done;
        }
    } while (isFile && cfg::LOOP_VIDEO && !state->decodeFinished);

done:
    if (useBsf) {
        av_bsf_send_packet(bsfCtx, nullptr);
        while (av_bsf_receive_packet(bsfCtx, bsfPkt) == 0)
            av_packet_unref(bsfPkt);
        av_bsf_free(&bsfCtx);
    }
    av_packet_free(&bsfPkt);
    av_packet_free(&pkt);
    avformat_close_input(&fmtCtx);
    decoder->Flush();

    {
        uint32_t sent     = state->packetsSent.load();
        auto     deadline = Clock::now() + std::chrono::milliseconds(2000);
        while (state->callbackCount.load() < sent && Clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    state->decodeFinished = true;
    state->queueCv.notify_all();
}

// ─────────────────────────────────────────────────────────────────────────────
// § 7  StreamWorker
// ─────────────────────────────────────────────────────────────────────────────
StreamWorker::StreamWorker(uint32_t id, const std::string& url, bool isFile,
                           uint32_t deviceId, uint32_t physicalSlot)
    : id_(id), url_(url), isFile_(isFile),
      deviceId_(deviceId), physicalSlot_(physicalSlot) {}

APP_ERROR StreamWorker::Init(MuxCallbackCtx* muxCtx, StreamInfo probed) {
    muxCtx_ = muxCtx;
    if (muxCtx_) {
        // Secondary worker sharing an existing decoder slot — no decoder needed.
        muxCtx_->logicalStates.push_back(&state_);
        return APP_ERR_OK;
    }

    ownMuxCtx_ = std::make_unique<MuxCallbackCtx>();
    ownMuxCtx_->logicalStates.push_back(&state_);
    muxCtx_    = ownMuxCtx_.get();

    // Use the pre-probed dimensions supplied by RunPipeline (probed once for
    // all workers before the creation loop).  Fall back to 1920×1080 only if
    // the caller passed a zeroed StreamInfo.
    if (probed.width == 0 || probed.height == 0) {
        std::cerr << "[WARN] StreamWorker " << id_
                  << ": no stream info supplied, defaulting to 1920×1080\n";
        probed.width  = 1920;
        probed.height = 1080;
    }
    streamInfo_ = probed;

    MxBase::VideoDecodeConfig vcfg;
    vcfg.width             = streamInfo_.width;
    vcfg.height            = streamInfo_.height;
    // Auto-detect codec from stream info
    if (streamInfo_.codec == "h264") {
        vcfg.inputVideoFormat = StreamFormat::H264_MAIN_LEVEL;
    } else {
        vcfg.inputVideoFormat = StreamFormat::H265_MAIN_LEVEL;
    }
    vcfg.outputImageFormat = ImageFormat::RGB_888; //YUV_SP_420
    vcfg.callbackFunc      = OnFrameDecodedMux;
    vcfg.skipInterval      = cfg::DECODE_SKIP_INTERVAL;

    std::cout << "[DBG] Creating VideoDecoder device=" << deviceId_
              << " slot=" << physicalSlot_
              << " size=" << streamInfo_.width << "×" << streamInfo_.height
              << "\n";
    decoder_ = std::make_unique<MxBase::VideoDecoder>(vcfg, deviceId_,
                                                       physicalSlot_);
    std::cout << "[DBG] VideoDecoder created device=" << deviceId_
              << " slot=" << physicalSlot_ << "\n";
    return APP_ERR_OK;
}

void StreamWorker::Start() {
    MxBase::VideoDecoder* dec = decoder_ ? decoder_.get() : sharedDecoder_;
    readerThread_ = std::thread(&FFmpegReaderThread, url_, isFile_, dec,
                                muxCtx_);
}

void StreamWorker::Stop() {
    {
        std::unique_lock<std::mutex> g(state_.queueMutex);
        state_.decodeFinished = true;
        state_.queueCv.notify_all();
    }
    if (readerThread_.joinable()) readerThread_.join();
}

void StreamWorker::SetSharedDecoder(MxBase::VideoDecoder* dec) {
    sharedDecoder_ = dec;
}

bool StreamWorker::PopFrame(DecodedFrame& df) {
    std::lock_guard<std::mutex> g(state_.queueMutex);
    if (state_.frameQueue.empty()) return false;
    df = std::move(state_.frameQueue.front());
    state_.frameQueue.pop();
    state_.queueCv.notify_one();
    return true;
}

bool         StreamWorker::IsActive()   const { return !state_.decodeFinished || !state_.frameQueue.empty(); }
uint32_t     StreamWorker::GetId()      const { return id_; }
size_t       StreamWorker::QueueDepth() const { return state_.frameQueue.size(); }
StreamStats& StreamWorker::GetStats()         { return state_.stats; }

MxBase::VideoDecoder* StreamWorker::GetDecoder()    const { return decoder_.get(); }
MuxCallbackCtx*       StreamWorker::GetMuxCtx()           { return ownMuxCtx_.get(); }
const StreamInfo&     StreamWorker::GetStreamInfo() const { return streamInfo_; }

// ─────────────────────────────────────────────────────────────────────────────
// § 8  Post-process helpers
// ─────────────────────────────────────────────────────────────────────────────
namespace {

TensorBase SliceBatchTensor(MxBase::Tensor& out, uint32_t batchIndex,
                             uint32_t batchSize) {
    size_t total      = out.GetByteSize();
    size_t frameBytes = total / batchSize;
    size_t offset     = batchIndex * frameBytes;

    auto shapeRaw = out.GetShape();
    shapeRaw[0]   = 1;
    std::vector<uint32_t> shape(shapeRaw.begin(), shapeRaw.end());

    std::vector<uint8_t> buf(frameBytes);
    std::memcpy(buf.data(),
                static_cast<uint8_t*>(out.GetData()) + offset,
                frameBytes);
    MemoryData mem(buf.data(), frameBytes);
    return TensorBase(mem, true, shape, TENSOR_DTYPE_FLOAT32);
}

MxBase::ResizedImageInfo MakeResizeInfo(const MxBase::Image& img,
                                         const ModelConfig&   mcfg) {
    MxBase::ResizedImageInfo info;
    info.widthOriginal  = img.GetOriginalSize().width;
    info.heightOriginal = img.GetOriginalSize().height;
    info.widthResize    = mcfg.resizeWidth;
    info.heightResize   = mcfg.resizeHeight;
    info.resizeType     = MxBase::RESIZER_STRETCHING; // informational only; actual resize is letterbox
    return info;
}

void DrawDetections(const std::vector<ObjectInfo>& objs,
                    cv::Mat& mat,
                    const cv::Scalar& colour,
                    float confThreshold) {
    for (const auto& obj : objs) {
        if (obj.confidence < confThreshold) continue;
        std::ostringstream lbl;
        lbl << obj.className << " "
            << std::fixed << std::setprecision(2) << obj.confidence;
        cv::Rect r(static_cast<int>(obj.x0),
                   static_cast<int>(obj.y0),
                   static_cast<int>(obj.x1 - obj.x0),
                   static_cast<int>(obj.y1 - obj.y0));
        cv::rectangle(mat, r, colour, 2);
        cv::putText(mat, lbl.str(),
                    cv::Point(r.x, std::max(r.y - 5, 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, colour, 2);
    }
}

void DrawTrackerResults(const SimpleTracker& tracker, cv::Mat& mat) {
    // 16 visually distinct BGR colours, cycled by track ID.
    static const cv::Scalar palette[16] = {
        {255,  80,  80}, { 80, 255,  80}, { 80,  80, 255}, {255, 255,  80},
        {255,  80, 255}, { 80, 255, 255}, {255, 160,  80}, {160,  80, 255},
        {  0, 200, 255}, {255, 200,   0}, {  0, 255, 160}, {200,   0, 255},
        {255,   0, 100}, {100, 255,   0}, {  0, 100, 255}, {200, 200, 200},
    };

    for (const auto& t : tracker.tracks()) {
        if (t.misses > 0) continue;   // only draw currently-matched tracks
        const cv::Scalar& col = palette[t.id % 16];

        // Bounding box + label
        cv::Rect r(static_cast<int>(t.bbox.x),
                   static_cast<int>(t.bbox.y),
                   static_cast<int>(t.bbox.width),
                   static_cast<int>(t.bbox.height));
        cv::rectangle(mat, r, col, 2);
        cv::putText(mat, "ID:" + std::to_string(t.id),
                    cv::Point(r.x, std::max(r.y - 5, 15)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, col, 2);

        // Trajectory polyline
        if (t.traj.size() >= 2) {
            for (size_t j = 1; j < t.traj.size(); ++j) {
                cv::line(mat,
                         cv::Point(static_cast<int>(t.traj[j-1].x),
                                   static_cast<int>(t.traj[j-1].y)),
                         cv::Point(static_cast<int>(t.traj[j].x),
                                   static_cast<int>(t.traj[j].y)),
                         col, 2);
            }
        }
    }
}

} // anonymous namespace

// ── GetDetections* — run post-processing without drawing ─────────────────────

APP_ERROR GetDetectionsYolov3(Yolov3PostProcessNew&         pp,
                               MxBase::Image&               decodedImage,
                               std::vector<MxBase::Tensor>& outputs,
                               uint32_t                     batchIndex,
                               const ModelConfig&           mcfg,
                               std::vector<ObjectInfo>&     out)
{
    std::vector<TensorBase> tensors;
    tensors.reserve(outputs.size());
    for (auto& o : outputs)
        tensors.emplace_back(SliceBatchTensor(o, batchIndex, mcfg.batchSize));

    auto info = MakeResizeInfo(decodedImage, mcfg);
    std::vector<std::vector<ObjectInfo>> objInfos;
    APP_ERROR ret = pp.Process(tensors, objInfos, {info});
    if (ret != APP_ERR_OK) return ret;
    if (!objInfos.empty()) out = std::move(objInfos[0]);
    return APP_ERR_OK;
}

APP_ERROR GetDetectionsYolov11(MxBase::Yolov11PostProcess&  pp,
                                MxBase::Image&              decodedImage,
                                std::vector<MxBase::Tensor>& outputs,
                                uint32_t                    batchIndex,
                                const ModelConfig&          mcfg,
                                std::vector<ObjectInfo>&    out)
{
    std::vector<TensorBase> tensors;
    tensors.reserve(outputs.size());
    for (auto& o : outputs) {
        auto   shape      = o.GetShape();
        size_t rows       = shape.size() >= 2 ? shape[1] : 1;
        size_t cols       = shape.size() >= 3 ? shape[2] : 1;
        size_t frameBytes = rows * cols * sizeof(float);
        size_t offset     = static_cast<size_t>(batchIndex) * frameBytes;

        std::vector<uint32_t> newShape = {
            1,
            static_cast<uint32_t>(rows),
            static_cast<uint32_t>(cols)
        };
        std::vector<uint8_t> buf(frameBytes);
        std::memcpy(buf.data(),
                    static_cast<uint8_t*>(o.GetData()) + offset,
                    frameBytes);
        MemoryData mem(buf.data(), frameBytes);
        tensors.emplace_back(mem, true, newShape, TENSOR_DTYPE_FLOAT32);
    }

    auto info = MakeResizeInfo(decodedImage, mcfg);
    std::vector<std::vector<ObjectInfo>> objInfos;
    APP_ERROR ret = pp.Process(tensors, objInfos, {info});
    if (ret != APP_ERR_OK) return ret;
    if (!objInfos.empty()) out = std::move(objInfos[0]);
    return APP_ERR_OK;
}

APP_ERROR RunPostProcessYolov3(Yolov3PostProcessNew&         pp,
                                MxBase::Image&               decodedImage,
                                std::vector<MxBase::Tensor>& outputs,
                                uint32_t                     batchIndex,
                                cv::Mat&                     resultMat,
                                const ModelConfig&           mcfg)
{
    std::vector<ObjectInfo> dets;
    APP_ERROR ret = GetDetectionsYolov3(pp, decodedImage, outputs,
                                        batchIndex, mcfg, dets);
    if (ret != APP_ERR_OK) return ret;
    DrawDetections(dets, resultMat, cv::Scalar(255, 0, 0), mcfg.confidenceThreshold);
    return APP_ERR_OK;
}

APP_ERROR RunPostProcessYolov11(MxBase::Yolov11PostProcess&   pp,
                                 MxBase::Image&               decodedImage,
                                 std::vector<MxBase::Tensor>& outputs,
                                 uint32_t                     batchIndex,
                                 cv::Mat&                     resultMat,
                                 const ModelConfig&           mcfg)
{
    std::vector<ObjectInfo> dets;
    APP_ERROR ret = GetDetectionsYolov11(pp, decodedImage, outputs,
                                         batchIndex, mcfg, dets);
    if (ret != APP_ERR_OK) return ret;
    DrawDetections(dets, resultMat, cv::Scalar(0, 255, 0), mcfg.confidenceThreshold);
    return APP_ERR_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// § 9  ReadyFrame  — output of per-worker resize threads
// ─────────────────────────────────────────────────────────────────────────────
struct ReadyFrame {
    MxBase::Tensor tensor;   // always host-resident (ToHost called before push)
    DecodedFrame   df;
    uint32_t       workerId;
    uint32_t       deviceId;
    cv::Mat        frameMat; // BGR display frame captured while device memory is still valid
};

// ─────────────────────────────────────────────────────────────────────────────
// § 10  RunPipeline
// ─────────────────────────────────────────────────────────────────────────────
PipelineResult RunPipeline(const std::vector<std::string>& urls,
                            bool isFile,
                            const ModelConfig& mcfg,
                            uint32_t   runSeconds,
                            CsvLog*    csv, int channels,
                            StreamInfo srcInfo)
{
    std::cout << "[DBG] RunPipeline entered, urls=" << urls.size()
              << " NUM_DEVICES=" << cfg::NUM_DEVICES << "\n";

    const uint32_t numDevices = cfg::NUM_DEVICES;
    const uint32_t totalSlots = numDevices * cfg::MAX_VDEC_CHANNELS;
    const uint32_t numLogical = static_cast<uint32_t>(urls.size());

    // ── Slot assignment (round-robin across physical DVPP channels) ───────────
    auto slotForLogical  = [&](uint32_t i) { return i % totalSlots; };
    auto deviceForSlot   = [&](uint32_t s) { return s / cfg::MAX_VDEC_CHANNELS; };
    auto dvppChanForSlot = [&](uint32_t s) { return s % cfg::MAX_VDEC_CHANNELS; };

    const uint32_t streamsPerSlot =
        (numLogical + totalSlots - 1) / totalSlots;

    // ── Raw queue depth ───────────────────────────────────────────────────────
    // Distribute a total budget of 4×BATCH_SIZE raw frames evenly across all
    // workers.  Floor at 4 so a single-channel run still has headroom.
    //
    // Old value was a fixed batchSize*4 regardless of channel count.  At 32
    // channels / batch=16 that gave 64 frames × 32 workers = 2048 total — the
    // resize threads raced far ahead, ready queue pinned at ceiling, and the
    // reported lag was 2300 ms (pure queue depth, not real inference latency).
    {
        const uint32_t totalBudget = mcfg.batchSize * 4;
        const uint32_t perWorker   = std::max(4u, totalBudget / numLogical);
        cfg::MAX_QUEUE = perWorker;
        std::cout << "[INFO] channels=" << numLogical
                  << "  raw_queue/worker=" << cfg::MAX_QUEUE
                  << "  total_raw_cap="    << cfg::MAX_QUEUE * numLogical
                  << "\n";
    }

    // ── Ready queue ceiling ───────────────────────────────────────────────────
    // Tight absolute cap keeps back-pressure on resize threads so tensors never
    // age in the queue long enough for ACL device-memory to be reclaimed.
    const size_t READY_QUEUE_MAX = static_cast<size_t>(cfg::BATCH_SIZE) * 4;

    std::cout << "[INFO] mux_ratio="  << streamsPerSlot
              << "  ready_cap="       << READY_QUEUE_MAX
              << "  decode_skip="     << cfg::DECODE_SKIP_INTERVAL << "\n";

    if (streamsPerSlot > 1)
        std::cout << "[MUX] " << numLogical << " logical streams → "
                  << totalSlots << " DVPP slots ("
                  << streamsPerSlot << " streams/slot) across "
                  << numDevices << " device(s).\n";
    else
        std::cout << "[ALLOC] " << numLogical << " stream(s) on "
                  << numDevices << " device(s).\n";

    // ── Model / ImageProcessor per device ────────────────────────────────────
    std::set<uint32_t> usedDevices;
    for (uint32_t i = 0; i < numLogical; ++i)
        usedDevices.insert(deviceForSlot(slotForLogical(i)));

    // MxBase::Model must NOT be heap-allocated via unique_ptr/shared_ptr.
    // std::deque gives stable addresses without reallocation.
    std::deque<MxBase::Model>          modelStore;
    std::deque<MxBase::ImageProcessor> imgProcStore;
    std::map<uint32_t, MxBase::Model*>          models;
    std::map<uint32_t, MxBase::ImageProcessor*> imgProcs;

    for (uint32_t d : usedDevices) {
        std::string mp = mcfg.modelPath;
        modelStore.emplace_back(mp, static_cast<int32_t>(d));
        imgProcStore.emplace_back(d);
        models[d]   = &modelStore.back();
        imgProcs[d] = &imgProcStore.back();
        std::cout << "[DEVICE " << d << "] Model and ImageProcessor ready.\n";
    }

    // ── Post-processor ────────────────────────────────────────────────────────
    std::map<std::string, std::string> ppCfg = {
        {"postProcessConfigPath", mcfg.configPath},
        {"labelPath",             mcfg.labelPath},
        // Inject thresholds from config.json — overrides values in .cfg
        {"SCORE_THRESH",          std::to_string(mcfg.confidenceThreshold)},
        {"OBJECTNESS_THRESH",     std::to_string(mcfg.confidenceThreshold)},
        {"IOU_THRESH",            std::to_string(mcfg.nmsIouThreshold)}
    };

    bool useV11 = (mcfg.modelType == ModelType::YOLOV11);
    Yolov3PostProcessNew       yolov3PP;
    MxBase::Yolov11PostProcess yolov11PP;

    if (useV11) {
        if (yolov11PP.Init(ppCfg) != APP_ERR_OK) {
            std::cerr << "Failed to init Yolov11PostProcess\n";
            return {};
        }
    } else {
        if (yolov3PP.Init(ppCfg) != APP_ERR_OK) {
            std::cerr << "Failed to init Yolov3PostProcessNew\n";
            return {};
        }
    }

    auto DispatchGetDets = [&](MxBase::Image&               img,
                                std::vector<MxBase::Tensor>& outs,
                                uint32_t                     batchIdx,
                                std::vector<ObjectInfo>&     dets) -> APP_ERROR {
        return useV11
            ? GetDetectionsYolov11(yolov11PP, img, outs, batchIdx, mcfg, dets)
            : GetDetectionsYolov3 (yolov3PP,  img, outs, batchIdx, mcfg, dets);
    };

    const cv::Scalar detColour = useV11 ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);

    if (cfg::WRITE_FRAMES) {
        std::filesystem::remove_all("out");
        MkdirP("out");
    }
    if (cfg::WRITE_TRACK_FRAMES) {
        std::filesystem::remove_all("out_track");
        MkdirP("out_track");
    }

    // ── Stream geometry ───────────────────────────────────────────────────────
    // Use the StreamInfo supplied by the caller (probed once before the sweep
    // loop) to avoid re-opening the source on every step.  Only fall back to
    // probing here if the caller passed a zeroed struct (single RunInfer call
    // with no pre-probe, or direct API use without srcInfo).
    StreamInfo probed = srcInfo;
    if (probed.width == 0 || probed.height == 0) {
        probed = ProbeStream(urls[0], isFile);
        if (probed.width == 0 || probed.height == 0)
            std::cerr << "[WARN] Stream probe failed — DVPP will default to 1920×1080\n";
    }

    // ── Workers ───────────────────────────────────────────────────────────────
    std::map<uint32_t, StreamWorker*>          slotOwner;
    std::vector<std::unique_ptr<StreamWorker>> workers;
    workers.reserve(urls.size());

    for (uint32_t i = 0; i < numLogical; ++i) {
        uint32_t slot     = slotForLogical(i);
        uint32_t devId    = deviceForSlot(slot);
        uint32_t dvppChan = dvppChanForSlot(slot);

        auto w = std::make_unique<StreamWorker>(i, urls[i], isFile, devId,
                                                dvppChan);
        if (slotOwner.count(slot) == 0) {
            if (w->Init(nullptr, probed) != APP_ERR_OK) {
                std::cerr << "Worker " << i << " (slot " << slot
                          << ") init failed\n";
                continue;
            }
            slotOwner[slot] = w.get();
        } else {
            StreamWorker*   primary = slotOwner[slot];
            MuxCallbackCtx* ctx     = primary->GetMuxCtx();
            if (w->Init(ctx) != APP_ERR_OK) {
                std::cerr << "Worker " << i << " (mux slot " << slot
                          << ") init failed\n";
                continue;
            }
            w->SetSharedDecoder(primary->GetDecoder());
        }
        w->Start();
        workers.push_back(std::move(w));
    }

    // One tracker per logical stream (indexed by workerId == logical stream index).
    std::vector<SimpleTracker> trackers(numLogical);

    // Monotonic save counter per worker — used for output filenames instead of
    // the DVPP frameId.  The DVPP fires its callback in decode order (DTS), but
    // H.264/H.265 B-frames mean DTS ≠ PTS (display order).  collectAndProcess
    // is called sequentially one batch at a time, so this counter always
    // reflects true chronological output order.
    std::vector<uint64_t> saveCounters(numLogical, 0);
    for (auto& tr : trackers) {
        tr.iouThresh = mcfg.trackerIouThresh;
        tr.maxMisses = mcfg.trackerMaxMisses;
        tr.maxTraj   = mcfg.trackerMaxTraj;
    }

    // ── Per-worker resize thread pool ─────────────────────────────────────────
    // Each stream gets a dedicated resize thread.  After resize, the tensor is
    // immediately moved to host RAM (ToHost) so it is safe to copy at any
    // later point — the ACL device-side allocation cannot be reclaimed after
    // the tensor is host-resident.
    std::deque<ReadyFrame>  readyQueue;
    std::mutex              readyMu;
    std::condition_variable readyCv;
    std::atomic<bool>       readyStop{false};

    std::vector<std::thread> resizeThreads;
    resizeThreads.reserve(workers.size());

    for (size_t wi = 0; wi < workers.size(); ++wi) {
        StreamWorker* w     = workers[wi].get();
        uint32_t      wId   = w->GetId();
        uint32_t      wSlot = slotForLogical(wId < numLogical ? wId : 0);
        uint32_t      wDev  = deviceForSlot(wSlot);

        resizeThreads.emplace_back([&, w, wId, wDev]() {
            while (!readyStop.load(std::memory_order_relaxed)) {
                DecodedFrame df;
                if (!w->PopFrame(df)) {
                    if (!w->IsActive()) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                // Letterbox resize — preserves aspect ratio by:
                //   1. Resize to the largest size that fits inside model dims
                //   2. Pad the remainder with grey (114,114,114) symmetrically
                // This matches YOLO training preprocessing and avoids the
                // aspect-ratio distortion of plain stretch resize.
                auto origSz = df.image.GetOriginalSize();
                float gain  = std::min(
                    static_cast<float>(mcfg.resizeWidth)  / origSz.width,
                    static_cast<float>(mcfg.resizeHeight) / origSz.height);
                // Align to 2 — required for YUV420SP chroma subsampling
                uint32_t newW = static_cast<uint32_t>(origSz.width  * gain) & ~1u;
                uint32_t newH = static_cast<uint32_t>(origSz.height * gain) & ~1u;
                uint32_t padLeft   = ((mcfg.resizeWidth  - newW) / 2) & ~1u;
                uint32_t padTop    = ((mcfg.resizeHeight - newH) / 2) & ~1u;
                uint32_t padRight  = mcfg.resizeWidth  - newW - padLeft;
                uint32_t padBottom = mcfg.resizeHeight - newH - padTop;

                MxBase::Image scaled;
                imgProcs[wDev]->Resize(df.image, MxBase::Size(newW, newH), scaled);
                MxBase::Image resized;
                MxBase::Dim padDim;
                padDim.top    = padTop;
                padDim.bottom = padBottom;
                padDim.left   = padLeft;
                padDim.right  = padRight;
                MxBase::Color padColor;
                padColor.channel_zero = 114;
                padColor.channel_one = 114;
                padColor.channel_two = 114;
                imgProcs[wDev]->Padding(scaled, padDim, padColor,
                                        MxBase::BorderType::BORDER_CONSTANT,
                                        resized);

                // Move to host RAM immediately — makes the tensor copy-safe and
                // immune to ACL device-memory reclaim while sitting in the queue.
                MxBase::Tensor t = resized.ConvertToTensor();
                t.ToHost();

                // Encode display frame NOW — df.image device memory is still valid
                // here (we just decoded it).  By the time collectAndProcess runs,
                // DVPP may have recycled that buffer, so we must capture it here.
                cv::Mat frameMat;
                if (cfg::WRITE_FRAMES || cfg::WRITE_TRACK_FRAMES) {
                    std::string dir = "out/" + std::to_string(wId);
                    MkdirP(dir);
                    std::string tmp = dir + "/.enc_" + std::to_string(wId) + ".jpg";
                    imgProcs[wDev]->Encode(df.image, tmp);
                    frameMat = cv::imread(tmp);
                    std::remove(tmp.c_str());
                    if (frameMat.empty()) {
                        auto sz = df.image.GetOriginalSize();
                        if (sz.width > 0 && sz.height > 0)
                            frameMat = cv::Mat(static_cast<int>(sz.height),
                                               static_cast<int>(sz.width),
                                               CV_8UC3, cv::Scalar(0, 0, 0));
                    }
                }

                // Push into the shared ready queue; block on back-pressure
                {
                    std::unique_lock<std::mutex> lk(readyMu);
                    readyCv.wait(lk, [&]{
                        return readyQueue.size() < READY_QUEUE_MAX
                            || readyStop.load(std::memory_order_relaxed);
                    });
                    if (readyStop.load(std::memory_order_relaxed)) break;
                    readyQueue.push_back({std::move(t), std::move(df),
                                          wId, wDev, std::move(frameMat)});
                }
                readyCv.notify_one();
            }
        });
    }

    // ── Batch timeout ─────────────────────────────────────────────────────────
    // Initial value derived from source FPS; will be adapted every batch based
    // on the measured NPU inference duration (rolling average over last 16
    // batches).  Clamped to [10 ms, 150 ms] at all times.
    const double sourceFps = (probed.fps > 0.0) ? probed.fps : 25.0;
    {
        const double msPerBatch = (1000.0 / sourceFps)
                                  * static_cast<double>(cfg::BATCH_SIZE)
                                  / static_cast<double>(channels);
        uint32_t initMs = std::max(10u, std::min(150u,
                              static_cast<uint32_t>(msPerBatch * 1.5)));
        std::cout << "[INFO] batchTimeout initial=" << initMs << "ms"
                  << "  (source_fps=" << std::fixed << std::setprecision(1)
                  << sourceFps << ")\n";
    }
    // Runtime-mutable timeout; updated after every inferred batch.
    std::chrono::milliseconds batchTimeout{
        std::max(10u, std::min(150u,
            static_cast<uint32_t>(
                (1000.0 / sourceFps)
                * static_cast<double>(cfg::BATCH_SIZE)
                / static_cast<double>(channels) * 1.5)))};

    // Rolling window for adaptive timeout (last 16 NPU batch durations).
    constexpr size_t INFER_WINDOW = 16;
    std::vector<double> inferMsWindow;
    inferMsWindow.reserve(INFER_WINDOW + 1);
    double inferMsSum = 0.0;

    // ── Triple-stage inference loop ───────────────────────────────────────────
    //
    // The key insight from profiling: the old double-buffer loop ran
    // batch-assembly AFTER inferFuture.get(), so they were serial:
    //
    //   OLD:  [get/post-process 35ms] → [assemble 40ms] → [launch] → repeat
    //         NPU idle during assemble ↑
    //
    // New order moves assembly BEFORE get(), so it overlaps with NPU:
    //
    //   NEW:  [assemble batch N+1 (40ms)]   ← runs while NPU does batch N
    //         [get batch N + post-process]  ← NPU likely already done
    //         [launch batch N+1]
    //         → repeat
    //
    // This eliminates the NPU idle gap between batches.  The assembler uses
    // batchTimeout derived from the rolling NPU average so it doesn't wait
    // longer than the NPU actually takes.
    //
    // State carried across iterations:
    //   pendingFuture   — in-flight NPU batch
    //   pendingRecord   — BatchRecord whose outputs pendingFuture will produce
    //   nextRec         — assembled batch ready to launch next iteration

    struct BatchRecord {
        std::vector<ReadyFrame> slots;
        size_t                  realCount{0};
        uint32_t                batchDev{0};
    };

    // Helper: drain readyQueue into rec up to BATCH_SIZE within deadline.
    auto assembleBatch = [&](BatchRecord& rec) {
        rec.slots.reserve(cfg::BATCH_SIZE);
        auto batchDeadline = Clock::now() + batchTimeout;
        while (rec.slots.size() < cfg::BATCH_SIZE
               && Clock::now() < batchDeadline)
        {
            std::unique_lock<std::mutex> lk(readyMu);
            readyCv.wait_until(lk, batchDeadline,
                [&]{ return !readyQueue.empty(); });
            while (!readyQueue.empty()
                   && rec.slots.size() < cfg::BATCH_SIZE) {
                rec.slots.push_back(std::move(readyQueue.front()));
                readyQueue.pop_front();
            }
            lk.unlock();
            readyCv.notify_all();
        }
        rec.realCount = rec.slots.size();
        if (!rec.slots.empty())
            rec.batchDev = rec.slots[0].deviceId;
    };

    // Helper: pad rec to exactly BATCH_SIZE (deep-copies host tensor from slot 0).
    auto padBatch = [&](BatchRecord& rec) {
        while (rec.slots.size() < cfg::BATCH_SIZE) {
            ReadyFrame pad;
            pad.tensor   = rec.slots[0].tensor;
            pad.workerId = rec.slots[0].workerId;
            pad.deviceId = rec.batchDev;
            rec.slots.push_back(std::move(pad));
        }
    };

    // Helper: launch async NPU inference for rec; returns future + launch time.
    auto launchInfer = [&](BatchRecord& rec)
        -> std::pair<std::future<std::vector<MxBase::Tensor>>, Clock::time_point>
    {
        std::vector<MxBase::Tensor> inferTensors;
        inferTensors.reserve(rec.slots.size());
        for (const auto& rf : rec.slots)
            inferTensors.push_back(rf.tensor);

        uint32_t dev = rec.batchDev;
        auto t0 = Clock::now();
        auto fut = std::async(std::launch::async,
            [&models,
             inferTensors = std::move(inferTensors),
             dev]() mutable -> std::vector<MxBase::Tensor>
            {
                MxBase::Tensor batched;
                if (BatchConcat(inferTensors, batched) != APP_ERR_OK)
                    return {};
                batched.ToDevice(static_cast<int32_t>(dev));
                std::vector<MxBase::Tensor> inferInputs = {batched};
                auto outputs = models[dev]->Infer(inferInputs);
                for (auto& o : outputs) o.ToHost();
                return outputs;
            });
        return {std::move(fut), t0};
    };

    // Declared here so collectAndProcess lambda can capture them by reference.
    uint64_t inferCount  = 0;
    double   lastInferMs = 0.0;

    // Helper: collect NPU outputs and run post-processing + stats update.
    auto collectAndProcess =
        [&](std::future<std::vector<MxBase::Tensor>>& fut,
            BatchRecord&                               rec,
            Clock::time_point                          launchTime)
    {
        auto outputs  = fut.get();
        auto now      = Clock::now();

        // Measure NPU duration; update rolling window; adapt batchTimeout.
        double measuredMs = std::chrono::duration_cast<
                                std::chrono::microseconds>(
                                now - launchTime).count() / 1000.0;
        inferMsWindow.push_back(measuredMs);
        inferMsSum += measuredMs;
        if (inferMsWindow.size() > INFER_WINDOW) {
            inferMsSum -= inferMsWindow.front();
            inferMsWindow.erase(inferMsWindow.begin());
        }
        double avgMs = inferMsSum / static_cast<double>(inferMsWindow.size());
        lastInferMs  = avgMs;
        uint32_t newTimeout = static_cast<uint32_t>(avgMs * 1.2);
        newTimeout = std::max(10u, std::min(150u, newTimeout));
        batchTimeout = std::chrono::milliseconds(newTimeout);

        if (outputs.empty()) return;

        for (size_t i = 0; i < rec.realCount; ++i) {
            auto& rf    = rec.slots[i];
            auto  lagUs = std::chrono::duration_cast<
                              std::chrono::microseconds>(
                              now - rf.df.decodeTime).count();
            auto& st = workers[rf.workerId]->GetStats();
            st.totalLagUs     += static_cast<uint64_t>(lagUs);
            st.lagSamples     += 1;
            st.framesInferred += 1;

            if (cfg::WRITE_FRAMES || cfg::WRITE_TRACK_FRAMES) {
                uint64_t    saveIdx   = saveCounters[rf.workerId]++;
                std::string frameStem = "/f" + std::to_string(saveIdx) + ".jpg";

                // On the very first frame, write meta.txt so make_video.sh can
                // compute the correct playback fps without relying on frame ID gaps.
                if (saveIdx == 0) {
                    auto writeMeta = [&](const std::string& dir) {
                        std::ofstream m(dir + "/meta.txt");
                        m << "skip=" << cfg::DECODE_SKIP_INTERVAL << "\n";
                    };
                    if (cfg::WRITE_FRAMES)       writeMeta("out");
                    if (cfg::WRITE_TRACK_FRAMES) writeMeta("out_track");
                }

                // rf.frameMat was captured in the resize thread while df.image
                // device memory was still valid.  Use it directly here.
                cv::Mat baseMat = rf.frameMat;

                // Get detections.
                std::vector<ObjectInfo> dets;
                DispatchGetDets(rf.df.image, outputs,
                                static_cast<uint32_t>(i), dets);

                // Class filter — remove detections not in the allow-list.
                if (!mcfg.classFilter.empty()) {
                    dets.erase(
                        std::remove_if(dets.begin(), dets.end(),
                            [&](const ObjectInfo& o) {
                                return mcfg.classFilter.find(o.className)
                                       == mcfg.classFilter.end();
                            }),
                        dets.end());
                }

                // Option 1 – detection-only frame (--frames).
                if (cfg::WRITE_FRAMES) {
                    cv::Mat detMat = baseMat.clone();
                    DrawDetections(dets, detMat, detColour,
                                   mcfg.confidenceThreshold);
                    cv::imwrite("out" + frameStem, detMat);
                }

                // Option 2 – tracker frame with IDs and trajectories (--track-frames).
                if (cfg::WRITE_TRACK_FRAMES) {
                    std::vector<cv::Rect2f> rects;
                    rects.reserve(dets.size());
                    for (const auto& o : dets)
                        if (o.confidence >= mcfg.confidenceThreshold)
                            rects.emplace_back(o.x0, o.y0,
                                               o.x1 - o.x0, o.y1 - o.y0);
                    trackers[rf.workerId].update(rects);

                    cv::Mat trackMat = baseMat.clone();
                    DrawTrackerResults(trackers[rf.workerId], trackMat);
                    cv::imwrite("out_track" + frameStem, trackMat);
                }
            }
        }
        inferCount += rec.realCount;
    };

    std::future<std::vector<MxBase::Tensor>> pendingFuture;
    BatchRecord                              pendingRecord;
    Clock::time_point                        pendingLaunch;

    double   fillSum     = 0.0;
    uint32_t fillSamples = 0;

    auto pipeStart = Clock::now();
    auto lastLog   = pipeStart;
    auto deadline  = runSeconds
                     ? (pipeStart + std::chrono::seconds(runSeconds))
                     : Clock::time_point::max();
    double lastFps = 0.0;

    auto anyActive = [&]() -> bool {
        for (auto& w : workers)
            if (w->IsActive()) return true;
        return false;
    };

    // Prime the pipeline: assemble and launch the very first batch before
    // entering the main loop so the NPU is never idle on iteration 1.
    {
        BatchRecord first;
        assembleBatch(first);
        if (!first.slots.empty()) {
            fillSum += static_cast<double>(first.realCount) / cfg::BATCH_SIZE;
            ++fillSamples;
            padBatch(first);
            pendingRecord = std::move(first);
            auto [fut, t0] = launchInfer(pendingRecord);
            pendingFuture  = std::move(fut);
            pendingLaunch  = t0;
        }
    }

    while (Clock::now() < deadline) {

        // ── Step 1: assemble batch N+1 while NPU runs batch N ────────────────
        // This is the core change: assembly happens HERE, concurrently with
        // the in-flight pendingFuture, not after we block on it.
        BatchRecord nextRec;
        assembleBatch(nextRec);
        
        if (nextRec.slots.empty() && !anyActive()) {
            // No more frames and all streams done — drain last batch and exit.
            if (pendingFuture.valid())
                collectAndProcess(pendingFuture, pendingRecord, pendingLaunch);   
            break;
        }
        

        // ── Step 2: collect NPU outputs for batch N (likely already done) ────
        if (pendingFuture.valid())
            collectAndProcess(pendingFuture, pendingRecord, pendingLaunch);
        
        

        if (nextRec.slots.empty()) continue; // spurious wakeup, try again

        // ── Step 3: record fill, pad, launch batch N+1 ───────────────────────
        fillSum += static_cast<double>(nextRec.realCount) / cfg::BATCH_SIZE;
        ++fillSamples;
        padBatch(nextRec);
        pendingRecord = std::move(nextRec);
        auto [fut, t0] = launchInfer(pendingRecord);
        pendingFuture  = std::move(fut);
        pendingLaunch  = t0;

        // ── Periodic console + CSV log ────────────────────────────────────────
        auto now        = Clock::now();
        auto msSinceLog = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now - lastLog).count();
        if (msSinceLog >= static_cast<int64_t>(cfg::LOG_INTERVAL_MS)) {
            double secs    = std::chrono::duration_cast<
                                 std::chrono::milliseconds>(
                                 now - pipeStart).count() / 1000.0;
            double fps     = secs > 0 ? inferCount / secs : 0;
            lastFps        = fps;
            double avgFill = fillSamples ? fillSum / fillSamples : 0;

            uint64_t tLag = 0, tSamples = 0;
            uint64_t totalDecoded = 0, totalResized = 0;
            for (auto& w : workers) {
                tLag         += w->GetStats().totalLagUs.load();
                tSamples     += w->GetStats().lagSamples.load();
                totalDecoded += w->GetStats().framesDecoded.load();
                totalResized += w->GetStats().framesInferred.load();
            }
            double lagMs = tSamples ? (tLag / 1000.0 / tSamples) : 0;

            size_t rqSize = 0;
            { std::lock_guard<std::mutex> lk(readyMu); rqSize = readyQueue.size(); }

            std::ostringstream qd;
            qd << "ready=" << rqSize << " raw=[ ";
            for (auto& w : workers) qd << w->QueueDepth() << " ";
            qd << "]";

            std::cout << "[" << Timestamp() << "] "
                      << "channels="   << channels
                      << "  inferred=" << inferCount
                      << "  decoded="  << totalDecoded
                      << "  resized="  << totalResized
                      << "  fps="      << std::fixed << std::setprecision(1)
                                       << fps
                      << "  fpsPerCh=" << std::fixed << std::setprecision(1)
                                       << (fps / channels)
                      << "  fill="     << std::fixed << std::setprecision(2)
                                       << avgFill
                      << "  lag="      << std::fixed << std::setprecision(1)
                                       << lagMs << "ms"
                      << "  inferMs="  << std::fixed << std::setprecision(1)
                                       << lastInferMs << "ms"
                      << "  timeout="  << batchTimeout.count() << "ms"
                      << "  skip="     << cfg::DECODE_SKIP_INTERVAL
                      << "  q=[ "      << qd.str() << "]\n";

            if (csv)
                csv->Write(static_cast<int>(channels), inferCount, fps,
                           avgFill, cfg::DECODE_SKIP_INTERVAL, lagMs,
                           lastInferMs);
            lastLog = now;
        }
    }

    // Drain the last in-flight batch
    if (pendingFuture.valid())
        collectAndProcess(pendingFuture, pendingRecord, pendingLaunch);

    // ── Teardown ──────────────────────────────────────────────────────────────
    readyStop.store(true, std::memory_order_relaxed);
    readyCv.notify_all();
    for (auto& t : resizeThreads) if (t.joinable()) t.join();
    for (auto& w : workers) w->Stop();
    if (useV11) yolov11PP.DeInit(); else yolov3PP.DeInit();

    PipelineResult res;
    res.totalInferred = inferCount;
    res.numChannels   = static_cast<uint32_t>(channels);
    res.avgFps        = lastFps;
    res.fpsPerChannel = res.numChannels > 0
                        ? res.avgFps / res.numChannels : 0;
    res.avgFill       = fillSamples ? fillSum / fillSamples : 0;

    uint64_t tLag = 0, tSamples = 0;
    for (auto& w : workers) {
        tLag     += w->GetStats().totalLagUs.load();
        tSamples += w->GetStats().lagSamples.load();
    }
    res.avgLagMs = tSamples ? (tLag / 1000.0 / tSamples) : 0;
    return res;
}