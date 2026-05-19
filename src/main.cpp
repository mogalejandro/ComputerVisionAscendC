/**
 * main.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * Three modes in one binary:
 *
 *   --video  <file.mp4>  <channels>           Run inference on a video file
 *   --rtsp   <rtsp_url>  <channels>           Run inference on RTSP stream(s)
 *   --sweep  --video|--rtsp <src> <start> <end> <step>
 *                                             Find max real-time channel count
 *
 * Common options (all modes):
 *   --skip      <N>                  Decode 1 in every N+1 frames (default: 0).
 *
 * Video / RTSP only:
 *   --frames                       Write annotated JPEGs to ./out/
 *   --track-frames                 Write tracker JPEGs to ./out_track/
 *   --duration  <N>                Stop after N seconds (default: 0 = run until stream ends)
 *   --make-video                   Automatically generate MP4 video from frames after inference
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include "detection_utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

// ─── Sweep knobs ──────────────────────────────────────────────────────────────
namespace sweep {
    constexpr uint32_t MEASURE_SEC     = 90;
    constexpr double   DEFAULT_SRC_FPS = 25.0;
}

// ─── CLI helpers ──────────────────────────────────────────────────────────────
static bool HasFlag(int argc, char* argv[], const std::string& f) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == f) return true;
    return false;
}

static std::string FlagVal(int argc, char* argv[], const std::string& f) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == f) return argv[i + 1];
    return "";
}

static void PrintUsage(const char* prog) {
    std::cout <<
        "\nUsage:\n"
        "  Video  : " << prog << " --video <file.mp4> <channels> [options]\n"
        "  RTSP   : " << prog << " --rtsp  <rtsp_url> <channels> [options]\n"
        "  Sweep  : " << prog << " --sweep --video <file.mp4> <start> <end> <step> [options]\n"
        "           " << prog << " --sweep --rtsp  <rtsp_url>  <start> <end> <step> [options]\n"
        "\nCommon options:\n"
        "  --skip      <N>                 Decode 1 in every N+1 frames (default: 0).\n"
        "\nVideo / RTSP options:\n"
        "  --frames                       Write detection JPEGs to ./out/\n"
        "  --track-frames                 Write tracker JPEGs (IDs + trajectories) to ./out_track/\n"
        "  --duration  <N>                Stop after N seconds (default: 0 = run until stream ends)\n"
        "  --make-video                   Automatically generate MP4 video from frames after inference\n"
        "\nExamples:\n"
        "  " << prog << " --video test.mp4 1 --track-frames --make-video\n"
        "  " << prog << " --video test.mp4 8\n"
        "  " << prog << " --rtsp rtsp://cam/stream 16\n"
        "  " << prog << " --sweep --video test.mp4 1 32 1\n"
        "  " << prog << " --sweep --rtsp rtsp://cam/stream 1 64 4\n\n";
}

// ─── Video Generation Helper ─────────────────────────────────────────────────
static void GenerateVideoFromFrames(bool isTrackFrames, int fpsOverride = 0) {
    std::string framesDir = isTrackFrames ? "out_track" : "out";
    
    // Verificar si el directorio existe y tiene frames
    if (!fs::exists(framesDir)) {
        std::cout << "[VIDEO] Directory '" << framesDir << "' not found, skipping video generation." << std::endl;
        return;
    }
    
    // Contar frames
    int frameCount = 0;
    for (const auto& entry : fs::directory_iterator(framesDir)) {
        if (entry.path().extension() == ".jpg") {
            frameCount++;
        }
    }
    
    if (frameCount == 0) {
        std::cout << "[VIDEO] No frames found in '" << framesDir << "', skipping video generation." << std::endl;
        return;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "🎬 GENERATING VIDEO FROM FRAMES" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "📁 Frames directory: " << framesDir << std::endl;
    std::cout << "📸 Frames count: " << frameCount << std::endl;
    
    // Construir comando make_video.sh
    std::string cmd = "bash make_video.sh";
    
    if (isTrackFrames) {
        cmd += " --track";
    } else {
        cmd += " --det";
    }
    
    if (fpsOverride > 0) {
        cmd += " --fps " + std::to_string(fpsOverride);
    }
    
    std::cout << "🚀 Executing: " << cmd << std::endl;
    
    int ret = system(cmd.c_str());
    
    if (ret == 0) {
        std::cout << "✅ Video generated successfully!" << std::endl;
        
        // Mostrar el archivo generado
        int idx = 0;
        std::string outputFile;
        while (true) {
            std::string testFile = (isTrackFrames ? "track_" : "det_") + std::to_string(idx) + ".mp4";
            if (fs::exists(testFile)) {
                outputFile = testFile;
                idx++;
            } else {
                break;
            }
        }
        if (!outputFile.empty()) {
            auto size = fs::file_size(outputFile) / (1024.0 * 1024.0);
            std::cout << "📹 Output video: " << outputFile << " (" << std::fixed << std::setprecision(1) << size << " MB)" << std::endl;
        }
    } else {
        std::cerr << "[VIDEO] Failed to generate video (error code: " << ret << ")" << std::endl;
        std::cerr << "[VIDEO] Make sure ffmpeg is installed: sudo apt install ffmpeg" << std::endl;
    }
    std::cout << "========================================\n" << std::endl;
}

// ─── Mode: video / rtsp ───────────────────────────────────────────────────────
static int RunInfer(const std::string& source, bool isFile,
                    int channels, const ModelConfig& mcfg,
                    uint32_t durationSec = 0,
                    StreamInfo srcInfo = {})
{
    if (channels <= 0) { std::cerr << "channels must be > 0\n"; return -1; }

    std::cout << "[" << Timestamp() << "] "
              << (isFile ? "Video" : "RTSP") << " mode: " << source
              << "  channels=" << channels
              << "  skip=" << cfg::DECODE_SKIP_INTERVAL;
    if (durationSec > 0)
        std::cout << "  duration=" << durationSec << "s";
    std::cout << "\n";

    std::vector<std::string> urls(static_cast<size_t>(channels), source);
    CsvLog csv("infer_results.csv");
    RunPipeline(urls, isFile, mcfg, durationSec, &csv, channels, srcInfo);
    return 0;
}

// ─── Mode: sweep ─────────────────────────────────────────────────────────────
static int RunSweep(const std::string& source, bool isFile,
                    int chanStart, int chanEnd, int chanStep,
                    double sourceFps, const ModelConfig& mcfg,
                    StreamInfo srcInfo)
{
    if (chanStart <= 0 || chanEnd < chanStart || chanStep <= 0) {
        std::cerr << "Invalid sweep range: start=" << chanStart
                  << " end=" << chanEnd << " step=" << chanStep << "\n";
        return -1;
    }

    std::cout
        << "\n╔══════════════════════════════════════════════════════╗\n"
        <<   "║        Ascend 910B  Channel Capacity Sweep           ║\n"
        <<   "╚══════════════════════════════════════════════════════╝\n"
        << "  Source     : " << source << (isFile ? " [file]" : " [rtsp]") << "\n"
        << "  Source FPS : " << std::fixed << std::setprecision(1) << sourceFps << "\n"
        << "  Range      : " << chanStart << " → " << chanEnd
        << "  step " << chanStep << "\n"
        << "  Measure    : " << sweep::MEASURE_SEC << "s\n"
        << "  Model      : " << (mcfg.modelType == ModelType::YOLOV11 ? "yolov11" : "yolov3")
        << "  " << mcfg.resizeWidth << "×" << mcfg.resizeHeight
        << "  batch=" << mcfg.batchSize << "\n\n";

    cfg::WRITE_FRAMES = false;
    cfg::LOOP_VIDEO   = true;

    CsvLog csv("sweep_results.csv");

    struct StepResult {
        int    channels;
        double totalFps;
        double fpsPerChan;
        double lagMs;
        uint32_t skipUsed;
    };
    std::vector<StepResult> results;

    const uint32_t fixedSkip = cfg::DECODE_SKIP_INTERVAL;

    for (int ch = chanStart; ch <= chanEnd; ch += chanStep) {
        std::vector<std::string> urls(static_cast<size_t>(ch), source);
        std::cout << "── " << ch << " channel(s) ──\n";
        std::cout << "  [measure " << sweep::MEASURE_SEC << "s]\n";

        auto res = RunPipeline(urls, isFile, mcfg, sweep::MEASURE_SEC, &csv, ch, srcInfo);

        results.push_back({ch, res.avgFps, res.fpsPerChannel, res.avgLagMs, fixedSkip});

        std::cout << "  → total=" << std::fixed << std::setprecision(1) << res.avgFps
                  << "  fps/ch=" << std::fixed << std::setprecision(2) << res.fpsPerChannel
                  << "  lag="    << std::fixed << std::setprecision(1) << res.avgLagMs << "ms"
                  << "  skip="   << cfg::DECODE_SKIP_INTERVAL << "\n\n";
    }

    // Summary table
    std::cout
        << "┌───────────┬───────────┬────────────┬───────────┬──────┐\n"
        << "│ Channels  │ Total FPS │ FPS/Chan   │  Lag (ms) │ Skip │\n"
        << "├───────────┼───────────┼────────────┼───────────┼──────┤\n";
    for (const auto& r : results) {
        std::cout
            << "│ " << std::setw(9)  << r.channels                                  << " │ "
            << std::setw(9)  << std::fixed << std::setprecision(1) << r.totalFps    << " │ "
            << std::setw(10) << std::fixed << std::setprecision(2) << r.fpsPerChan  << " │ "
            << std::setw(9)  << std::fixed << std::setprecision(1) << r.lagMs       << " │ "
            << std::setw(4)  << r.skipUsed                                           << " │\n";
    }
    std::cout
        << "└───────────┴───────────┴────────────┴───────────┴──────┘\n\n";
    return 0;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) { PrintUsage(argv[0]); return 0; }

    const std::string mode = argv[1];
    if (mode != "--video" && mode != "--rtsp" && mode != "--sweep") {
        std::cerr << "Unknown mode '" << mode << "'.\n";
        PrintUsage(argv[0]);
        return -1;
    }

    MxInit();

    // ── Shared flags ──────────────────────────────────────────────────────────
    if (auto s = FlagVal(argc, argv, "--skip"); !s.empty())
        cfg::DECODE_SKIP_INTERVAL = static_cast<uint32_t>(std::stoi(s));
    
    // Flag para generar video automáticamente
    bool makeVideo = HasFlag(argc, argv, "--make-video");

    ModelConfig mcfg = LoadModelConfig("./config.json");
    mcfg.numDevices  = cfg::NUM_DEVICES;

    {
        uint32_t slots = cfg::NUM_DEVICES * cfg::MAX_VDEC_CHANNELS;
        std::cout << "[INFO] devices=" << cfg::NUM_DEVICES
                  << "  vdec_slots=" << slots
                  << "  skip=" << cfg::DECODE_SKIP_INTERVAL << "\n";
    }

    int ret = 0;

    // ─────────────────────────────────────────────────────────────────────────
    // Mode: --video <file> <channels> [options]
    // ─────────────────────────────────────────────────────────────────────────
    if (mode == "--video") {
        if (argc < 4) { PrintUsage(argv[0]); MxDeInit(); return -1; }

        const std::string file = argv[2];
        const int channels     = std::stoi(argv[3]);

        bool trackFrames = false;
        bool detectionFrames = false;

        if (HasFlag(argc, argv, "--frames") || HasFlag(argc, argv, "--track-frames")) {
            if (channels != 1) {
                std::cerr << "[ERROR] --frames / --track-frames require channels=1.\n";
                MxDeInit(); return -1;
            }
            if (HasFlag(argc, argv, "--frames")) {
                cfg::WRITE_FRAMES = true;
                detectionFrames = true;
            }
            if (HasFlag(argc, argv, "--track-frames")) {
                cfg::WRITE_TRACK_FRAMES = true;
                trackFrames = true;
            }
        }

        uint32_t dur = 0;
        if (auto s = FlagVal(argc, argv, "--duration"); !s.empty())
            dur = static_cast<uint32_t>(std::stoul(s));

        StreamInfo srcInfo = ProbeStream(file, /*isFile=*/true);
        ret = RunInfer(file, /*isFile=*/true, channels, mcfg, dur, srcInfo);
        
        // Generar video automáticamente si se solicitó
        if (makeVideo && ret == 0) {
            if (trackFrames) {
                GenerateVideoFromFrames(true);
            } else if (detectionFrames) {
                GenerateVideoFromFrames(false);
            }
        }

    // ─────────────────────────────────────────────────────────────────────────
    // Mode: --rtsp <url> <channels> [options]
    // ─────────────────────────────────────────────────────────────────────────
    } else if (mode == "--rtsp") {
        if (argc < 4) { PrintUsage(argv[0]); MxDeInit(); return -1; }

        const std::string url = argv[2];
        const int channels    = std::stoi(argv[3]);

        bool trackFrames = false;
        bool detectionFrames = false;

        if (HasFlag(argc, argv, "--frames") || HasFlag(argc, argv, "--track-frames")) {
            if (channels != 1) {
                std::cerr << "[ERROR] --frames / --track-frames require channels=1.\n";
                MxDeInit(); return -1;
            }
            if (HasFlag(argc, argv, "--frames")) {
                cfg::WRITE_FRAMES = true;
                detectionFrames = true;
            }
            if (HasFlag(argc, argv, "--track-frames")) {
                cfg::WRITE_TRACK_FRAMES = true;
                trackFrames = true;
            }
        }
        cfg::LOOP_VIDEO = true;

        uint32_t dur = 0;
        if (auto s = FlagVal(argc, argv, "--duration"); !s.empty())
            dur = static_cast<uint32_t>(std::stoul(s));

        StreamInfo srcInfo = ProbeStream(url, /*isFile=*/false);
        ret = RunInfer(url, /*isFile=*/false, channels, mcfg, dur, srcInfo);
        
        // Generar video automáticamente si se solicitó
        if (makeVideo && ret == 0) {
            if (trackFrames) {
                GenerateVideoFromFrames(true);
            } else if (detectionFrames) {
                GenerateVideoFromFrames(false);
            }
        }

    // ─────────────────────────────────────────────────────────────────────────
    // Mode: --sweep --video|--rtsp <src> <start> <end> <step> [options]
    // ─────────────────────────────────────────────────────────────────────────
    } else {
        if (argc < 7) { PrintUsage(argv[0]); MxDeInit(); return -1; }

        const std::string srcMode = argv[2];
        if (srcMode != "--video" && srcMode != "--rtsp") {
            std::cerr << "--sweep requires --video or --rtsp as second argument.\n";
            MxDeInit(); return -1;
        }

        const std::string source = argv[3];
        const int         start  = std::stoi(argv[4]);
        const int         end    = std::stoi(argv[5]);
        const int         step   = std::stoi(argv[6]);
        const bool        isFile = (srcMode == "--video");

        std::cout << "[INFO] Probing stream from " << source << " ...\n";
        StreamInfo srcInfo = ProbeStream(source, isFile);
        double sourceFps = sweep::DEFAULT_SRC_FPS;
        if (srcInfo.width > 0 && srcInfo.fps > 0.0) {
            sourceFps = srcInfo.fps;
            std::cout << "[INFO] Detected: "
                      << srcInfo.width << "×" << srcInfo.height
                      << "  " << std::fixed << std::setprecision(2)
                      << sourceFps << " fps\n";
        } else {
            std::cout << "[WARN] Probe failed — using default "
                      << sourceFps << " fps\n";
        }

        ret = RunSweep(source, isFile, start, end, step, sourceFps, mcfg, srcInfo);
    }

    MxDeInit();
    return ret;
}