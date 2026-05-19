#pragma once
/**
 * tracker/SimpleTracker.h
 * ─────────────────────────────────────────────────────────────────────────────
 * Header-only greedy IoU multi-object tracker.
 *
 * Usage (per stream):
 *   SimpleTracker tracker;
 *   // each frame:
 *   std::vector<cv::Rect2f> dets = ...;
 *   tracker.update(dets);
 *   for (const auto& t : tracker.tracks()) { ... }
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <algorithm>
#include <deque>
#include <vector>
#include "opencv2/opencv.hpp"

struct Track {
    int              id     = 0;
    cv::Rect2f       bbox;
    std::deque<cv::Point2f> traj;   ///< centroid history (oldest first)
    int              misses = 0;    ///< frames since last matched detection
};

class SimpleTracker {
public:
    // Hyperparameters — set before first update(), or via ModelConfig in RunPipeline.
    int   maxMisses = 30;    ///< delete track after this many unmatched frames
    int   maxTraj   = 50;    ///< max trajectory length (centroid points)
    float iouThresh = 0.30f; ///< min IoU to match a detection to an existing track

    /**
     * Match @p dets to existing tracks (greedy IoU), update track states,
     * and create new tracks for unmatched detections.
     */
    void update(const std::vector<cv::Rect2f>& dets) {
        std::vector<bool> detUsed(dets.size(), false);

        // ── Match detections to existing tracks (greedy, descending IoU) ──────
        for (auto& t : tracks_) {
            float bestScore = iouThresh;
            int   bestDet   = -1;
            for (size_t d = 0; d < dets.size(); ++d) {
                if (detUsed[d]) continue;
                float score = iou(t.bbox, dets[d]);
                if (score > bestScore) {
                    bestScore = score;
                    bestDet   = static_cast<int>(d);
                }
            }
            if (bestDet >= 0) {
                t.bbox   = dets[static_cast<size_t>(bestDet)];
                t.misses = 0;
                detUsed[static_cast<size_t>(bestDet)] = true;
                appendCentroid(t);
            } else {
                ++t.misses;
            }
        }

        // ── Remove stale tracks ───────────────────────────────────────────────
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                           [this](const Track& t){ return t.misses > maxMisses; }),
            tracks_.end());

        // ── Create new tracks for unmatched detections ────────────────────────
        for (size_t d = 0; d < dets.size(); ++d) {
            if (detUsed[d]) continue;
            Track t;
            t.id   = nextId_++;
            t.bbox = dets[d];
            appendCentroid(t);
            tracks_.push_back(std::move(t));
        }
    }

    const std::vector<Track>& tracks() const { return tracks_; }

private:
    std::vector<Track> tracks_;
    int nextId_ = 0;

    static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
        cv::Rect2f inter = a & b;
        float ia = inter.area();
        if (ia <= 0.f) return 0.f;
        return ia / (a.area() + b.area() - ia);
    }

    void appendCentroid(Track& t) {
        cv::Point2f c(t.bbox.x + t.bbox.width  * 0.5f,
                      t.bbox.y + t.bbox.height * 0.5f);
        t.traj.push_back(c);
        if (static_cast<int>(t.traj.size()) > maxTraj)
            t.traj.pop_front();
    }
};
