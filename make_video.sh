#!/usr/bin/env bash
set -euo pipefail

SOURCE_FPS=25
DO_DET=false
DO_TRACK=false
FPS=""
CRF=18

while [[ $# -gt 0 ]]; do
    case "$1" in
        --det)    DO_DET=true;   shift ;;
        --track)  DO_TRACK=true; shift ;;
        --fps)    FPS="$2";       shift 2 ;;
        --crf)    CRF="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! $DO_DET && ! $DO_TRACK; then
    DO_DET=true
fi

if ! command -v ffmpeg &>/dev/null; then
    echo "ERROR: ffmpeg not found. Install with: sudo apt install ffmpeg"
    exit 1
fi

make_video() {
    local framedir="$1"
    local outfile="$2"

    if [[ ! -d "$framedir" ]]; then
        echo "Directory '$framedir' not found — skipping."
        return
    fi

    local frames_sorted
    frames_sorted=$(find "$framedir" -maxdepth 1 -name 'f*.jpg' \
                    | awk -F'f' '{n=$NF; gsub(/\.jpg$/,"",n); print n+0, $0}' \
                    | sort -n | awk '{print $2}')

    local nframes=$(echo "$frames_sorted" | wc -l)

    if [[ "$nframes" -eq 0 ]]; then
        echo "No frames found in '$framedir' — skipping."
        return
    fi

    local effective_fps
    if [[ -n "$FPS" ]]; then
        effective_fps="$FPS"
    elif [[ -f "$framedir/meta.txt" ]]; then
        local skip=$(grep -oE 'skip=[0-9]+' "$framedir/meta.txt" | grep -oE '[0-9]+' || echo "0")
        effective_fps=$((SOURCE_FPS / (skip + 1)))
        [[ "$effective_fps" -lt 1 ]] && effective_fps=1
        echo "skip=$skip → playback fps=$effective_fps"
    else
        effective_fps="$SOURCE_FPS"
    fi

    echo "$nframes frames → $outfile (fps=$effective_fps crf=$CRF)"

    local listfile=$(mktemp)
    for frame in $frames_sorted; do
        echo "file '$(realpath "$frame")'" >> "$listfile"
    done

    ffmpeg -y -f concat -safe 0 -r "$effective_fps" -i "$listfile" \
        -c:v libx264 -pix_fmt yuv420p -crf "$CRF" -movflags +faststart "$outfile"

    rm -f "$listfile"
    echo "Saved → $outfile"
}

next_index() {
    local idx=0
    while true; do
        $DO_DET && [[ -f "det_${idx}.mp4" ]] && { idx=$((idx+1)); continue; }
        $DO_TRACK && [[ -f "track_${idx}.mp4" ]] && { idx=$((idx+1)); continue; }
        break
    done
    echo "$idx"
}

IDX=$(next_index)

$DO_DET && make_video "out" "det_${IDX}.mp4"
$DO_TRACK && make_video "out_track" "track_${IDX}.mp4"
