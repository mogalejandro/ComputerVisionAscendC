#!/bin/bash

SOURCE="/home/byz/mxVison_detection/data/test3_1280.mp4"
CHANNELS=24  # Change this to however many you need
BASE_URL="rtsp://localhost:8554/mystream"

# for (( i=0; i<$CHANNELS; i++ ))
# do
#     echo "Starting stream: ${BASE_URL}_$i"
#     # ffmpeg -re -stream_loop -1 -i "$SOURCE" -c:v copy -f rtsp "${BASE_URL}_$i" &
#     
# done

# # Keep the script running so you can kill all streams at once with Ctrl+C
# wait

# ffmpeg -hide_banner -thread_queue_size 1024 -re -stream_loop -1 -i "$SOURCE" -c:v copy -rtsp_transport tcp -f rtsp "${BASE_URL}"

ffmpeg -hide_banner -thread_queue_size 1024 -re -stream_loop -1 -i "$SOURCE" \
  -c:v libx265 -profile:v main -level 5.1 \
  -tag:v hvc1 \
  -c:a copy \
  -rtsp_transport tcp -f rtsp "${BASE_URL}"

# ffmpeg -hide_banner -thread_queue_size 1024 -re -stream_loop -1 \
#   -i "$SOURCE" \
#   -c:v libx265 \
#   -pix_fmt nv12 \
#   -profile:v main -level 5.1 \
#   -tag:v hvc1 \
#   -rtsp_transport tcp -f rtsp "${BASE_URL}"