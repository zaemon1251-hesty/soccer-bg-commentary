#!/bin/bash

# video と audion を結合する

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sample_id> <lang>"
    exit 1
fi

sample_id=$(printf "%04d" $1)
lang=$2

src_video_path=outputs/demo-step3/$sample_id-$lang.mp4
audio_path=outputs/demo-step2/$sample_id/commentary-full-$lang.wav
dst_video_path=outputs/demo-step4/$sample_id-$lang.mp4

ffmpeg -i $src_video_path -i $audio_path -c:v copy -map 0:v:0 -map 1:a:0 -shortest $dst_video_path

