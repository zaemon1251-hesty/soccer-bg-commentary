#!/bin/bash

# video と audio を結合する

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sample_id> <lang>"
    exit 1
fi

sample_id=$(printf "%04d" $1)
lang=$2

dst_base_dir=outputs/demo-step4/$sample_id/
mkdir -p $dst_base_dir

add_wav () {
    suffix=$1
    src_video_path=$2
    audio_path=outputs/demo-step2/$sample_id/commentary-full-openai-$lang.wav
    dst_video_path=$dst_base_dir/$lang-$suffix.mp4
    ffmpeg -y -i $src_video_path -i $audio_path -c:v copy -map 0:v:0 -map 1:a:0 -shortest $dst_video_path
}


suffix=wo-sub-wo-gsr
src_video_path=outputs/demo-step2/$sample_id/video.mp4
add_wav $suffix $src_video_path


suffix=wo-sub-w-gsr
src_video_path=outputs/demo-step2/$sample_id/gsr_video.mp4
add_wav $suffix $src_video_path

suffix=w-sub-wo-gsr
src_video_path=outputs/demo-step3/$sample_id-$lang.mp4
add_wav $suffix $src_video_path

suffix=w-sub-w-gsr
src_video_path=outputs/demo-step3/$sample_id-$lang-gsr.mp4
add_wav $suffix $src_video_path

