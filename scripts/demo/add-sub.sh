#!/bin/bash

# video と subtitle を結合する

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sample_id> <lang>"
    exit 1
fi

sample_id=$(printf "%04d" $1)
lang=$2

# 言語ごとにフォントの設定を分岐
if [ "$lang" = "en" ]; then
    font_settings="FontName=Helvetica,FontSize=24"
else
    font_settings="FontName=ヒラギノ角ゴシック W4,FontSize=24"
fi

# 普通の映像
src_video_path=outputs/demo-step2/$sample_id/video.mp4
srt_path=outputs/demo-step2/$sample_id/commentary-full-$lang.srt
dst_video_path=outputs/demo-step3/$sample_id-$lang.mp4

ffmpeg -y -i $src_video_path -vf subtitles=$srt_path $dst_video_path

# GSR 付きの映像
src_video_path=outputs/demo-step2/$sample_id/gsr_video.mp4
srt_path=outputs/demo-step2/$sample_id/commentary-full-$lang.srt
dst_video_path=outputs/demo-step3/$sample_id-$lang-gsr.mp4


ffmpeg -y -i $src_video_path -vf "subtitles='$srt_path':force_style='$font_settings'" $dst_video_path