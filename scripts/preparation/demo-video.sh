#!/bin/bash

src_dir="/Users/heste/Downloads/video"
src_gsr_dir="/Users/heste/workspace/soccernet/tracklab/outputs/sn-gamestate-v2/2024-12-17/10-57-24/visualization/videos"
dst_dir="./outputs/demo-step2"
for i in {0008..0033}; do
  i=$(printf %04d $i)
  # コピー先ディレクトリがない場合に備えて作成
  mkdir -p "$dst_dir/$i"
  # コピー
  cp "$src_dir/$i.mp4" "$dst_dir/$i/video.mp4"
  cp "$src_gsr_dir/$i.mp4" "$dst_dir/$i/gsr_video.mp4"
done
