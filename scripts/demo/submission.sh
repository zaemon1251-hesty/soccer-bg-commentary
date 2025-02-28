#!/bin/sh
# 成果発表用に、実況生成からデモ動画作成までを一気通貫で行うスクリプト

PWD=`pwd`

# # # 実況生成
# ./scripts/demo/run_from_csv.sh
# ./scripts/demo/run_from_csv_ja.sh

# # # 音声合成
# 自分の環境にある piper-phonemize に移動して実行する
cd $HOME/piper-phonemize
./scripts/run-openai.sh

cd $PWD
# sample_ids="25 28"
sample_ids="8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30"

for sample_id in $sample_ids;
do
    # 字幕焼付け
    # ./scripts/demo/add-sub.sh $sample_id ja
    ./scripts/demo/add-sub.sh $sample_id en

    # 音声を動画に結合
    # ./scripts/demo/add-wav.sh $sample_id ja
    ./scripts/demo/add-wav.sh $sample_id en
done
