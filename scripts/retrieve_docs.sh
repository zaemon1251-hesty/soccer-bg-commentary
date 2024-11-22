#!/bin/bash
# related to spotting json file
# v3 のアノテーションがある、プレミアリーグ映像を選択した

INPUT_JSON_DIR="data/spotting"
SPOTTING_MODEL="commentary_gold"
# TARGET_GAME="england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
TARGET_GAME="england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"
INPUT_FILE="$INPUT_JSON_DIR/$SPOTTING_MODEL/$TARGET_GAME/results_spotting.json"

# related to comment csv file
INPUT_CSV_DIR="data/commentary"
COMMENT_CSV_FILE="$INPUT_CSV_DIR/scbi-v2.csv"

uv run python src/sn_providing/retrieve_documents.py \
    --game "$TARGET_GAME" \
    --retriever_type "openai-embedding" \
    --comment_csv "$COMMENT_CSV_FILE" 
