#!/bin/bash
# related to spotting json file
# v3 のアノテーションがある、プレミアリーグ映像を選択した

INPUT_JSON_DIR="data/spotting"
SPOTTING_MODEL="commentary_gold"
TARGET_GAME="england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"
INPUT_FILE="$INPUT_JSON_DIR/$SPOTTING_MODEL/$TARGET_GAME/results_spotting.json"

# related to comment csv file
INPUT_CSV_DIR="data/commentary"
COMMENT_CSV_FILE="$INPUT_CSV_DIR/scbi-v2.csv"

# related to output file
OUTPUT_FILE="outputs/$TARGET_GAME/2024-11-09-14-48-results_spotting_query.jsonl"
mkdir -p "outputs/$TARGET_GAME"


uv run python src/sn_providing/construct_query.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --comment_csv "$COMMENT_CSV_FILE" \
