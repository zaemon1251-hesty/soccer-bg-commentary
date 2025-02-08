#!/bin/bash

# related to spotting json file
INPUT_JSON_DIR="data/spotting"
SPOTTING_MODEL="commentay_gold"
TARGET_GAME="england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"
INPUT_FILE="$INPUT_JSON_DIR/$SPOTTING_MODEL/$TARGET_GAME/results_spotting.json"

# related to comment csv file
INPUT_CSV_DIR="data/commentary"
COMMENT_CSV_FILE_HALF1="$INPUT_CSV_DIR/gpt-3.5-turbo-1106_500game_1_llm_annotation.csv"
COMMENT_CSV_FILE_HALF2="$INPUT_CSV_DIR/gpt-3.5-turbo-1106_500game_2_llm_annotation.csv"

# related to output file
OUTPUT_FILE="outputs/$TARGET_GAME/2024-09-19-14-05-results_spotting_query.jsonl"
mkdir -p "outputs/$TARGET_GAME"


uv run python src/sn_providing/construct_query.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --comment_csv_file_half1 "$COMMENT_CSV_FILE_HALF1" \
    --comment_csv_file_half2 "$COMMENT_CSV_FILE_HALF2"
