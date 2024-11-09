#!/bin/bash
# related to spotting json file
INPUT_JSON_DIR="data/spotting"
SPOTTING_MODEL="commentary_gold"
TARGET_GAME="england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"
INPUT_FILE="$INPUT_JSON_DIR/$SPOTTING_MODEL/$TARGET_GAME/results_spotting.json"

# related to comment csv file
INPUT_CSV_DIR="data/commentary"
COMMENT_CSV_FILE="$INPUT_CSV_DIR/scbi-v2.csv"

# related to output file
OUTPUT_FILE="outputs/$TARGET_GAME/2024-10-26-14-48-results_spotting_query.jsonl"
mkdir -p "outputs/$TARGET_GAME"


uv run python src/sn_providing/construct_query.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --comment_csv "$COMMENT_CSV_FILE" \
