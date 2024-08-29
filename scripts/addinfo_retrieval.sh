#!/bin/bash

TARGET_GAME="england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"
INPUT_FILE="outputs/$TARGET_GAME/results_spotting_query.jsonl"
OUTPUT_FILE="outputs/$TARGET_GAME/results_addinfo_retrieval.jsonl"

uv run python src/sn_providing/addinfo_retrieval.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
