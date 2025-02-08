#!/bin/bash

# MODEL_CONFIG = {
#     "model": "gpt-3.5-turbo",
#     "temperature": 0,
# }

# EMBEDDING_CONFIG = {
#     "model": "text-embedding-ada-002",
#     "chunk_size": 1000,
# }


TARGET_GAME="england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"
INPUT_FILE="outputs/$TARGET_GAME/2024-09-19-14-05-results_spotting_query.jsonl"
OUTPUT_FILE="outputs/$TARGET_GAME/2024-10-03-21-40-results_addinfo_retrieval.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --retriever_type "openai-embedding"
