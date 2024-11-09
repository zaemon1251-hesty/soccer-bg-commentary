#!/bin/bash

# MODEL_CONFIG = {
#     "model": "gpt-3.5-turbo",
#     "temperature": 0,
# }

# EMBEDDING_CONFIG = {
#     "model": "text-embedding-ada-002",
#     "chunk_size": 1000,
# }


TARGET_GAME="england_epl/2015-2016/2015-08-29 - 17-00 Liverpool 0 - 3 West Ham"
INPUT_FILE="outputs/$TARGET_GAME/2024-10-26-14-48-results_spotting_query.jsonl"
OUTPUT_FILE="outputs/$TARGET_GAME/2024-10-26-14-48-results_addinfo_retrieval.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --retriever_type "openai-embedding"
