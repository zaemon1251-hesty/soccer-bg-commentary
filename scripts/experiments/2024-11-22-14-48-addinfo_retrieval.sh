#!/bin/bash
# v3 にアノテーションがある、プレミアリーグ映像を選択した

# MODEL_CONFIG = {
#     "model": "gpt-3.5-turbo",
#     "temperature": 0,
# }

# EMBEDDING_CONFIG = {
#     "model": "text-embedding-ada-002",
#     "chunk_size": 1000,
# }


TARGET_GAME="europe_uefa-champions-league/2014-2015/2015-05-05 - 21-45 Juventus 2 - 1 Real Madrid"

INPUT_FILE="outputs/$TARGET_GAME/2024-11-22-14-48-results_spotting_query.jsonl"
OUTPUT_FILE="outputs/$TARGET_GAME/2024-11-22-14-48-results_addinfo_retrieval.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --retriever_type "openai-embedding"
