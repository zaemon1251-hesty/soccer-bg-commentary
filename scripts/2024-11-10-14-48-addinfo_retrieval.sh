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


TARGET_GAME="england_epl/2016-2017/2016-09-24 - 14-30 Manchester United 4 - 1 Leicester"

INPUT_FILE="outputs/$TARGET_GAME/2024-11-09-14-48-results_spotting_query.jsonl"
OUTPUT_FILE="outputs/$TARGET_GAME/2024-11-09-14-48-results_addinfo_retrieval.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --game "$TARGET_GAME" \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --retriever_type "openai-embedding"
