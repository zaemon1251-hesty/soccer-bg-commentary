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



INPUT_FILE="outputs/evaluation-target.jsonl"
OUTPUT_FILE="outputs/evaluation-target.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_FILE" \
    --retriever_type "openai-embedding"
