#!/bin/bash
# v3 にアノテーションがある、プレミアリーグ映像を選択した

# MODEL_CONFIG = {
#     "model": "gpt-4",
#     "temperature": 0,
# }

# EMBEDDING_CONFIG = {
#     "model": "text-embedding-ada-002",
#     "chunk_size": 1000,
# }

model_name=gpt-4o

# A: ドキュメント検索せず、LLMに任せる
INPUT_FILE="outputs/step2/evaluation-target-sn-gamestate.jsonl"  # 選手同定モジュールの出力を入れる
OUTPUT_A_FILE="outputs/step3/$model_name/evaluation-target-a.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_A_FILE" \
    --retriever_type "openai-embedding" \
    --no_retrieval


# B: システム全体を動かす
INPUT_FILE="outputs/step2/evaluation-target-sn-gamestate.jsonl"
OUTPUT_B_FILE="outputs/step3/$model_name/evaluation-target-b.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_B_FILE" \
    --retriever_type "openai-embedding"


# C: 正解の選手リスト、正解の文書を選ぶ
INPUT_FILE="outputs/step2/evaluation-target-correct-player-list.jsonl"
OUTPUT_C_FILE="outputs/step3/$model_name/evaluation-target-c.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_C_FILE" \
    --retriever_type "openai-embedding" \
    --reference_documents_yaml "data/reference_documents/evaluation-samples.yaml"


# Astar: ドキュメント検索せず、LLMに任せる + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-sn-gamestate-w-action.jsonl"  # 選手同定モジュールの出力を入れる
OUTPUT_Astar_FILE="outputs/step3/$model_name/evaluation-target-a-star.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Astar_FILE" \
    --retriever_type "openai-embedding" \
    --no_retrieval


# Bstar: システム全体を動かす + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-sn-gamestate-w-action.jsonl"
OUTPUT_Bstar_FILE="outputs/step3/$model_name/evaluation-target-b-star.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Bstar_FILE" \
    --retriever_type "openai-embedding"


# Cstar: 正解の選手リスト、正解の文書を選ぶ + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-correct-player-list-w-action.jsonl"
OUTPUT_Cstar_FILE="outputs/step3/$model_name/evaluation-target-c-star.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Cstar_FILE" \
    --retriever_type "openai-embedding" \
    --reference_documents_yaml "data/reference_documents/evaluation-samples.yaml"