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


# Astar: ドキュメント検索せず、LLMに任せる + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-sn-gamestate-w-action.jsonl"  # 選手同定モジュールの出力を入れる
OUTPUT_Astar_FILE="outputs/step3/$model_name/evaluation-target-a-star.jsonl"
echo "Start Baseline"
uv run python src/sn_providing/addinfo_retrieval.py \
    --model $model_name \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Astar_FILE" \
    --retriever_type "openai-embedding" \
    --no_retrieval


# Bstar: システム全体を動かす + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-sn-gamestate-w-action.jsonl"
OUTPUT_Bstar_FILE="outputs/step3/$model_name/evaluation-target-b-star.jsonl"
echo "Start System"
uv run python src/sn_providing/addinfo_retrieval.py \
    --model $model_name \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Bstar_FILE" \
    --retriever_type "openai-embedding"


# Cstar: 正解の選手リスト、正解の文書を選ぶ + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-correct-player-list-w-action.jsonl"
OUTPUT_Cstar_FILE="outputs/step3/$model_name/evaluation-target-c-star.jsonl"
echo "Start System + GD + GP"
uv run python src/sn_providing/addinfo_retrieval.py \
    --model $model_name \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Cstar_FILE" \
    --retriever_type "openai-embedding" \
    --reference_documents_yaml "data/reference_documents/evaluation-samples.yaml"


# Bsharp: 正解の選手リスト + 文書抽出 + アクション情報を追加
INPUT_FILE="outputs/step2/evaluation-target-correct-player-list-w-action.jsonl"
OUTPUT_Bsharp_FILE="outputs/step3/$model_name/evaluation-target-b-sharp.jsonl"
echo "Start System + GP"
uv run python src/sn_providing/addinfo_retrieval.py \
    --model $model_name \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_Bsharp_FILE" \
    --retriever_type "openai-embedding" \
