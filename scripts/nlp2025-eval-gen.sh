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

# A: ドキュメント検索せず、LLMに任せる
INPUT_B_FILE="outputs/evaluation-target-b.jsonl"  # 選手同定モジュールの出力を入れる
OUTPUT_A_FILE="outputs/evaluation-target-a.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_B_FILE" \
    --output_file "$OUTPUT_A_FILE" \
    --retriever_type "openai-embedding" \
    --no_retrieval


# B: システム全体を動かす
INPUT_B_FILE="outputs/evaluation-target-b.jsonl"
OUTPUT_B_FILE="outputs/evaluation-target-b.jsonl"

if [ ! -e "$INPUT_B_FILE" ]; then # まだ作っていない場合は収集する
    uv run python src/sn_providing/select_evaluation_examples.py \
        --query_json_dir outputs \
        --exist_target_txt data/exist_targets.txt \
        --jsonl_filename sn-gamestate-results_spotting_query.jsonl \
        --output_dir outputs \
        --output_basename evaluation-target-b
fi
INPUT_B_FILE="outputs/evaluation-target-b.jsonl"
OUTPUT_B_FILE="outputs/evaluation-target-b.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_B_FILE" \
    --output_file "$OUTPUT_B_FILE" \
    --retriever_type "openai-embedding"


# C: 正解の選手リスト、正解の文書を選ぶ
OUTPUT_C_FILE="outputs/evaluation-target-c.jsonl"
uv run python src/sn_providing/addinfo_retrieval.py \
    --input_file "$INPUT_FILE" \
    --output_file "$OUTPUT_C_FILE" \
    --retriever_type "openai-embedding" \
    --reference_documents_csv "data/reference_documents/evaluation-samples.yaml"


