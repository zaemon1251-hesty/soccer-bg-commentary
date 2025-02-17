#!/bin/bash

model_name=gpt-4o

uv run python src/sn_providing/collect_evaluation_comments.py \
    --input_a_file "outputs/step3/$model_name/evaluation-target-a.jsonl" \
    --input_b_file "outputs/step3/$model_name/evaluation-target-b.jsonl" \
    --input_c_file "outputs/step3/$model_name/evaluation-target-c.jsonl" \
    --input_a_star_file "outputs/step3/$model_name/evaluation-target-a-star.jsonl" \
    --input_b_star_file "outputs/step3/$model_name/evaluation-target-b-star.jsonl" \
    --input_c_star_file "outputs/step3/$model_name/evaluation-target-c-star.jsonl" \
    --input_b_sharp_file "outputs/step3/$model_name/evaluation-target-b-sharp.jsonl" \
    --output_file "outputs/step4/$model_name/evaluation-comments.csv" \
    --reference_documents_yaml "data/reference_documents/evaluation-samples.yaml"