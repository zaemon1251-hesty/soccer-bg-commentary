# Option 1
#  sn-gamestate-w-action
# uv run python src/sn_providing/select_evaluation_examples.py \
#     --query_json_dir outputs/step1 \
#     --exist_target_txt data/exist_targets.txt \
#     --jsonl_filename sn-gamestate-w-action.jsonl \
#     --output_dir outputs/step2 \
#     --output_basename evaluation-target-sn-gamestate


# Option 2
#  correct-player-list
# uv run python src/sn_providing/select_evaluation_examples.py \
#     --query_json_dir outputs/step1 \
#     --exist_target_txt data/exist_targets.txt \
#     --jsonl_filename correct-player-list.jsonl \
#     --output_dir outputs/step2 \
#     --output_basename evaluation-target-correct-player-list

# Option 3 
#  sn-gamestate-w-action
# uv run python src/sn_providing/select_evaluation_examples.py \
#     --query_json_dir outputs/step1 \
#     --exist_target_txt data/exist_targets.txt \
#     --jsonl_filename sn-gamestate-w-action.jsonl \
#     --output_dir outputs/step2 \
#     --output_basename evaluation-target-sn-gamestate-w-action

# Option 4
#  correct-player-list-w-action
uv run python src/sn_providing/select_evaluation_examples.py \
    --query_json_dir outputs/step1 \
    --exist_target_txt data/exist_targets.txt \
    --jsonl_filename correct-player-list-w-action.jsonl \
    --output_dir outputs/step2 \
    --output_basename evaluation-target-correct-player-list-w-action