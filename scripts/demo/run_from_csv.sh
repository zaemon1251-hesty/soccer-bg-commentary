# 20サンプルのデモ動画のつくるためのスクリプト

# sample_id: 22

# arg path csv (id,game,half,start,end)
csv_path="data/demo/sample_metadata.csv"

base_dir="outputs/demo-step2"

# デモ実況を出力
uv run python src/sn_providing/main.py \
    --input_method csv \
    --mode run \
    --input_csv $csv_path \
    --output_base_dir $base_dir \
    --seed 101010

# # 現実の実況を出力
# uv run python src/sn_providing/main.py \
#     --input_method csv \
#     --mode reference \
#     --input_csv $csv_path \
#     --output_base_dir $base_dir

