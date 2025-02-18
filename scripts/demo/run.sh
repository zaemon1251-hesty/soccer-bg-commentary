# 1つのサンプルをもとにデモ実況を出力
target_game="england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"
target_half=2
target_start=474
target_end=504

uv run python src/sn_providing/main.py \
    --game "$target_game" \
    --half $target_half \
    --start $target_start  \
    --end $target_end \
    --save_jsonl "outputs/demo-step2/13.jsonl" \
    --save_srt "outputs/demo-step2/13.srt" \
    --mode run
