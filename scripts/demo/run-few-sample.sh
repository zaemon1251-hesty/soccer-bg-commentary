# 1サンプルのデモ動画のつくるためのスクリプト

# sample_id: 22



# sample_id: 13
if [ ! -e outputs/demo-step2/13.jsonl ]; then
    target_game="england_epl/2015-2016/2015-08-23 - 15-30 West Brom 2 - 3 Chelsea"
    target_half=2
    target_start=474
    target_end=504

    # デモ実況を出力
    uv run python src/sn_providing/main.py \
        --game "$target_game" \
        --half $target_half \
        --start $target_start  \
        --end $target_end \
        --save_jsonl "outputs/demo-step2/13.jsonl" \
        --save_srt "outputs/demo-step2/13.srt" \
        --mode run
    
    # 現実の実況を出力
    uv run python src/sn_providing/main.py \
        --game "$target_game" \
        --half $target_half \
        --start $target_start  \
        --end $target_end \
        --save_jsonl "outputs/demo-step2/13-ref.jsonl" \
        --save_srt "outputs/demo-step2/13-ref.srt" \
        --mode reference
fi


# sample_id: 15
if [ ! -e outputs/demo-step2/15.jsonl ]; then
    target_game="germany_bundesliga/2016-2017/2016-11-05 - 17-30 Hamburger SV 2 - 5 Dortmund"
    target_half=1
    target_start=1600
    target_end=1630

    # デモ実況を出力
    uv run python src/sn_providing/main.py \
        --game "$target_game" \
        --half $target_half \
        --start $target_start  \
        --end $target_end \
        --save_jsonl "outputs/demo-step2/15.jsonl" \
        --save_srt "outputs/demo-step2/15.srt" \
        --mode run
    
    # 現実の実況を出力
    uv run python src/sn_providing/main.py \
        --game "$target_game" \
        --half $target_half \
        --start $target_start  \
        --end $target_end \
        --save_jsonl "outputs/demo-step2/15-ref.jsonl" \
        --save_srt "outputs/demo-step2/15-ref.srt" \
        --mode reference
fi
