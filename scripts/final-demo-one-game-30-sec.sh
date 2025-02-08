game="europe_uefa-champions-league/2016-2017/2017-04-12 - 21-45 Bayern Munich 1 - 2 Real Madrid"

# construct query
step1_query_output="outputs/demo-step1/$game.jsonl"
uv run python src/sn_providing/construct_query.py \
    --game "$game" \
    --input_file "data/spotting/commentary_gold/$game/results_spotting.json" \
    --output_file "$step1_query_output" \
    --comment_csv "data/commentary/scbi-v2.csv" \
    --video_data_csv "data/from_video/players_in_frames_sn_gamestate.csv"

# add info retrieval