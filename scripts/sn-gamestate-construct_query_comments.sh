#!/bin/bash
# related to spotting json file
# v3 のアノテーションがある、プレミアリーグ映像を選択した

INPUT_JSON_DIR="data/spotting"
SPOTTING_MODEL="commentary_gold"

GAME_LIST_TXT="data/exist_targets.txt"

# 空白でブレークせずに
cat $GAME_LIST_TXT | while read TARGET_GAME;
do
    echo $TARGET_GAME

    INPUT_FILE="$INPUT_JSON_DIR/$SPOTTING_MODEL/$TARGET_GAME/results_spotting.json"
    if [ ! -e "$INPUT_FILE" ]; then
        echo "File not found: $INPUT_FILE"
        continue
    fi

    # A,Bを作るためならこれ↓
    video_data_csv="data/from_video/players_in_frames_sn_gamestate.csv"

    # C',C を作るためならこれ↓
    # video_data_csv="data/from_video/players_in_frames.csv"

    # related to comment csv file
    INPUT_CSV_DIR="data/commentary"
    COMMENT_CSV_FILE="$INPUT_CSV_DIR/scbi-v2.csv"

    # related to output file
    OUTPUT_FILE="outputs/$TARGET_GAME/sn-gamestate-results_spotting_query.jsonl"
    mkdir -p "outputs/$TARGET_GAME"

    uv run python src/sn_providing/construct_query.py \
        --game "$TARGET_GAME" \
        --input_file "$INPUT_FILE" \
        --output_file "$OUTPUT_FILE" \
        --comment_csv "$COMMENT_CSV_FILE" \
        --video_data_csv "$video_data_csv"
done
