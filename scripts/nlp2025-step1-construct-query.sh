#!/bin/bash
# related to spotting json file
# v3 のアノテーションがある、プレミアリーグ映像を選択した


INPUT_JSON_DIR="data/spotting"
SPOTTING_MODEL="commentary_gold"
GAME_LIST_TXT="data/exist_targets.txt"

process () {
    video_data_csv=$1
    spotting_csv=$2
    option_name=$3

    # 空白でブレークせずに
    cat $GAME_LIST_TXT | while read TARGET_GAME;
    do
        echo $TARGET_GAME

        INPUT_FILE="$INPUT_JSON_DIR/$SPOTTING_MODEL/$TARGET_GAME/results_spotting.json"
        if [ ! -e "$INPUT_FILE" ]; then
            echo "File not found: $INPUT_FILE"
            continue
        fi

        # related to comment csv file
        INPUT_CSV_DIR="data/commentary"
        COMMENT_CSV_FILE="$INPUT_CSV_DIR/scbi-v2.csv"

        # related to output file
        OUTPUT_FILE="outputs/step1/$TARGET_GAME/$option_name.jsonl"
        mkdir -p "outputs/step1/$TARGET_GAME"

        # w-action を含むなら spotting_csv が必要
        if echo "$option_name" | grep -q "w-action"; then
            uv run python src/sn_providing/construct_query.py \
                --game "$TARGET_GAME" \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --comment_csv "$COMMENT_CSV_FILE" \
                --video_data_csv "$video_data_csv" \
                --spotting_csv "$spotting_csv"
        else
            uv run python src/sn_providing/construct_query.py \
                --game "$TARGET_GAME" \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --comment_csv "$COMMENT_CSV_FILE" \
                --video_data_csv "$video_data_csv"
        fi
    done
}

# outputs/step1/evaluation-target-sn-gamestate.jsonl 用のゲームごとのクエリ (Option 1), 
video_data_csv="data/from_video/players_in_frames_sn_gamestate.csv"
spotting_csv=""
process $video_data_csv $spotting_csv sn-gamestate

# outputs/step1/evaluation-target-correct-player-list.jsonl 用のゲームごとのクエリ (Option 2)
video_data_csv="data/from_video/players_in_frames.csv"
spotting_csv=""
process $video_data_csv $spotting_csv correct-player-list 

# outputs/step1/evaluation-target-sn-gamestate-w-action.jsonl 用のゲームごとのクエリ (Option 3)
video_data_csv="data/from_video/players_in_frames_sn_gamestate.csv"
spotting_csv="data/from_video/soccernet_spotting_labels.csv"
process $video_data_csv $spotting_csv sn-gamestate-w-action

