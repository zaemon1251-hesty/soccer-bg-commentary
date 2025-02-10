# システム全体の実行スクリプト
# 正しい入力かチェック
if [ $# -ne 4 ]; then
    echo "Usage: $0 game half start end"
    exit 1
fi

# 引数の取得
game=$1
half=$2
start=$3
end=$4


# ディレクトリの設定
base_dir=.
spotting_dir=$base_dir/sn-caption/Benchmarks/TemporallyAwarePooling
script_dir=$base_dir/sn-script
tracking_dir=$base_dir/tracklab
rag_dir=$base_dir/sn-providing

# データのパス
silence_dist_csv="$spotting_dir/data/silence_distribution_over_1sec.csv"
action_rate_csv="$spotting_dir/data/Additional_Info_Ratios__Before_and_After.csv"
SoccerNet_path="/Users/heste/workspace/SoccerNet"

side_team_map_csv="/Users/heste/workspace/soccernet/sn-script/database/misc/side_to_team.csv"

player_master_csv="/Users/heste/workspace/soccernet/sn-script/database/misc/sncaption_players.csv"

# 評価用の映像のメタデータを管理するCSV: sample_id,game,half,time
evaluatoin_sample_path="/Users/heste/workspace/soccernet/sn-script/database/misc/RAGモジュール出力サンプル-13090437a14481f485ffdf605d3408cd.csv"

output_csv_path="/Users/heste/workspace/soccernet/sn-script/database/misc/players_in_frames_sn_gamestate_29-33.csv"


# モデルのパス
trained_weights="/local/moriy/model/soccernet/sn-gamestate/reid/sn-gamestate-v3-720p/2024-11-26-10-21-36/0/2024_11_26_10_22_05_22S9c0e99da-e916-4634-95c1-4ab4e94998e5model/job-0_29_model.pth.tar"



# 実行時の一時変数の設定
export CUDA_VISIBLE_DEVICES=3
exp_name=$(date '+%Y-%m-%d_%H-%M')
tmp_result_path=$base_dir/$exp_name
tmp_video_path=$base_dir/$exp_name/video

# ディレクトリの作成
mkdir -p $tmp_video_path
mkdir -p $tmp_result_path

# スポッティングモジュール
spotting_jsonl=$tmp_result_path/spotting.jsonl

cd $spotting_dir
python src/main_v2.py --game $game --min $start --max $end --output_jsonl $spotting_jsonl \
    --ignore_under_1sec  \
    --timing_algo empirical \
        --empirical_dist_csv $silence_dist_csv \
    --label_algo action_spotting \
        --action_window_size 15 \
        --action_rate_csv $action_rate_path \
        --default_rate 0.18 \
        --addinfo_force \
        --only_offplay

# 選手同定モジュール
player_jsonl=$tmp_result_path/player.jsonl

## 1. 対象の箇所の切り取り: 
cd $script_dir
python src/sn_script/video2images.py \
    --output_base_path $tmp_video_path \
    --SoccerNet_path $SoccerNet_path \
    --resolution 720p \
    --fps 25 \
    --threads 4 \
    --target_game $game \
    --target_half $start \
    --output video \

## 2. 選手同定の実行: 
cd $tracking_dir
python -m tracklab.main -cn soccernet-v2-exp005 \
    modules.reid.training_enabled=False \
    modules.team._target_=sn_gamestate.team.TrackletTeamClustering \
    wandb.name=$exp_name \
    modules.reid.cfg.model.load_weights=$trained_weights \
    dataset.dataset_path="$tmp_video_path" \
    hydra.run.dir=$tmp_result_path

## 3. 選手名への対応付け: 
gsr_result_pklz="$tmp_result_path/states/sn-gamestate-v2.pklz"

# TODO これどうする？
evaluatoin_sample_path=None

cd $script_dir
python src/sn_script/player_identification/result2player.py \
    --gsr_result_pklz $gsr_result_pklz \
    --output_jsonl $player_jsonl \
    --side_team_map_csv $side_team_map_csv \
    --player_master_csv $player_master_csv \
    --evaluatoin_sample_path $evaluatoin_sample_path

# RAGモジュール

## 1. クエリ構築:
cd $rag_dir
python src/sn_providing/construct_query.py
## 2. コメント生成:
python src/sn_providing/addinfo_retrieval.py
