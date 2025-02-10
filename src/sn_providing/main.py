"""
デモ動画を作成するためのスクリプト
"""
import sys
import os
import logging
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI as LangChainOpenAI
from tap import Tap

from sn_providing.entity import CommentDataList, SpottingDataList, VideoData, CommentData, SpottingData
from sn_providing.spotting_module import SpottingModule, MainV2Argument
from sn_providing.construct_query import build_query
from sn_providing.addinfo_retrieval import (
    get_rag_chain, 
    get_retriever, 
    MODEL_CONFIG, 
    EMBEDDING_CONFIG, 
    SEARCH_CONFIG, 
    PERSIST_LANGCHAIN_DIR
)

# .env の環境変数を読み込む
load_dotenv()



# ロガーの設定
time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
## ログファイル出力
file_handler = logging.FileHandler(f"logs/main--{time_string}.log")
## 標準出力
stream_handler = logging.StreamHandler()
## ログ+標準出力
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class MainArgument(Tap):
    """
    メインスクリプトの引数
    """
    game: str
    half: int
    start: float
    end: float
    video_path: str


def get_utterance_length(utterance: str):
    # 200 wpm (早口)という設定で，発話の長さを計算
    # return 秒数
    words = utterance.split()
    word_count = len(words)
    time = word_count * (60.0 / 200.0)
    return  time


def generate_commentary_for_demo(
    game: str, 
    half: int, 
    start: float, 
    end: float,
    comment_csv: str,
    video_csv: str,
    label_csv: str
):
    """
    デモ動画のための実況を生成する
    """
    # スポッティングモジュール関連
    spotting_params = MainV2Argument().as_dict()
    spotting_model = SpottingModule(spotting_params)
    
    # 選手同定+ragモジュール関連
    game_metadata = SpottingDataList.extract_data_from_game(game)
    gold_comment_data = CommentDataList.read_csv(comment_csv) # 映像の説明のモック用・comment_data_listのコメント履歴が使い物にならない時に，検索クエリを補強する用
    video_data = VideoData(video_csv, label_csv=label_csv)
    llm = LangChainOpenAI(**MODEL_CONFIG)
    retriever = get_retriever(
        "openai-embedding",
        langchain_store_dir=PERSIST_LANGCHAIN_DIR,
        embedding_config=EMBEDDING_CONFIG,
        search_config=SEARCH_CONFIG,
    )
    _, _, rag_chain = get_rag_chain(
        retriever=retriever, langchain=llm
    )
    
    def build_query_for_demo(comment_data_list, game, half, time, game_metadata):
        filtered_comment_list = CommentDataList.filter_by_half_and_time(
            comment_data_list, half, time
        )
        video_data_dict = video_data.get_data(
            game, half, time
        )

        query_args = {
            "comments": filtered_comment_list,
            "game_metadata": game_metadata,
            "players": video_data_dict["players"],
            "actions": video_data_dict["actions"]
        }

        query = build_query(**query_args)
        return query

    
    # 終了条件
    finish_time = end

    # ループ中に変更が加わる変数
    prev_end = start
    comment_data_list = CommentDataList([]) # ここに生成したコメントを順々に履歴として追加していく

    while 1:
        # 1ループ: 
        ## スポッティング
        next_ts, next_label = spotting_model(
            previous_t=prev_end, current_t=finish_time, game=game, half=half
        )
        
        if next_label == 0: # 映像の説明
            # TODO 1 まずモックとして，近傍の映像の説明を取得する処理を実装
            # TODO 2 tanaka-san の Integrate System を使って，映像の説明を取得する
            
            # 発話開始時間, 発話終了時間を設定
            next_start = next_ts
            next_end = next_ts + 1.0 

        elif next_label == 1: # 付加的情報
            # コメント生成
            query = build_query_for_demo(
                comment_data_list, game, half, next_ts, game_metadata
            )
            response = rag_chain.invoke(
                query=query, retriever=retriever, langchain=llm
            )

            # 発話開始時間, 発話終了時間を設定
            next_start = next_ts
            next_end = next_ts + get_utterance_length(response)

            # コメント履歴に追加
            comment_data_list.comments.append(
                CommentData(
                    half=half, start_time=next_start, text=response, category=next_label
                )
            )

        # 次のループのために更新
        prev_end = next_end

        # 終了条件
        if finish_time <= prev_end:
            break
    
    # 生成したコメントを保存
    comment_data_list.to_jsonline("outputs/demo-step2/commentary.jsonl")