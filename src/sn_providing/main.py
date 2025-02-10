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


class DemoRunner:
    def __init__(
        self,
        game: str, 
        half: int, 
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
        gold_comment_data_list: CommentDataList = CommentDataList.read_csv(comment_csv, game) # 映像の説明のモック用・comment_data_listのコメント履歴が使い物にならない時に，検索クエリを補強する用
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
        
        # メンバー変数
        self.game = game
        self.half = half
        self.gold_comment_data_list = gold_comment_data_list
        self.video_data = video_data
        self.spotting_model = spotting_model
        self.game_metadata = game_metadata
        self.rag_chain = rag_chain
        
        
    def build_query_for_demo(self, comment_data_list, time):
        """ game, half, game_metadata, video_data は関数内で定義された変数を用いる  """
        filtered_comment_list = CommentDataList.filter_by_half_and_time(
            comment_data_list, self.half, time
        )
        video_data_dict = self.video_data.get_data(
            self.game, self.half, time
        )

        query_args = {
            "comments": filtered_comment_list,
            "game_metadata": self.game_metadata,
            "players": video_data_dict["players"],
            "actions": video_data_dict["actions"]
        }

        query = build_query(**query_args)
        return query

    def run(
        self,
        start: float, 
        end: float,
    ):
        # 終了条件
        finish_time = end

        # ループ中に変更が加わる変数
        prev_end = start
        comment_data_list = CommentDataList([]) # ここに生成したコメントを順々に履歴として追加していく
        
        # 文脈情報として使うため，gold_comment_data_list に含まれる，start までの comment を追加
        for comment in self.gold_comment_data_list.comments:
            if comment.half ==self. half and comment.end_time < start:
                comment_data_list.comments.append(comment)
        
        while 1:
            ## スポッティング
            next_ts, next_label = self.spotting_model(
                previous_t=prev_end, game=self.game, half=self.half
            )
            
            if next_label == 0: # 映像の説明
                # まずモックとして，近傍の映像の説明を取得する
                comment = self.gold_comment_data_list.get_comment_nearest_time(next_ts)
                
                # TODO 上記の実装をコメントアウトし，tanaka-san の Integrate System を使って映像の説明を生成する
                
                # 発話開始時間, 発話終了時間を設定
                next_start = next_ts
                next_end = next_ts + get_utterance_length(comment)
                
                # コメント履歴に追加
                comment_data_list.comments.append(
                    CommentData(
                        half=self.half, start_time=next_start, end_time=next_end,
                        text=comment, category=next_label
                    )
                )

            elif next_label == 1: # 付加的情報
                # 付加的情報を生成
                query = self.build_query_for_demo(
                    comment_data_list, next_ts
                )
                spot = SpottingData(
                    game=self.game, 
                    half=self.half, 
                    category=next_label, 
                    game_time=next_ts, 
                    query=query,
                    position=int(next_ts)*1000,
                    confidence=1.
                )
                response = self.rag_chain.invoke(spot)

                # 発話開始時間, 発話終了時間を設定
                next_start = next_ts
                next_end = next_ts + get_utterance_length(response)

                # コメント履歴に追加
                comment_data_list.comments.append(
                    CommentData(
                        half=self.half, start_time=next_start, end_time=next_end,
                        text=response, category=next_label
                    )
                )
            else:
                raise RuntimeError(f"無効な発話ラベルです: {next_label}")

            # 次のループのために更新
            prev_end = next_end

            # 終了条件
            if finish_time <= prev_end:
                break
        
        # 文脈情報として使うためにいれたgold_comment_data_listのコメントを削除
        for comment in self.gold_comment_data_list.comments:
            if comment.half == self.half and comment.start_time < start:
                comment_data_list.comments.remove(comment)
        
        # 保存
        comment_data_list.to_jsonline("outputs/demo-step2/commentary.jsonl")


if __name__ == "__main__":
    args = MainArgument().parse_args()
    
    demo_runner = DemoRunner(
        args.game, args.half,
        "outputs/demo-step1/commentary.csv",
        "outputs/demo-step1/video.csv",
        "outputs/demo-step1/label.csv"
    )
    
    demo_runner.run(args.start, args.end)