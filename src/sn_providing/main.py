"""
デモ動画を作成するためのスクリプト
"""

import os
import logging
from datetime import datetime
from traceback import print_exc
from typing import Literal, Optional, Tuple

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI as LangChainOpenAI
from tap import Tap
import pandas as pd
import numpy as np

from sn_providing.entity import (
    CommentDataList,
    SpottingDataList,
    VideoData,
    CommentData,
    SpottingData,
)
from sn_providing.util import format_docs, log_documents, log_prompt
from sn_providing.spotting_module import SpottingModule, SpottingArgment
from sn_providing.construct_query import build_query
from sn_providing.addinfo_retrieval import get_rag_chain, get_retriever
from sn_providing.play_by_play import PlayByPlayGenerator
from sn_providing.constants import (
    MODEL_CONFIG,
    EMBEDDING_CONFIG,
    SEARCH_CONFIG,
    PERSIST_LANGCHAIN_DIR,
    INSTRUCTION,
    INSTRUCTION_JA,
    COMMENT_TO_EVENT_TEXT_PROMPT,
    COMMENT_TO_EVENT_TEXT_PROMPT_JA,
)


# .env の環境変数を読み込む
load_dotenv()


# ===========
# Logger
# ===========
time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# フォーマッタ
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ログファイル出力
file_handler = logging.FileHandler(f"logs/main--{time_string}.log")
file_handler.setFormatter(formatter)

# 標準出力
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

# ログ+標準出力
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# ===========
# Main
# ===========
class MainArgument(Tap):
    """
    メインスクリプトの引数
    """

    mode: Literal["run", "reference"] = "run"
    lang: Literal["ja", "en"] = "en"
    input_method: Literal["manual", "csv"] = "manual"
    comment_csv: str = "data/commentary/scbi-v2.csv"
    video_csv: str = "data/demo/players_in_frames_sn_gamestate.csv"
    label_csv: str = "data/from_video/soccernet_spotting_labels.csv"
    # input_method == "manual"
    game: str = None
    half: int = None
    start: float = None
    end: float = None
    save_jsonl: str = "outputs/demo-step2/commentary.jsonl"
    save_srt: str = "outputs/demo-step2/commentary.srt"
    pbp_jsonl: Optional[str] = None
    # input_method == "csv"
    input_csv: Optional[str] = None
    output_base_dir: Optional[str] = "outputs/demo-step2"
    pbp_base_dir: Optional[str] = "data/demo/pbp"

    # その他
    default_rate: float = 0.18
    default_text_threshold: float = 1.0

    seed: int = 100


def get_utterance_length(utterance: str):
    # 200 wpm (早口)という設定で，発話の長さを計算
    # return 秒数
    words = utterance.split()
    word_count = len(words)
    time = word_count * (60.0 / 200.0)
    return time


def get_utterance_length_ja(
    utterance: str, base_time: float = 0.12, pause_time: float = 0.2
) -> float:
    """
    日本語テキストのおおよその発話時間（秒）を計算する

    各文字の発話に base_time 秒、句読点（「、」「。」）の後には追加で pause_time 秒のポーズを想定

    Parameters:
        utterance (str): 発話テキスト
        base_time (float, optional): 1文字あたりの発話時間（秒）。デフォルトは 0.12 秒 (o1の提案を採用)
        pause_time (float, optional): 句読点後の追加ポーズ時間（秒）。デフォルトは 0.2 秒

    Returns:
        float: 推定発話時間（秒）
    """
    # punctuation = "、。"
    # 句読点の数をカウント
    # punctuation_count = sum(1 for c in utterance if c in punctuation)

    # 各文字に対する時間 + 句読点に対する追加ポーズ
    total_time = len(utterance) * base_time  # + punctuation_count * pause_time
    return total_time


class DemoRunner:
    """
    デモ動画のための実況を生成する
    """

    def __init__(
        self,
        game: str,
        half: int,
        comment_csv: str,
        video_csv: str,
        label_csv: str,
        lang: str = "en",
        seed: int = 100,
        **kwargs,
    ):
        # スポッティングモジュール関連
        rng = np.random.RandomState(seed)

        spotting_params: SpottingArgment = SpottingArgment().parse_known_args()[0]
        spotting_params.default_rate = kwargs.get(
            "default_rate", spotting_params.default_rate
        )

        spotting_model = SpottingModule(spotting_params, rng=rng)

        # 選手同定+ragモジュール関連
        instruction = INSTRUCTION_JA if lang == "ja" else INSTRUCTION
        game_metadata = SpottingDataList.extract_data_from_game(game)
        gold_comment_data_list: CommentDataList = CommentDataList.read_csv(
            comment_csv, game
        )
        video_data = VideoData(video_csv, label_csv=label_csv)
        llm = LangChainOpenAI(**MODEL_CONFIG)
        retriever = get_retriever(
            "openai-embedding",
            langchain_store_dir=PERSIST_LANGCHAIN_DIR,
            embedding_config=EMBEDDING_CONFIG,
            search_config=SEARCH_CONFIG,
        )
        rag_chain, _, _ = get_rag_chain(
            retriever=retriever,
            llm=llm,
            log_documents=log_documents,
            log_prompt=log_prompt,
            format_docs=format_docs,
            instruction=instruction,
        )
        # メンバ変数
        self.llm = llm
        self.default_text_threshold = kwargs.get("default_text_threshold", 1.0)
        self.lang = lang
        self.rng = rng
        self.seed = seed
        self.game = game
        self.half = half
        self.gold_comment_data_list = (
            gold_comment_data_list  # 映像の説明のモック用・検索クエリを補強する用
        )
        self.video_data = video_data
        self.spotting_model = spotting_model
        self.game_metadata = game_metadata
        self.rag_chain = rag_chain
        self.func_utterance_length = (
            get_utterance_length_ja if lang == "ja" else get_utterance_length
        )

    def build_extended_query(self, comment_data_list: CommentDataList, time: float):
        """game, half, game_metadata, video_data は関数内で定義された変数を用いる"""
        filtered_comment_list = CommentDataList.filter_by_half_and_time(
            comment_data_list, self.half, time
        )
        video_data_dict = self.video_data.get_data(self.game, self.half, time)

        query_args = {
            "comments": filtered_comment_list,
            "game_metadata": self.game_metadata,
            "players": video_data_dict["players"],
            "actions": video_data_dict["actions"],
        }

        query = build_query(**query_args)
        return query

    def get_pbp_alternative_commentary(self, comment: str) -> str:
        # COMMENT_TO_EVENT_TEXT_PROMPT
        template = (
            COMMENT_TO_EVENT_TEXT_PROMPT_JA
            if self.lang == "ja"
            else COMMENT_TO_EVENT_TEXT_PROMPT
        )
        prompt = template.format(comment=comment)
        response = self.llm.generate([[prompt]])
        return response.generations[0][0].text

    def run(
        self,
        start: float,
        end: float,
        save_jsonl: str,
        save_srt: str,
        play_by_play_jsonl: Optional[str] = None,
    ):
        finish_time = end
        prev_end = start
        comment_data_list = self.initialize_comment_data_list(start)

        play_by_play_func = self.setup_play_by_play(play_by_play_jsonl, start)

        while True:
            next_ts, next_label = self.spotting_model(
                previous_t=prev_end, game=self.game, half=self.half
            )
            comment = self.generate_comment(
                next_ts, next_label, comment_data_list, play_by_play_func
            )

            if comment is None or comment_data_list.is_duplicate(comment):
                prev_end = self.adjust_prev_end(prev_end)
                continue

            next_start, next_end = self.calculate_comment_timing(next_ts, comment)
            self.add_comment_to_list(
                comment_data_list, next_start, next_end, comment, next_label
            )

            logger.info(
                f"s: {next_start:.02f}, e: {next_end:.02f}, label: {next_label}, comment: {comment}"
            )

            if finish_time <= next_end:
                break

            prev_end = next_end

        self.save_comments(comment_data_list, save_jsonl, save_srt, start)

    def initialize_comment_data_list(self, start: float) -> CommentDataList:
        """検索クエリ増強のため、指定した時間範囲より過去のコメントでコメント履歴を初期化する"""
        comment_data_list = CommentDataList([])
        for comment in self.gold_comment_data_list.comments:
            if comment.half == self.half and comment.end_time < start:
                comment_data_list.comments.append(comment)
        return comment_data_list

    def setup_play_by_play(self, play_by_play_jsonl: Optional[str], start: float):
        if play_by_play_jsonl and os.path.exists(play_by_play_jsonl):
            """映像の説明を生成する関数を設定
            play_by_play_jsonl が指定されている場合、田中さんのシステムによる映像の説明を利用する
            play_by_play_jsonl が指定されていない場合、現実の実況を参考にしつつ、映像の説明を生成する
            """
            self.play_by_play_generator = PlayByPlayGenerator(
                pbp_jsonl=play_by_play_jsonl,
                lang=self.lang,
                rng=self.rng,
                base_time=start,
                default_text_threshold=self.default_text_threshold,
            )
            return self.play_by_play_generator.generate
        else:
            return lambda next_ts: self.get_pbp_alternative_commentary(
                self.gold_comment_data_list.get_comment_nearest_time(next_ts)
            )

    def generate_comment(
        self,
        next_ts: float,
        next_label: int,
        comment_data_list: CommentDataList,
        play_by_play_func,
    ) -> Optional[str]:
        """次のタイムスタンプとラベルに基づいてコメントを生成する関数

        引数:
        next_ts (float): 次のタイムスタンプ
        next_label (int): 次のラベル (0または1)
        comment_data_list (CommentDataList): コメント履歴
        play_by_play_func (function): 映像の説明を生成する関数

        戻り値:
        Optional[str]: 生成されたコメント。ラベルが0の場合は映像の説明、ラベルが1の場合は付加的情報。
        """
        if next_label == 0:
            return play_by_play_func(next_ts)
        elif next_label == 1:
            query = self.build_extended_query(comment_data_list, next_ts)
            spot = SpottingData(
                game=self.game,
                half=self.half,
                category=next_label,
                game_time=next_ts,
                query=query,
                position=int(next_ts) * 1000,
                confidence=1.0,
            )
            return self.rag_chain.invoke(spot)
        else:
            raise RuntimeError(f"無効な発話ラベルです: {next_label}")

    def adjust_prev_end(self, prev_end: float) -> float:
        return prev_end + 0.5

    def calculate_comment_timing(
        self, next_ts: float, comment: str
    ) -> Tuple[float, float]:
        next_start = next_ts
        next_end = next_ts + self.func_utterance_length(comment)
        return next_start, next_end

    def add_comment_to_list(
        self,
        comment_data_list: CommentDataList,
        next_start: float,
        next_end: float,
        comment: str,
        next_label: int,
    ):
        comment_data_list.comments.append(
            CommentData(
                half=int(self.half),
                start_time=next_start,
                end_time=next_end,
                text=comment,
                category=int(next_label),
            )
        )

    def save_comments(
        self,
        comment_data_list: CommentDataList,
        save_jsonl: str,
        save_srt: str,
        start: float,
    ):
        for comment in self.gold_comment_data_list.comments:
            if comment in comment_data_list.comments:
                comment_data_list.comments.remove(comment)
        comment_data_list.to_jsonline(save_jsonl)
        comment_data_list.to_srt(save_srt, base_time=start)

    def reference(self, start: float, end: float, save_jsonl: str, save_srt: str):
        """現実の実況（参照例）を出力する"""
        comment_data_list = CommentDataList([])
        for comment in self.gold_comment_data_list.comments:
            if (
                comment.half == self.half
                and comment.start_time >= start
                and comment.start_time < end
            ):
                comment_data_list.comments.append(comment)
        comment_data_list.to_jsonline(save_jsonl)
        comment_data_list.to_srt(save_srt, base_time=start)


def run_commentary_generation_for_video(
    mode: Literal["run", "reference"],
    game: str,
    half: int,
    start: float,
    end: float,
    comment_csv: str,
    video_csv: str,
    label_csv: str,
    save_jsonl: str,
    save_srt: str,
    lang: str = "ja",
    seed: int = 100,
    play_by_play_jsonl: str = None,
):
    demo_runner = DemoRunner(
        game, half, comment_csv, video_csv, label_csv, lang=lang, seed=seed
    )
    try:
        if mode == "run":
            demo_runner.run(
                start, end, save_jsonl, save_srt, play_by_play_jsonl=play_by_play_jsonl
            )
        elif mode == "reference":
            demo_runner.reference(start, end, save_jsonl, save_srt)
        else:
            raise ValueError(f"無効なモードです: {mode}")
    except Exception:
        logger.error(f"エラーが出ました: {save_jsonl=}")
        print_exc()
        return


if __name__ == "__main__":
    args = MainArgument().parse_args()

    if args.input_method == "manual":
        assert (
            (args.game is not None)
            and (args.half is not None)
            and (args.start is not None)
            and (args.end is not None)
        )

        logger.info(
            f"実行中 => ゲーム: {args.game}, ハーフ: {args.half}, 開始時間: {args.start}, 終了時間: {args.end}, "
            f"JSONL保存先: {args.save_jsonl}, SRT保存先: {args.save_srt}"
        )
        run_commentary_generation_for_video(
            args.mode,
            args.game,
            args.half,
            args.start,
            args.end,
            args.comment_csv,
            args.video_csv,
            args.label_csv,
            args.save_jsonl,
            args.save_srt,
            lang=args.lang,
            seed=args.seed,
            play_by_play_jsonl=args.pbp_jsonl,
        )

    elif args.input_method == "csv":
        assert (args.input_csv is not None) and (args.output_base_dir is not None)

        # 複数動画 まとめて実況生成
        input_df = pd.read_csv(args.input_csv)
        assert {"id", "game", "half", "start", "end"}.issubset(set(input_df.columns))

        # 保存ファイル名
        save_basename = "commentary" if args.mode == "run" else "ref"

        for _, row in input_df.iterrows():
            pbp_jsonl = os.path.join(
                f"{args.pbp_base_dir}",
                f"{row['id']:04d}",
                f"play-by-play-{args.lang}.jsonl",
            )
            save_jsonl = os.path.join(
                f"{args.output_base_dir}",
                f"{row['id']:04d}",
                f"{save_basename}-full-{args.lang}.jsonl",
            )
            save_srt = os.path.join(
                f"{args.output_base_dir}",
                f"{row['id']:04d}",
                f"{save_basename}-full-{args.lang}.srt",
            )

            if (
                os.path.exists(save_jsonl)
                and os.path.exists(save_srt)
                and input(f"{row['id']} 上書きしますか？ [y/n] ").lower() != "y"
            ):
                logger.info(f"すでに存在するためSkip: {row['id']}")
                continue

            if not os.path.exists(pbp_jsonl):
                logger.info(f"Play by Play がないためSkip: {row['id']}")
                continue

            logger.info(
                f"RUN => Game: {row['game']}, Half: {row['half']}, Start: {row['start']}, End: {row['end']},"
                f"Save JSONL: {save_jsonl}, Save SRT: {save_srt}"
            )
            run_commentary_generation_for_video(
                args.mode,
                row["game"],
                row["half"],
                row["start"],
                row["end"],
                args.comment_csv,
                args.video_csv,
                args.label_csv,
                save_jsonl,
                save_srt,
                lang=args.lang,
                seed=args.seed,
                play_by_play_jsonl=pbp_jsonl,
            )

        logger.info("Finished")
    else:
        raise ValueError(f"無効な入力方法です: {args.input_method}")
