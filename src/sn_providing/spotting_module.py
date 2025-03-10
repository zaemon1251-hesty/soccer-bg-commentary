from __future__ import annotations
import polars as pl
import numpy as np
import json
from typing import Callable
from sklearn.metrics import confusion_matrix
from tap import Tap
from scipy.stats import lognorm, expon, gamma
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class SpottingArgment(Tap):
    path: str = (
        "data/spotting"  # spottingモジュール学習・評価に使うデータセットのパス (デモでは使わない)
    )
    fps: int = 1

    split: str = (
        "test"  # spottingモジュール学習・評価に使うデータセットのパス (デモでは使わない)
    )

    # タイミング生成
    timing_algo: str = "empirical"
    mean_silence_sec: float = (
        5.58  # 1秒以上の空白があるコメント集合における 平均的な発話間隔
    )
    lognorm_params: dict = {"shape": 2.2926, "loc": 0.0, "scale": 0.2688}
    gamma_params: dict = {"shape": 0.3283, "loc": 0.0, "scale": 6.4844}
    expon_params: dict = {"loc": 0.0, "scale": 2.1289}
    ignore_under_1sec: bool = False
    empirical_dist_csv: str = "data/demo/silence_distribution.csv"

    # ラベル生成
    default_rate: float = 0.185
    label_algo: str = "action_spotting"
    action_spotting_label_csv: str = "data/from_video/soccernet_spotting_labels.csv"
    action_rate_csv: str = "data/demo/Additional_Info_Ratios__Before_and_After.csv"
    action_window_size: float = 15
    addinfo_force: bool = False
    only_offplay: bool = False

    seed: int = 12

    def configure(self):
        self.add_argument("--lognorm_params", type=json.loads, required=False)
        self.add_argument("--gamma_params", type=json.loads, required=False)
        self.add_argument("--expon_params", type=json.loads, required=False)


def preprocess_action_df(spotting_df: pl.DataFrame) -> pl.DataFrame:
    # " - "で2分割（固定長2パーツ）
    split_cols = pl.col("gameTime").str.split_exact(" - ", 1)

    half_col = split_cols.struct.field("field_0").cast(pl.Float32).alias("half")
    game_time_str = split_cols.struct.field("field_1")

    # コロンの数をカウント
    colon_count = game_time_str.str.count_matches(":")

    # "HH:MM:SS"用にコロン2つの場合のsplit(3パーツ)
    three_parts = game_time_str.str.split_exact(":", 2)
    # "MM:SS"用にコロン1つの場合のsplit(2パーツ)
    two_parts = game_time_str.str.split_exact(":", 1)

    # colon_countに基づいて時間を計算
    time_col = (
        pl.when(colon_count == 2)
        .then(  # HH:MM:SS
            (three_parts.struct.field("field_1").cast(pl.Int32) * 60)
            + three_parts.struct.field("field_2").cast(pl.Int32)
        )
        .otherwise(  # MM:SS
            (two_parts.struct.field("field_0").cast(pl.Int32) * 60)
            + two_parts.struct.field("field_1").cast(pl.Int32)
        )
        .alias("time")
    )

    return spotting_df.with_columns(
        [half_col, time_col, pl.col("game").str.strip_chars_end("/").alias("game")]
    )


def to_gametime(half, seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    milsec = (seconds - int(seconds)) * 100
    return f"{int(half)} - {int(minutes):02d}:{int(seconds):02d}.{int(milsec):02d}"


class SpottingModule:
    def __init__(self, args: SpottingArgment, rng=np.random.default_rng()):
        self.args = args
        self.label_space = [0, 1]  # 映像の説明, 付加的情報
        self.label_prob = [
            1 - args.default_rate,
            args.default_rate,
        ]  # 全体のラベル割合分布

        self.mean_silence_sec = args.mean_silence_sec
        self.timing_algo = args.timing_algo
        self.label_algo = args.label_algo

        if args.action_spotting_label_csv is not None:
            self.action_df = pl.read_csv(args.action_spotting_label_csv)
            self.action_df = preprocess_action_df(self.action_df)
        else:
            self.action_df = None

        if args.action_rate_csv is not None:
            self.action_rate_df = pl.read_csv(args.action_rate_csv)
        else:
            self.action_rate_df = None

        self.action_window_size = args.action_window_size
        self.addinfo_force = args.addinfo_force
        self.offplay_events_a = [
            "Foul",
            "Goal",
            "Penalty",
            "Red card",
            "Yellow card",
            "Yellow->red card",
            "Substitution",
            "Offside",
            "Ball out of play",
        ]
        self.offplay_events_b = ["Kick-off"]
        self.only_offplay = args.only_offplay

        self.lognorm_params = args.lognorm_params
        self.gamma_params = args.gamma_params
        self.expon_params = args.expon_params

        self.rng = rng

        if args.timing_algo == "empirical":
            self.silence_dist = pl.read_csv(args.empirical_dist_csv)
            assert {"duration", "p"}.issubset(self.silence_dist.columns)
            # 4秒以内になるように、分布をcutする TODO CSV側を修正するべきだが、めんどくさいのでここでやる
            self.silence_dist = self.silence_dist.filter(pl.col("duration") <= 4)
            # 合計で値を割って、和が1になるようにする
            normalize_val = self.silence_dist["p"].sum()
            self.silence_dist = self.silence_dist.with_columns(
                pl.col("p") / normalize_val
            )

    def __call__(self, previous_t, game=None, half=None, target_ts=None):
        if target_ts is not None:
            # teacher forcing for debug label prediciton
            next_ts = target_ts
        else:
            next_ts = self._next_ts(previous_t)

        next_label = self._next_label(game, half, next_ts)
        return (next_ts, next_label)

    def _next_ts(self, previous_t):
        if self.timing_algo == "constant":
            next_ts = previous_t + self.mean_silence_sec
        elif self.timing_algo == "lognorm":
            next_ts = (
                lognorm.rvs(
                    s=self.lognorm_params["shape"],
                    loc=self.lognorm_params["loc"],
                    scale=self.lognorm_params["scale"],
                    random_state=self.rng,
                )
                + previous_t
            )
        elif self.timing_algo == "gamma":
            next_ts = (
                gamma.rvs(
                    self.gamma_params["shape"],
                    scale=self.gamma_params["scale"],
                    loc=self.gamma_params["loc"],
                    random_state=self.rng,
                )
                + previous_t
            )
        elif self.timing_algo == "expon":
            next_ts = (
                expon.rvs(
                    scale=self.expon_params["scale"],
                    loc=self.expon_params["loc"],
                    random_state=self.rng,
                )
                + previous_t
            )
        elif self.timing_algo == "empirical":
            delta_t = self.rng.choice(
                self.silence_dist["duration"].to_numpy(),
                p=self.silence_dist["p"].to_numpy(),
            )
            next_ts = previous_t + delta_t
        else:
            raise ValueError(f"{self.timing_algo} is not supported")
        return next_ts

    def _next_label(self, game, half, next_t):
        # game, half, next_t が入力として必要
        assert isinstance(game, str)
        assert half in [1, 2]
        assert isinstance(next_t, int) or isinstance(next_t, float)

        if self.label_algo == "constant":
            next_label = np.random.choice(self.label_space, p=self.label_prob)
            return next_label

        assert self.label_algo == "action_spotting"

        label_result = self.action_df.filter(
            (self.action_df["game"] == game)
            & (self.action_df["half"] == half)
            & (
                self.action_df["time"] <= next_t + self.action_window_size * 0
            )  # 発話タイミングの未来のアクションは考慮しない
            & (self.action_df["time"] >= next_t - self.action_window_size * 1)
        )

        if len(label_result) == 0:
            label_prob = self.label_prob
            next_label = self.rng.choice(self.label_space, p=self.label_prob)
            return next_label

        # polars 行アクセス
        # 最も self.action_df["time"]とnext_tが近い行を取得
        nearest_action_row = label_result.row(0, named=True)
        nearest_label = nearest_action_row["label"]
        is_before = next_t < nearest_action_row["time"]
        action_rate_result = self.action_rate_df.filter(
            (self.action_rate_df["label"] == nearest_label)
        )

        if nearest_label in self.offplay_events_b:
            # Kick-off は常に付加的情報
            next_label = 1
            return next_label

        # 未来のアクションは考慮しない
        # out of play eventのみ
        if (
            is_before
            or len(action_rate_result) == 0
            or (self.only_offplay and nearest_label not in self.offplay_events_a)
        ):
            next_label = self.rng.choice(self.label_space, p=self.label_prob)
            return next_label

        # labelの 前(rate_before) or 後(rate_after) の付加的情報の割合
        col_suffix = "before" if is_before else "after"
        addinfo_rate = action_rate_result.row(0, named=True)[f"rate_{col_suffix}"]
        assert 0 <= addinfo_rate <= 1

        # 付加的情報の割合が高い(アクション,前後)場合、付加的情報とする
        if self.addinfo_force and addinfo_rate > 0.18:
            addinfo_rate = 1.0
        elif self.addinfo_force and addinfo_rate <= 0.18:
            pass

        # ラベルを生成
        label_prob = [1 - addinfo_rate, addinfo_rate]
        next_label = self.rng.choice(self.label_space, p=label_prob)
        return next_label


def evaluate_diff_and_label(dataset, predict_model: Callable):
    result_dict = {
        "metadata": {
            "predict_ts": [],
            "target_ts": [],
            "predict_label": [],
            "predict_label_w_gold_timing": [],
            "target_label": [],
            "diff": [],
        },
        "content": [],
    }
    for game, half, previous_ts, target_ts, target_label in dataset:
        if args.ignore_under_1sec and (target_ts - previous_ts < 1):
            continue

        predict_ts, predict_label = predict_model(previous_ts, game, half)

        _, predict_label_w_gold_timing = predict_model(
            previous_ts, game, half, target_ts=target_ts
        )

        diff = (predict_ts - target_ts) ** 2

        result_dict["metadata"]["predict_ts"].append(predict_ts)
        result_dict["metadata"]["target_ts"].append(target_ts)
        result_dict["metadata"]["predict_label"].append(int(predict_label))
        result_dict["metadata"]["predict_label_w_gold_timing"].append(
            int(predict_label_w_gold_timing)
        )
        result_dict["metadata"]["target_label"].append(int(target_label))
        result_dict["metadata"]["diff"].append(int(diff))

        result_dict["content"].append(
            {
                "game": game,
                "half": half,
                "previsou_ts": previous_ts,
                "previous_end_time": to_gametime(half, previous_ts),
                "predict_start_time": to_gametime(half, predict_ts),
                "predict_label": (
                    "付加的情報の提供" if int(predict_label) == 1 else "映像の説明"
                ),
            }
        )
    result_dict["content"] = sorted(
        result_dict["content"],
        key=lambda x: (x["game"], x["half"], x["previsou_ts"]),
    )

    # timing: calculate diff average
    diff_average = np.mean(result_dict["metadata"]["diff"])
    print(f"diff_average: {diff_average}")

    # label: confusion matrix
    print("label evaluation")
    tn, fp, fn, tp = confusion_matrix(
        result_dict["metadata"]["target_label"],
        result_dict["metadata"]["predict_label"],
    ).ravel()
    print(f"confusion matrix: {tn=} {fp=} {fn=} {tp=}")
    # calculate label F1
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    f1_score = 2 * pr * re / (pr + re)
    print(f"label_accuracy: {(tp + tn) / (tp + tn + fp + fn):.3%}")
    print(f"label_precision: {pr:.3%}")
    print(f"label_recall: {re:.3%}")
    print(f"label_f1: {f1_score:.3%}")

    # label w/ gold timing: confusion matrix
    print("label w/ gold timing evaluation")
    tn, fp, fn, tp = confusion_matrix(
        result_dict["metadata"]["target_label"],
        result_dict["metadata"]["predict_label_w_gold_timing"],
    ).ravel()
    print(f"confusion matrix: {tn=} {fp=} {fn=} {tp=}")
    # calculate label F1
    pr = tp / (tp + fp)
    re = tp / (tp + fn)
    f1_score = 2 * pr * re / (pr + re)
    print(f"label_accuracy: {(tp + tn) / (tp + tn + fp + fn):.3%}")
    print(f"label_precision: {pr:.3%}")
    print(f"label_recall: {re:.3%}")
    print(f"label_f1: {f1_score:.3%}")

    # save result dict json
    del result_dict["metadata"]
    result_dict["evaluation"] = {
        "diff_average": float(diff_average),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "label_accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "label_precision": float(pr),
        "label_recall": float(re),
        "label_f1": float(f1_score),
    }

    # 保存する
    with open(
        f"logs/spotting-result-{args.label_algo}-{args.timing_algo}.json", "w"
    ) as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)
    return


class CommentaryClipsForDiffEstimation(Dataset):
    """
    直前の発話終了(開始)時間を考慮したデータセット
    """

    def __init__(
        self,
        path,
        split="train",
        prev_ts_col="target_frameid",
        ts_col="target_frameid",
        label_col="category",
    ):
        assert isinstance(split, str), "split should be a string"
        assert split in [
            "train",
            "valid",
            "test",
        ], "split should be either 'train' or 'valid'"

        self.path = path

        label_template = "{split}.csv"

        self.label_df = pd.read_csv(
            os.path.join(self.path, label_template.format(split=split))
        )

        # TODO データ準備の時点で対処すべき
        self.label_df = self.label_df.dropna(subset=["target_label"])

        assert "target_label" in self.label_df.columns

        # TODO データ準備の時点で対処すべき
        list_games = self.label_df["game"].unique().tolist()

        self.listGames = list_games

        self.num_classes = 2  # 映像の説明, 付加的情報

        self.data = []

        for game in tqdm(self.listGames):
            # filter labels by game
            label_df_game: pd.DataFrame = self.label_df[
                self.label_df["game"] == game
            ].sort_values("start")
            label_df_game.reset_index(inplace=True, drop=True)

            # 直前の発話開始フレームを self.prev_data に入れて
            # 現在の発話開始フレーム, class を self.current_data に入れる
            for i, row in label_df_game.iterrows():
                # 最初の行は無視
                if i < 1:
                    continue
                previous_ts = label_df_game.iloc[i - 1][prev_ts_col]
                target_ts = row[ts_col]
                target_label = row[label_col]
                half = row["half"]

                self.data.append((game, half, previous_ts, target_ts, target_label))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            game (str): game 名
            half (int): 1 or 2
            previous_ts[index-1] (int): 直前の発話開始時間
            target_ts[index] (int): 現在(ターゲット)の発話開始時間
            target_label[index] (int): 現在(ターゲット)の発話クラス
        """
        # self.data[index] = [game, half, previous_ts, target_ts, target_label]
        return self.data[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    args = SpottingArgment().parse_args()

    rng = np.random.default_rng(args.seed)

    dataset = CommentaryClipsForDiffEstimation(
        path=args.path,
        split=args.split,
        prev_ts_col="end",
        ts_col="start",
        label_col="付加的情報か",
    )

    print(f"len(dataset_Test): {len(dataset)}")

    spotting_model = SpottingModule(args, rng=rng)

    # 簡単な調査として、action_df の game と dataset の game がどれだけ一致しているかを調べる
    print(
        "action_df と datasetの game が一致してる数:"
        f" {len(set(spotting_model.action_df['game'].to_list()) & set(dataset.listGames))}"
    )

    evaluate_diff_and_label(dataset, spotting_model.__call__)
