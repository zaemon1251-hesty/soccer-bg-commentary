import numpy as np
import pandas as pd
import json
from typing import Dict, List


location_map_en = {
    "Right center midfield": "the middle third",
    "Left center midfield": "the middle third",
    "Left top box": "the attacking third",
    "Right bottom box": "the defensive third",
    "Left bottom corner": "the defensive third",
    "Right top box": "the attacking third",
    "Left bottom midfield": "the defensive third",
    "Left top midfield": "the middle third",
}


location_map_jp = {
    "Right center midfield": "中盤",
    "Left center midfield": "中盤",
    "Left top box": "敵陣",
    "Right bottom box": "自陣",
    "Left bottom corner": "自陣",
    "Right top box": "敵陣",
    "Left bottom midfield": "自陣",
    "Left top midfield": "中盤",
}


# --- 3) イベント(action) × 発話長さ(time_length)ごとのテンプレート ---
commentary_data_en = {
    "PASS": {
        "very_short": [
            "Passing.",
            "{player}!"
        ],
        "short": [
            "{player} passes the ball!",
            "A crisp pass from {player}!"
        ],
        "mid": [
            "{player} from {team} delivers a neat pass at {location}!",
            "Smooth ball movement—{player} keeps possession around {location}."
        ]
    },
    "DRIVE": {
        "very_short": [
            "{player} dribbles.",
        ],
        "short": [
            "{player} driving into attacking Area!",
            "A strong run by {player}!"
        ],
        "mid": [
            "{player} of {team} pushes ahead through {location}!",
            "Powerful dribble by {player} around {location}, upping the tempo!"
        ]
    },
    "OUT": {
        # ボールがアウトになった、またはプレーが中断されたイメージ
        "very_short": [
            "Out!",
            "Line crossed!"
        ],
        "short": [
            "The ball goes out of play!",
            "{player} sends it over the line!"
        ],
        "mid": [
            "Out of bounds—play stops momentarily!",
        ]
    },
    "THROW IN": {
        "very_short": [
            "Throw-in!",
            "Restart!"
        ],
        "short": [
            "{player} takes a throw-in!",
            "Quick throw by {player}!"
        ],
        "mid": [
            "{player} takes a throw-in!",
        ]
    }
}


commentary_data_jp = {
    "PASS": {
        "very_short": [
            "パス！",
            "{player}！"
        ],
        "short": [
            "{player}のパス！",
            "素早いパス！"
        ],
        "mid": [
            "{team}の{player}、{location}で正確なパスを通します！",
            "ボールを繋ぐ{player}、{location}で落ち着いたパスワーク！"
        ]
    },
    "DRIVE": {
        "very_short": [
            "運ぶ。",
            "{player}！"
        ],
        "short": [
            "{player}が前に運ぶ！",
            "力強いドリブル！"
        ],
        "mid": [
            "{player}、{location}を突破して攻撃を加速！",
            "{player}が{location}を切り裂くように進みます！"
        ]
    },
    "OUT": {
        "very_short": [
            "ラインを割った！"
        ],
        "short": [
            "ボールが外に出ました！",
        ],
        "mid": [
            "ボールがラインを割り、一旦プレーはストップします！",
            "{player}が{location}でボールを外に出しました、仕切り直しです！"
        ]
    },
    "THROW IN": {
        "very_short": [
            "スローイン！",
        ],
        "short": [
            "{player}がスローイン！",
            "素早いスローイン！"
        ],
        "mid": [
            "{player}がスローインを行います！",
        ]
    }
}


def map_time_to_length(seconds):
    if seconds < 2.0:
        return "very_short"
    elif seconds < 5.0:
        return "short"
    else:
        return "mid"


def convert_location(location_str, lang="en"):
    """
    指定の場所文字列（例: "Right top box"）を、
    lang="en" なら英語向け自然表現、
    lang="jp" なら日本語向け自然表現に変換する。
    """
    if lang == "en":
        return location_map_en.get(location_str, location_str)
    elif lang == "jp":
        return location_map_jp.get(location_str, location_str)
    else:
        return location_str  # 想定外の場合はそのまま返す


def generate_commentary(
    event_data: Dict[str, Dict[str, List[str]]],
    lang: str = "en", 
    time_length: str = "short", 
    rng: np.random.Generator = np.random.default_rng(),
    default_text_threshold: float = 1.0
):
    """
    event_data: {
        "action": str,      # イベント名 (例: "HIGH PASS", "DRIVE", etc.)
        "location": str,    # 生のロケーション文字列 (例: "Right top box")
        "name": str,        # 選手名
        "team": str,        # チーム名
        ... ほかのデータ
    }
    lang: "en" or "jp"
    time_length: "very_short", "short", "mid" など

    戻り値: str (実況コメント)
    """
    action = event_data.get("action", "")
    location_str = event_data.get("location", "")
    player = event_data.get("name", "")
    team = event_data.get("team", "")
    text = event_data.get("text", "")

    if rng.uniform() <= default_text_threshold:
        return text

    # 場所を自然な言語表現に変換
    converted_location = convert_location(location_str, lang=lang)

    try:
        if lang == "en":
            action_dict = commentary_data_en.get(action)
            phrases = action_dict.get(time_length)
            template: str = rng.choice(phrases)
            return template.format(player=player, team=team, location=converted_location)

        elif lang == "jp":
            action_dict = commentary_data_jp.get(action)
            phrases = action_dict.get(time_length)
            template: str = rng.choice(phrases)
            return template.format(player=player, team=team, location=converted_location)
    except:
        pass
    
    return text

class PlayByPlayGenerator:
    def __init__(self, pbp_jsonl: str, lang: str = "en", rng=None, base_time=0, default_text_threshold=1.0, spotting_csv: str = None):
        self.lang = lang
        self.rng = rng or np.random.default_rng()
        self.base_time = base_time
        self.default_text_threshold = default_text_threshold

        self.lengths = ["very_short", "short", "mid"]

        self.play_by_play_data = []
        with open(pbp_jsonl, "r") as f:
            for line in f:
                self.play_by_play_data.append(json.loads(line))

        self.play_by_play_df = pd.DataFrame(self.play_by_play_data)

        self.play_by_play_df = self.preprocess_data(self.play_by_play_df)
        
        if spotting_csv:
            self.spotting_df = pd.read_csv(spotting_csv)
            self.spotting_df = self._preprocess_label_df(self.spotting_df)
            self.spotting_df = self.spotting_df.sort_values(by="start_time")
    
    def preprocess_data(self, play_by_play_df: pd.DataFrame):
        assert {"start_time", "end_time", "action", "location", "name", "team"}.issubset(
            play_by_play_df.columns
        ), "カラムに不足があります。"
        # 数値文字列を数値に変換
        play_by_play_df["start_time"] = play_by_play_df["start_time"].astype(float)
        play_by_play_df["end_time"] = play_by_play_df["end_time"].astype(float)

        # start_time, end_time は ビデオの開始時間が基準になっているから、base_time (= demo runner の start [試合経過時間(秒)]) を足して調整
        play_by_play_df["start_time"] += self.base_time
        play_by_play_df["end_time"] += self.base_time

        play_by_play_df = play_by_play_df.sort_values(by="start_time")

        return play_by_play_df

    def generate(self, time: float):
        event_data = self.play_by_play_df.loc[self.play_by_play_df["start_time"] <= time]
        if event_data.empty:
            return None
        event_data = event_data.iloc[-1].to_dict()

        length = self.rng.choice(self.lengths)

        return generate_commentary(
            event_data, lang=self.lang, time_length=length, rng=self.rng, default_text_threshold=self.default_text_threshold
        )
    
    @staticmethod
    def _preprocess_label_df(label_df: pd.DataFrame) -> pd.DataFrame:
        # 前処理
        label_df["half"] = label_df["gameTime"].str.split(" - ").str[0].astype(float)
        label_df["time"] = label_df["gameTime"].str.split(" - ").str[1].map(PlayByPlayGenerator.gametime_to_seconds).astype(float)
        label_df["game"] = label_df["game"].str.rstrip("/")
        return label_df

    @staticmethod
    def gametime_to_seconds(gametime):
        if isinstance(gametime, int) or isinstance(gametime, float):
            return gametime
        
        if gametime.count(":") == 0:
            return float(gametime)
        
        if gametime.count(":") == 2:
            gametime = ":".join(gametime.split(":")[:2])
        
        m, s = gametime.split(":")
        
        return int(m) * 60 + int(s)

# --- test ---
if __name__ == "__main__":
    event_list = [
        {
            "start_time": 0.08,
            "end_time": 1.58,
            "action": "HIGH PASS",
            "location": "Right top box",
            "name": "Bartra M.",
            "team": "Dortmund"
        },
        {
            "start_time": 2.24,
            "end_time": 3.74,
            "action": "HIGH PASS",
            "location": "Left center midfield",
            "name": "Scurrle",
            "team": "Dortmund"
        },
        {
            "start_time": 7.52,
            "end_time": 8.76,
            "action": "DRIVE",
            "location": "Left center midfield",
            "name": "Weidenfeller R.",
            "team": "Dortmund"
        },
        {
            "start_time": 17.12,
            "end_time": 17.16,
            "action": "SHOT",
            "location": "Left top box",
            "name": "Schmelzer",
            "team": "Dortmund"
        },
        {
            "start_time": 17.36,
            "end_time": 18.12,
            "action": "BALL PLAYER BLOCK",
            "location": "Right bottom box",
            "name": "Burke",
            "team": "RB Leipzig"
        }
    ]

    print("=== English Commentary Samples ===")
    for e in event_list:
        commentary_en = generate_commentary(e, lang="en", time_length="mid")
        print(f"{e['start_time']}s - {e['end_time']}s:", commentary_en)

    print("\n=== Japanese Commentary Samples (long) ===")
    for e in event_list:
        commentary_jp = generate_commentary(e, lang="jp", time_length="long")
        print(f"{e['start_time']}s - {e['end_time']}s:", commentary_jp)
