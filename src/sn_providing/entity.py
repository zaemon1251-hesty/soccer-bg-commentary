from pathlib import Path
import re
import math
from typing import Dict, Optional, List
import json
import pandas as pd
from dataclasses import dataclass
from loguru import logger
import yaml
from typing import Literal


@dataclass(eq=True)
class SpottingData:
    half: int
    game_time: int
    confidence: float
    position: int
    category: str # 0(映像の説明) or 1(付加的情報)
    game: Optional[str] = None
    query: Optional[str] = None
    generated_text: Optional[List[str]] = None
    reference_text: Optional[str] = None
    sample_id: Optional[str] = None
    documents: Optional[str] = None
    player_names: Optional[str] = None
    actions: Optional[str] = None


@dataclass
class SpottingDataList:
    """スポッティングデータの配列"""
    spottings: List[SpottingData]
    game_metadata: Dict = None
    
    @classmethod
    def read_csv(cls, json_file: str) -> "SpottingDataList":
        data = json.load(open(json_file))

        spottings = []
        for d in data["predictions"]:
            minutes, seconds = map(int, d["gameTime"].split(" ")[-1].split(":"))
            spotting = SpottingData(
                half=int(d["half"]),
                game_time=minutes * 60 + seconds,
                confidence=float(d["confidence"]),
                position=int(d["position"]),
                category=str(d["category"])
            )
            spottings.append(spotting)

        game_metadata = SpottingDataList.extract_data_from_game(data["game"])
        
        return cls(spottings, game_metadata)
    
    def filter_by_category_1(self):
        self.spottings = [s for s in self.spottings if s.category == "1"]

    @classmethod
    def from_jsonline(cls, input_file: str):
        spottings = []
        with open(input_file) as f:
            for line in f:
                spottings.append(SpottingData(**json.loads(line)))
        return cls(spottings)

    def to_json(self, output_file: str):
        json.dump([s.__dict__ for s in self.spottings], open(output_file, "w"), ensure_ascii=False)    
    
    def to_jsonline(self, output_file: str):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for s in self.spottings:
                f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
    
    def show_times(self, head: Optional[int] = None):
        head = head if head else len(self.spottings)
        for s in self.spottings[:head]:
            logger.info(f"{s.half=}, {s.game_time=}")

    @staticmethod
    def extract_data_from_game(game: str):
        league, season, match_data = game.split('/')
        # regexで抽出する
        date = re.search(r'\d{4}-\d{2}-\d{2}', match_data).group()
        kickoff_time = re.search(r'\d{2}-\d{2}', match_data).group()

        # score は 空白 数字 - 数字 空白
        home_score = int(re.search(r' (\d) - \d ', match_data).group(1))
        away_score = int(re.search(r' \d - (\d) ', match_data).group(1))

        # home-team は、 kickoff_time の後から score の前まで
        home_team = re.search(r' \d{2}-\d{2} (.*) \d - \d ', match_data).group(1)
        # away-teamは、scoreの後から終わりまで
        away_team = re.search(r' \d - \d (.*)', match_data).group(1)

        return {
            "league": league,
            "season": season,
            "date": date,
            "kickoff_time": kickoff_time,
            "home_team": home_team,
            "home_score": home_score,
            "away_score": away_score,
            "away_team": away_team,
        }


@dataclass(eq=True)
class CommentData:
    half: int
    start_time: int | float
    text: str
    category: str
    end_time: Optional[int | float] = None

@dataclass
class CommentDataList:
    comments: List[CommentData]
    
    @staticmethod
    def read_csv(comment_csv: str, game: str) -> "CommentDataList":
        comment_df = pd.read_csv(comment_csv)
        assert set(comment_df.columns) >= {"game", "half", "start", "end", "text", "付加的情報か"}
        
        # TODO 前処理はmethod分割したい
        
        def convert_time(time: str) -> int:
            minute, second = map(int, time.split(":"))
            return minute * 60 + second
        
        # 指定のgame に対応するコメントのみ取得
        comment_df = comment_df[comment_df["game"] == game]

        # start time を秒に変換
        if comment_df["start"].dtype == "O":
            comment_df["start"] = comment_df["start"].apply(convert_time)
        
        # 並び替え
        comment_df = comment_df.sort_values("start")
        comments = []
        for i, row in comment_df.iterrows():
            comment = CommentData(int(row["half"]), int(row["start"]), row["text"], str(row["付加的情報か"]),end_time=int(row["end"]))
            comments.append(comment)
        
        return CommentDataList(comments)

    @staticmethod
    def filter_by_half_and_time(
        comments: "CommentDataList", 
        half: int, 
        game_time: str, 
        seconds_before: int = 20
    ) -> "CommentDataList":
        """
        seconds_before 秒前から game_time までのコメントを取得
        """
        filterd_comments = [
            c for c in comments.comments
            if c.half == half and \
                c.start_time >= game_time - seconds_before and \
                c.start_time < game_time
            ]
        return CommentDataList(filterd_comments)
    
    def get_comment_by_time(self, game_time: int) -> str:
        for comment in self.comments:
            if comment.start_time == game_time:
                return comment.text
        return None

    def get_comment_nearest_time(self, game_time: int, thres: float = 15.,category: int = 0) -> str:
        diff = float("inf")
        nearest_comment = None
        for comment in self.comments:
            if abs(comment.start_time - game_time) < min(diff, thres) and comment.category == category:
                diff = abs(comment.start_time - game_time)
                nearest_comment = comment.text
        return nearest_comment
    
    def show_times(self, head: Optional[int] = None):
        head = head if head else len(self.comments)
        for s in self.comments[:head]:
            logger.info(f"{s.half=}, {s.start_time=}")

    def to_json(self, output_file: str):
        json.dump([s.__dict__ for s in self.comments], open(output_file, "w"), ensure_ascii=False)    
    
    def to_jsonline(self, output_file: str):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for s in self.comments:
                f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
    
    def to_srt(self, output_file: str, base_time: float = 0.0, video_end_time: float = 30.0):
        """
        start_time, end_time は float で保持
        base_time を引くことで、動画の任意のクリップに合わせた字幕に対応可能
        例:
        - base_time=120.0 なら、srtの「00:00:00.000」は実際には，もとの動画の120秒地点。
        """
        # TODO ひとつあたりのコメントの長さが長い場合に対応
        # modified_comments = []
        # for comment in self.comments:...
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        def format_vtt_time(sec: float) -> str:
            """
            秒(float)を WebVTT 形式 (HH:MM:SS.mmm) に変換
            """
            hours = int(sec // 3600)
            minutes = int((sec % 3600) // 60)
            seconds = sec % 60
            milliseconds = int((seconds - math.floor(seconds)) * 1000)
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i, s in enumerate(self.comments, start=1):
                start_sec = float(s.start_time) - base_time
                end_sec = float(s.end_time) - base_time
                end_sec = min(end_sec, video_end_time)

                start_vtt = format_vtt_time(start_sec)
                end_vtt = format_vtt_time(end_sec)
                f.write(f"{i}\n")
                f.write(f"{start_vtt} --> {end_vtt}\n")
                f.write(f"{s.text}\n\n")


class VideoData:
    def __init__(
        self, 
        player_csv: Path, 
        label_csv: Path = None, # イベント カメラ　など
        sec_window_player: int = 2, 
        sec_window_action: int = 15
    ):
        # パラメータを設定
        self.sec_window_player = sec_window_player
        self.sec_window = sec_window_action
        
        # データを読み込み
        self.player_df = pd.read_csv(player_csv)
        assert {"game", "half", "time", "team", "name", "jersey_number"}.issubset(set(self.player_df.columns))
        self.player_df["time"] = self.player_df["time"].apply(self.gametime_to_seconds)
        
        self.label_df = None
        if label_csv is not None:
            self.label_df = self._preprocess_label_df(pd.read_csv(label_csv))
            assert {"game", "half", "time", "label"}.issubset(set(self.label_df.columns))
            
    def get_data(self, game: str, half: int, game_time: int) -> dict[str, str]:
        result_dict = {
            "players": None,
            "actions": None
        }

        # self.sec_threshold 秒前 から self.sec_threshold 秒後の間に映っている選手名/teamを取得
        spot_players_df:pd.DataFrame = self.player_df.loc[
            (self.player_df["half"] == half) & \
            (self.player_df["game"] == game) & \
            (self.player_df["time"] >= game_time - self.sec_window_player) & \
            (self.player_df["time"] <= game_time + self.sec_window_player)
        ]
        # team name と player name, jersey number を unique に取得
        player_dict = (
            spot_players_df[['name', 'team', 'jersey_number']]
            .drop_duplicates()
            .to_dict(orient='records')
        )
        
        if self.label_df is None:
            result_dict["players"] = player_dict
            return result_dict
        
        # sec_window 秒前までのアクションを取得
        spot_action_df: pd.DataFrame = self.label_df.loc[
            (self.label_df["half"] == half) & \
            (self.label_df["game"] == game) & \
            (self.label_df["time"] <= game_time) & \
            (self.label_df["time"] >= game_time - self.sec_window)
        ].sort_values("time", ascending=True)
        # action を取得
        actions = spot_action_df["label"].to_list()
        
        result_dict["players"] = player_dict
        result_dict["actions"] = actions
        return result_dict
    
    def show_player_data(self, game: str, half: int, game_time: int):
        result_dict = self.get_data(game, half, game_time)
        player_dict = result_dict["players"]
        tmp_player_df = pd.DataFrame(player_dict)
        # csv 形式で表示
        print(tmp_player_df.to_csv(index=False))

    @staticmethod
    def _preprocess_label_df(label_df: pd.DataFrame) -> pd.DataFrame:
        # 前処理
        label_df["half"] = label_df["gameTime"].str.split(" - ").str[0].astype(float)
        label_df["time"] = label_df["gameTime"].str.split(" - ").str[1].map(VideoData.gametime_to_seconds).astype(float)
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


# 正解文書のデータ
@dataclass
class ReferenceDoc:
    id: str
    game: str
    half: str
    time: str
    content: str
    
    @staticmethod
    def get_list_from_yaml(reference_documents_yaml: str):
        with open(reference_documents_yaml, encoding='utf-8') as file:
            reference_doc_data = yaml.safe_load(file)["samples"]
            reference_doc_list: list[ReferenceDoc] = [
                ReferenceDoc(v["id"], v["game"], v["half"], v["time"], v["content"]) for v in reference_doc_data
            ]
        return reference_doc_list

    @staticmethod
    def get_reference_documents(game, half, time, reference_documents: list["ReferenceDoc"]):
        target_dcument = None
        for doc_data in reference_documents:
            if doc_data.game == game and doc_data.half == half and doc_data.time == time:
                target_dcument = doc_data.content
                logger.info(f"Match Reference Document Sample id: {doc_data.id}")
                break
        return target_dcument

    @staticmethod
    def get_reference_document_entity(game, half, time, reference_documents: list["ReferenceDoc"]):
        target_doc = None
        for doc_data in reference_documents:
            if doc_data.game == game and doc_data.half == half and doc_data.time == time:
                target_doc = doc_data
                logger.info(f"Match Reference Document Sample id: {doc_data.id}")
                break
        return target_doc


# 型エイリアス 文書スコアの算出方法方法
RetrieverType = Literal["tfidf", "openai-embedding"]
