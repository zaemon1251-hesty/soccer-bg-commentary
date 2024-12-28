from pathlib import Path
import re
from typing import Dict, Optional, List
from tap import Tap
import json
import pandas as pd
from dataclasses import dataclass
from loguru import logger
from datetime import datetime

logger.add("logs/{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
logger.level("DEBUG")

class Arguments(Tap):
    game: str
    input_file: str
    output_file: str
    comment_csv: str
    video_data_csv: str = None
    spotting_csv: str = None
    
    sec_window_player: int = 2
    sec_window_action: int = 15


@dataclass
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


@dataclass
class CommentData:
    half: int
    start_time: int
    text: str
    category: str


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
            comment = CommentData(int(row["half"]), int(row["start"]), row["text"], str(row["付加的情報か"]))
            comments.append(comment)
        
        return CommentDataList(comments)

    @staticmethod
    def filter_by_half_and_time(
        comments: "CommentDataList", 
        half: int, 
        game_time: str, 
        seconds_before: int = 60
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
    
    def show_times(self, head: Optional[int] = None):
        head = head if head else len(self.comments)
        for s in self.comments[:head]:
            logger.info(f"{s.half=}, {s.start_time=}")


class VideoData:
    def __init__(
        self, 
        player_csv: Path, 
        spotting_csv: Path = None, 
        sec_window_player: int = 2, 
        sec_window_action: int = 15
    ):
        # パラメータを設定
        self.sec_window_player = sec_window_player
        self.sec_window = sec_window_action
        
        # データを読み込み
        self.player_df = pd.read_csv(player_csv)
        assert {"game", "half", "time", "team", "name"}.issubset(set(self.player_df.columns))
        
        if spotting_csv is not None:
            self.spotting_df = self._preprocess_spotting_df(pd.read_csv(spotting_csv))
            assert {"game", "half", "time", "label"}.issubset(set(self.spotting_df.columns))
            

    def get_data(self, game: str, half: int, game_time: int) -> dict[str, str]:
        result_dict = {
            "player_team_names": None,
            "actions": None
        }

        # self.sec_threshold 秒前 から self.sec_threshold 秒後の間に映っている選手名/teamを取得
        spot_players_df:pd.DataFrame = self.player_df.loc[
            (self.player_df["half"] == half) & \
            (self.player_df["game"] == game) & \
            (self.player_df["time"] >= game_time - self.sec_window_player) & \
            (self.player_df["time"] <= game_time + self.sec_window_player)
        ]
        # team name と player name をuniqeuに取得
        player_team_names = spot_players_df[['team', 'name']].drop_duplicates().to_dict(orient='records')
        
        if args.spotting_csv is None:
            result_dict["player_team_names"] = player_team_names
            return result_dict
        
        # sec_window 秒前までのアクションを取得
        spot_action_df: pd.DataFrame = self.spotting_df.loc[
            (self.spotting_df["half"] == half) & \
            (self.spotting_df["game"] == game) & \
            (self.spotting_df["time"] <= game_time) & \
            (self.spotting_df["time"] >= game_time - self.sec_window)
        ].sort_values("time", ascending=True)
        # action を取得
        actions = spot_action_df["label"].to_list()
        
        result_dict["player_team_names"] = player_team_names
        result_dict["actions"] = actions
        return result_dict
    
    @staticmethod
    def _preprocess_spotting_df(spotting_df: pd.DataFrame) -> pd.DataFrame:
        # 前処理
        spotting_df["half"] = spotting_df["gameTime"].str.split(" - ").str[0].astype(float)
        spotting_df["time"] = spotting_df["gameTime"].str.split(" - ").str[1].map(VideoData.gametime_to_seconds).astype(float)
        spotting_df["game"] = spotting_df["game"].str.rstrip("/")
        return spotting_df

    @staticmethod
    def gametime_to_seconds(gametime):
        if gametime.count(":") == 2:
            gametime = ":".join(gametime.split(":")[:2])
        m, s = gametime.split(":")
        return int(m) * 60 + int(s)

def build_query(
    comments: CommentDataList, 
    max_length: int = 256, 
    *args, **kwargs
) -> str:
    """
    検索クエリの内容:
    - 直近のコメント
    - TODO SoccerNet Caption Labels-caption.json のメタデータ(選手情報など)
    - TODO Game State Reconstruction の情報
    - TODO OSL Spotiing のAction Spotting の情報
    """
    query = []
    total_length = 0
    
    # commentsは時系列順に並んでいるので、逆順にして直近のコメントから取得
    for comment in reversed(comments.comments):
        if total_length + len(comment.text) > max_length:
            break
        query.append(comment.text)
        total_length += len(comment.text) + 1
    
    # 逆順にしていたので、再度逆順にして返す
    query = " ".join(reversed(query))
    query = "Previous comments: " + query
    
    # 映像中に映っている選手の名前を取得
    if kwargs.get("player_team_names"):
        team_game_str = ", ".join([f"{p['name']} from {p['team']}" for p in kwargs['player_team_names']])
        query = f"Players shown in this frame: {team_game_str}\n" + query

    # 試合情報を取得
    if kwargs.get("game_metadata"):
        game_data = kwargs["game_metadata"]
        game_query = f"Game: {game_data['league']} {game_data['league']} {game_data['date']} {game_data['home_team']} vs {game_data['away_team']}"
        query = game_query + "\n" + query
    
    # アクション情報を取得
    if actions := kwargs.get("actions"):
        action_str = ", ".join(actions) #たいていは高々一つのはず
        query = f"Recent Event: {action_str}\n" + query

    return query


def run(args: Arguments):
    # input_file includes json: {"UrlLocal": "path", "predictions", [{"gameTime": "1 - 00:24", "category": 0 or 1}, {...}, ...]}
    """
    1. Read files
    2. for each timestamp with label "1", 
        a. get previous comments corresponding to the timestamp 
        b. get player names and from sn-gamestate
        c. (optional) get actions from yahoo-research
    3. construct query
    """
    logger.info("Start constructing query")
    logger.info(f"{args=}")
    
    spotting_data_list = SpottingDataList.read_csv(args.input_file)    
    spotting_data_list.filter_by_category_1()
    
    video_data = VideoData(args.video_data_csv, args.spotting_csv)
    
    comment_data_list = CommentDataList.read_csv(args.comment_csv, args.game)
    
    logger.info("Spotting data")
    logger.info(f"{len(spotting_data_list.spottings)=}")
    logger.info("Comment data")
    logger.info(f"{len(comment_data_list.comments)=}")
    logger.info("Game data")
    logger.info(f"{spotting_data_list.game_metadata=}")
    
    result_spottings = []
    
    # spotting データのtimeリスト と video_data の timeリスト を比較する
    spot_time_set = set()
    for spotting_data in spotting_data_list.spottings:
        spot_time_set.add((spotting_data.half, spotting_data.game_time))
    
    if video_data is not None:
        frame_time_set = set()
        for i, data in video_data.player_df.iterrows():
            frame_time_set.add((data["half"], data["time"]))

    for spotting_data in spotting_data_list.spottings:
        filtered_comment_list = CommentDataList.filter_by_half_and_time(
            comment_data_list, 
            spotting_data.half, 
            spotting_data.game_time
        )
        
        query_args = {"comments": filtered_comment_list, "video_data": None, "game_metadata": spotting_data_list.game_metadata}
        
        if video_data is not None:
            video_data_dict = video_data.get_data(args.game, spotting_data.half, spotting_data.game_time)
            query_args["player_team_names"] = video_data_dict["player_team_names"]
            query_args["actions"] = video_data_dict["actions"]
        
        query = build_query(**query_args)
        spotting_data.query = query
        
        # reference があれば追加
        if ref := comment_data_list.get_comment_by_time(spotting_data.game_time):
            spotting_data.reference_text = ref
        result_spottings.append(spotting_data)

    spotting_data_list.spottings = result_spottings

    if args.output_file.endswith(".json"):
        spotting_data_list.to_json(args.output_file)
    elif args.output_file.endswith(".jsonl"):
        spotting_data_list.to_jsonline(args.output_file)
    
    logger.info(f"Output file is saved at {args.output_file}")

if __name__ == "__main__":
    ### construct query from the input file
    args = Arguments().parse_args()
    run(args)
