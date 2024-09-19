from typing import Optional, List
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
    comment_csv_file_half1: str
    comment_csv_file_half2: str


@dataclass
class SpottingData:
    half: int
    game_time: int
    confidence: float
    position: int
    category: str # 0(映像の説明) or 1(付加的情報)
    query: Optional[str] = None
    addiofo: Optional[List[str]] = None


@dataclass
class SpottingDataList:
    spottings: List[SpottingData]
    
    @staticmethod
    def read_json(json_file: str) -> "SpottingDataList":
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

        return SpottingDataList(spottings)
    
    @staticmethod
    def filter_by_category_1(spottings: "SpottingDataList") -> "SpottingDataList":
        return SpottingDataList([s for s in spottings.spottings if s.category == "1"])

    @staticmethod
    def from_jsonline(input_file: str):
        spottings = []
        with open(input_file, 'r') as f:
            for line in f:
                spottings.append(SpottingData(**json.loads(line)))
        return SpottingDataList(spottings)

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
    def read_csv(csv_file_half1: str, csv_file_half2: str, game: str) -> "CommentDataList":
        comment_df_half1 = pd.read_csv(csv_file_half1)
        comment_df_half2 = pd.read_csv(csv_file_half2)
        
        comment_df_half1["half"] = 1
        comment_df_half2["half"] = 2
        
        # TODO 前処理はmethod分割したい
        
        def convert_time(time: str) -> int:
            minute, second = map(int, time.split(":"))
            return minute * 60 + second
        
        # 指定のgame に対応するコメントのみ取得
        comment_df_half1 = comment_df_half1[comment_df_half1["game"] == game]
        comment_df_half2 = comment_df_half2[comment_df_half2["game"] == game]

        # start time を秒に変換
        comment_df_half1["start"] = comment_df_half1["start"].apply(convert_time)
        comment_df_half2["start"] = comment_df_half2["start"].apply(convert_time)
        
        # 並び替え
        comment_df_half1 = comment_df_half1.sort_values("start")
        comment_df_half2 = comment_df_half2.sort_values("start")
        
        comment_df_all = pd.concat([comment_df_half1, comment_df_half2])
        
        comments = []
        for i, row in comment_df_all.iterrows():
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
                c.start_time <= game_time
            ]
        return CommentDataList(filterd_comments)
    
    def show_times(self, head: Optional[int] = None):
        head = head if head else len(self.comments)
        for s in self.comments[:head]:
            logger.info(f"{s.half=}, {s.start_time=}")



def build_query(
    comments: CommentDataList, 
    max_length: int = 256, 
    *args, **kargs
) -> str:
    """
    検索クエリの内容:
    - 直近のコメント
    - TODO SoccerNet Caption Labels-caption.json のメタデータ(選手情報など)
    - TODO Game State Reconstruction の情報
    - TODO Yahoo Research のAction Spotting の情報
    """
    query = ""
    
    # commentsは時系列順に並んでいるので、逆順にして直近のコメントから取得
    for comment in reversed(comments.comments):
        if len(query) > max_length:
            break
        query += comment.text + " "

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
    
    spotting_data_list = SpottingDataList.read_json(args.input_file)    
    spotting_data_list = SpottingDataList.filter_by_category_1(spotting_data_list)
    
    comment_data_list = CommentDataList.read_csv(args.comment_csv_file_half1, args.comment_csv_file_half2, args.game)
    
    logger.info("Spotting data")
    logger.info(f"{len(spotting_data_list.spottings)=}")
    spotting_data_list.show_times(4)
    logger.info("Comment data")
    logger.info(f"{len(comment_data_list.comments)=}")
    comment_data_list.show_times(16)
    
    result_spottings = []
    
    for spotting_data in spotting_data_list.spottings:
        filtered_comment_list = CommentDataList.filter_by_half_and_time(
            comment_data_list, 
            spotting_data.half, 
            spotting_data.game_time
        )
        query = build_query(comments=filtered_comment_list)
        spotting_data.query = query
        result_spottings.append(spotting_data)

    result_spottings = SpottingDataList(result_spottings)

    if args.output_file.endswith(".json"):
        result_spottings.to_json(args.output_file)
    elif args.output_file.endswith(".jsonl"):
        result_spottings.to_jsonline(args.output_file)
    
    logger.info(f"Output file is saved at {args.output_file}")

if __name__ == "__main__":
    ### construct query from the input file
    args = Arguments().parse_args()
    run(args)
