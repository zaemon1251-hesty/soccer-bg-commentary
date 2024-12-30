from tap import Tap
from loguru import logger
from datetime import datetime

from sn_providing.entity import CommentDataList, SpottingDataList, VideoData

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
            query_args["players"] = video_data_dict["players"]
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
