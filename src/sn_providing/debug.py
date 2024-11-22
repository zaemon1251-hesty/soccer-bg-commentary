from pathlib import Path
import json

template = "time=> {time_str}\nprevious-comments=> {query}\nreference=> {reference}\nsystem_output=> {addiofo}\n"

# デバッグ用
if __name__ == "__main__":
    output_dir = Path("outputs")

    target_game = "europe_uefa-champions-league/2014-2015/2015-05-05 - 21-45 Juventus 2 - 1 Real Madrid"
    results_path = output_dir / target_game / "2024-11-22-14-48-results_addinfo_retrieval.jsonl"

    with open(results_path, "r") as f:
        print("game: ", target_game)
        for line in f.readlines():
            data = json.loads(line)
            half = data["half"]
            minute = data["game_time"] // 60
            second = data["game_time"] % 60
            
            data["time_str"] = f"{half} - {minute:02}:{second:02}"
            data["query"] = data["query"].replace("Previous comments: ", "")
            filtered_data = {k: v for k, v in data.items() if k in ["time_str", "query", "reference", "addiofo"]}
            print(template.format(**data))
    print("done")