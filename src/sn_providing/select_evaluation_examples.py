import json
import os
import sys
from pathlib import Path

from tap import Tap


player_prompt = "Players shown in this frame"


class Argument(Tap):
    query_json_dir: str
    exist_target_txt: str
    jsonl_filename: str
    output_dir: str
    output_basename: str = "evaluation-target"


def main(args: Argument):
    with open(args.exist_target_txt) as f:
        exist_target_list = f.read().split("\n")
    
    all_json_data = []
    for game in exist_target_list:
        json_path = Path(args.query_json_dir) / game / args.jsonl_filename
        if not json_path.exists():
            print(json_path)
            continue
        with open(json_path) as f:
            for line in f.readlines():
                data = json.loads(line)
                data["game"] = game
                all_json_data.append(data)
    print(f"All data len: {len(all_json_data)}")
    
    filtered_json_data = []
    for data in all_json_data:
        if player_prompt in data["query"]:
            filtered_json_data.append(data)
    print(f"Filtered data len: {len(filtered_json_data)}")
    
    output_path = Path(args.output_dir) / f"{args.output_basename}.jsonl"
    with open(output_path, "w") as f:
        for data in filtered_json_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    args = Argument().parse_args()
    main(args)