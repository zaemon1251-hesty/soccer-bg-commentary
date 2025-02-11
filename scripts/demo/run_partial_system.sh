# 1サンプルのデモ動画のつくるためのスクリプト
# * 映像の説明はモックを使う
# * 選手同定モジュールですでに選手名を書き出した済みの映像を対象とする
# * 出力はjsonlines形式
# * gsrのvisualization映像と時間的な対応が取れるようにしたのち，webvttに書き出して字幕として表示してみる

target_game="europe_uefa-champions-league/2016-2017/2017-04-12 - 21-45 Bayern Munich 1 - 2 Real Madrid"
target_half=2
target_start=718
target_end=748


uv run python src/sn_providing/main.py \
    --game "$target_game" \
    --half $target_half \
    --start $target_start  \
    --end $target_end
