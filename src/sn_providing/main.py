import sys
import os

# スクリプトのあるディレクトリの絶対パスを取得
current_dir = os.path.dirname(os.path.abspath(__file__))

# TemporallyAwarePoolingディレクトリの絶対パスを作成
tap_path = os.path.join(
    current_dir, "..", "..", "sn-caption", "Benchmarks", "TemporallyAwarePooling", "src"
)

# sys.pathに追加
sys.path.append(tap_path)

from model import Video2Spot

print(Video2Spot.__name__)
