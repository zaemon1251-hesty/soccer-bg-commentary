# システム全体の処理フロー・入出力仕様

## システム全体の処理フロー

まず，ユーザが決定するものは以下の通り

- game
- duration (s -> e, sとeは mm:ss 形式)

処理の流れは以下の通り

0. ユーザがgameとdurationを コマンドラインで入力する
1. スポッティングモジュールがgameに対応するスポッティングデータを生成する
2. 推定されたタイミングごとに 前後5秒含めて選手同定モジュールを実行する
3. 選手同定モジュールの出力をもとに，クエリを構築
4. クエリをもとに，付加的情報を含む実況コメントを生成する

## 関連するプログラム

### スポッティングモジュール

リポジトリ: **sn-caption**
実行コマンド: `python Benchmark/TemporallyAwarePooling/src/main_v2.py (args)`

### 選手同定モジュール

リポジトリ: **tracklab**
実行コマンド: `python -m tracklab.main -cn soccernet (args)`

### RAGモジュール

リポジトリ: **sn-providing**
実行コマンド(クエリ構築): `python src/sn_providing/construct_query.py`
実行コマンド(コメント生成): `python src/sn_providing/addinfo_retrieval.py`

## 入出力仕様

### スポッティングモジュールの入出力

- 入力: game, duration
- 出力: json
  - metadata: dict
    - game: str
    - half: int
    - duration_start: str
    - duration_end: str
  - content: list of object
    - start: int
    - label: str

### 選手同定モジュールの入出力

- 入力: json
