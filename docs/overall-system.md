# システム全体の処理フロー・入出力仕様

## システム全体の処理フロー

まず，ユーザが決定するものは以下の通り

- game
- duration (s -> e, sとeは 整数[秒])

処理の流れは以下の通り

0. ユーザがgameとdurationを コマンドラインで入力する
1. スポッティングモジュールがgameに対応するスポッティングデータを生成する
2. 推定されたタイミングごとに 前後5秒含めて選手同定モジュールを実行する
3. 選手同定モジュールの出力をもとに，クエリを構築
4. クエリをもとに，付加的情報を含む実況コメントを生成する

## モジュールの詳細

## スポッティングモジュール

リポジトリ

- sn-caption

実行方法

```bash
# 1. スポッティングの実行:
python Benchmark/TemporallyAwarePooling/src/main_v2.py (args)
```

入力:

- game: str = `SoccerNet/england_epl/2016-2017/2016-08-14 - 18-00 Arsenal 3 - 4 Liverpool`
- half: int = 1
- start: int = 126 (2:06)
- end: itn = 256 (4:26)

出力 -> spotting jsonline

- game: str
- half: int
- start: int
- label: int = 0 or 1 (映像の説明 か 付加的情報)

## 選手同定モジュール

リポジトリ

- sn-script
- tracklab

実行方法

```bash
# 1. 対象の箇所の切り取り: 
cd (sn-script); python src/sn_script/video2images.py (args)
# 2. 選手同定の実行: 
cd (tracklab); python -m tracklab.main -cn soccernet (args)
# 3. 選手名への対応付け: 
cd (sn-script); python src/sn_script/player_identification/result2player.py (args)
```

入力:

- spotting jsonline

出力 -> player jsonline

- game: str
- half: int
- time: int
- team: str
- name: str

## RAGモジュール

リポジトリ

- sn-providing

実行方法

```bash
# 1. クエリ構築: 
python src/sn_providing/construct_query.py
# 2. コメント生成: 
python src/sn_providing/addinfo_retrieval.py
```

入力:

- spotting jsonline
- player jsonline

出力 -> comment jsonline

- game: str
- half: int
- start: int
- label: str
- generated_text: str
