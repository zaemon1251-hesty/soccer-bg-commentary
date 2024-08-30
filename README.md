# 付加的情報の提供システム

## 前提

[uv](https://github.com/astral-sh/uv)でパッケージ管理しています。

```bash
# uv自体のインストール
# uv は初期実行時に仮想環境のセットアップ、依存関係のインストールを行うので、これ以上やることはありません。(多分)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## data

archives/data.zip には以下のデータが含まれています。

- スポッティング結果 (json file)
  - half: # 1(前半) or 2(後半)
  - category: # 0(映像の説明) or 1(付加的情報)
  - gameTime: # 発話タイミング(フォーマット -> "half - mm:ss")
  - etc...
- ラベル付き実況コメントデータ (csv file)
- 知識ベース (txt files)
  - [wikipediaから選手情報を収集](https://github.com/zaemon1251-hesty/sn-script/blob/dev/src/sn_script/download_articles.py)
  - [trafilaturaでテキスト抽出](https://github.com/zaemon1251-hesty/sn-script/blob/dev/src/sn_script/extract_text.py)
- (example_data -> llamaindexのサンプルデータ)

## usage

```bash
# 1 prepare input data and fix paths in the scripts
scripts/setup_data.sh

# 2 construct query for retrieving additional information
scripts/construct_query_comments.sh

# 3 generate candidates of additional information
scripts/addinfo_retrieval.sh
```
