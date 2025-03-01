# 付加的情報の提供システム

## 前提

[uv](https://github.com/astral-sh/uv)でパッケージ管理しています。

```bash
# uv自体のインストール
# uv は初期実行時に仮想環境のセットアップ、依存関係のインストールを行うので、これ以上やることはありません。(多分)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## data

archives/system-data.zip には以下のデータが含まれています。

```
├── addinfo_retrieval  [3249 entries exceeds filelimit, not opening dir]
├── commentary
│   └── scbi-v2.csv
├── demo
│   ├── Action_Ratios_Before_and_After_15_Seconds.csv
│   ├── Action_and_Rates_Data.csv
│   ├── Additional_Info_Ratios__Before_and_After.csv
│   ├── Extracted_Action_Rates.csv
│   ├── Label_Algorithm_Averages__Rounded_to_3_Decimal_Places_.csv
│   ├── players_in_frames_sn_gamestate.csv
│   ├── sample_metadata.csv
│   ├── silence_distribution.csv
│   └── silence_distribution_over_1sec.csv
├── exist_targets.txt
├── from_video
│   ├── players_in_frames.csv
│   ├── players_in_frames_sn_gamestate.csv
│   └── soccernet_spotting_labels.csv
├── reference_documents
│   └── evaluation-samples.yaml
└── spotting
    ├── test.csv
    ├── train.csv
    └── valid.csv
```

このうち、重要なファイルを紹介します

- ラベル付き実況コメントデータ (commentary/)
- 外部知識 (addinfo_retrieval/)
  - [wikipediaから選手情報を収集](https://github.com/zaemon1251-hesty/sn-script/blob/dev/src/sn_script/download_articles.py)
  - [trafilaturaでテキスト抽出](https://github.com/zaemon1251-hesty/sn-script/blob/dev/src/sn_script/extract_text.py)
- 実況生成デモに用いる情報群 (from_video/, demo/)

## 使い方

```bash
# データのセットアップ
scripts/setup-data.sh

# 実況生成
scripts/demo/run.sh
```
