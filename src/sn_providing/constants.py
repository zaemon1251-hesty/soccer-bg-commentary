from pathlib import Path

# モデルのデフォルトパラメータ
MODEL_CONFIG = {
    "model": "gpt-4o",
    "temperature": 0,
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-ada-002",
    "chunk_size": 1000,
}

SEARCH_CONFIG = {
    "k": 10,
    "score_threshold": 0.7,
}

INSTRUCTION_NO_RETRIEVAL = """You are a professional color commentator for a live broadcast of soccer. 
provide just one comment with a fact, such as player records or team statistics, relevant to the current soccer match. 
Example:
Comment: He has scored 5 goals in the last 3 games.
Comment: This team so far beaten 3-0 at home by the former champion.
Comment: His transfer fee was the highest in the club's history.
Comment: He was in injury time in the last game.
Attention:
The comment should be short, clear, accurate, and suitable for live commentary. 
The game date will be given as YYYY-MM-DD. Do not use information dated after this.
This comment should be natural comments following the previous comments given to the prompt.
Please generate only one comment and do not say anything else.
The comment should be between 6 and 18 words, or shorter."""


# 指示文 TODO yamlなどに分けて，プロンプトを選択できるようにしたら，実験が楽になりそう．．．
INSTRUCTION = """You are a professional color commentator for a live broadcast of soccer. 
Using the documents below, 
provide just one comment with a fact, such as player records or team statistics, relevant to the current soccer match. 
Example:
Comment: He has scored 5 goals in the last 3 games.
Comment: This team so far beaten 3-0 at home by the former champion.
Comment: His transfer fee was the highest in the club's history.
Comment: He was in injury time in the last game.
Attention:
The comment should be short, clear, accurate, and suitable for live commentary. 
The game date will be given as YYYY-MM-DD. Do not use information dated after this.
This comment should be natural comments following the previous comments given to the prompt.
Please generate only one comment and do not say anything else.
The comment should be between 6 and 18 words, or shorter."""

INSTRUCTION_JA = """
あなたは、サッカーの生中継で活躍するプロのカラ―コメンテーターです。
下記のdoccumentをもとに、現在の試合に関連する事実（例：選手の記録やチームの統計情報など）を盛り込んだ実況コメントを、必ず1つだけ作成してください。

例:
Comment:彼は直近3試合で5ゴールを決めています。
Comment:このチームは、かつての王者にホームで3-0で敗れました。
Comment:彼の移籍金はクラブ史上最高額でした。
Comment:彼は前回の試合で延長戦に登場しました。

注意:
コメントは短く、明瞭で正確、かつ生中継にふさわしい内容にしてください。
試合の日付は YYYY-MM-DD の形式で提示されます。それ以降の日付の情報は使用しないでください。
コメントは、これまでの流れに沿った自然な形で作成してください。
出力は実況コメントのみとし、余計な情報は記載しないでください。
コメントの長さは、20文字から30文字程度にしてください。
出力は必ず、です・ます口調の日本語で行ってください。
"""

COMMENT_TO_EVENT_TEXT_PROMPT = """You are a professional play-by-play commentator for a live soccer broadcast.
Using the following live commentary, provide an alternative natural-sounding version.
Guidelines:
Refer to only one information even if multiple are provided in Real live commentary.
Use 10 words or fewer.
You may state only a player’s name.
You may state a player’s name with an action.
You may state only an action.
Occasionally, include emotions and details.
If there is no Reals live commentary, please mention something about someone's action such as pass, dribble.

Real live commentary:
{comment}
"""

COMMENT_TO_EVENT_TEXT_PROMPT_JA = """あなたは、ライブサッカー放送のプロの実況者です。
以下の英語のライブ実況をもとに、別の自然な日本語実況を作成してください。

ガイドライン：
実際のライブ実況に複数の情報が含まれていても、1つの情報だけに言及してください。
10単語以内で表現してください。
選手名のみでも構いません。
選手名とその行動のみでも構いません。
行動のみでも構いません。
時折、感情や詳細を含めても構いません。
もしライブ実況がない場合、誰かのパスやアクションなどを何かしら言及してください。

ライブ実況:
{comment}
"""

# No retrievalの場合のプロンプト
prompt_template_no_retrieval = """{instruction}

===
{query}

Comment:"""

# documentが与えられる場合のプロンプト
prompt_template = """{instruction}

===documents
{documents}
===
{query}

Comment:"""

# 知識ベースのデータ保存場所
DOCUMENT_DIR = Path("./data/addinfo_retrieval")

# langchainのデータ構造保存場所
PERSIST_LANGCHAIN_DIR = Path("./storage/langchain-embedding-ada002")
