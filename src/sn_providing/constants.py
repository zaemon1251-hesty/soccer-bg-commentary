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
