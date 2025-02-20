import logging
import textwrap


logger = logging.getLogger(__name__)

def wrap_text(text: str, max_width: int = 80) -> str:
    """
    英語の場合は単語単位で折り返し、
    日本語の場合は単純にmax_width文字ごとに改行します。
    """
    # 英語判定: 全ての文字がASCIIなら英語とみなす（簡易判定）
    if all(ord(c) < 128 for c in text):
        return "\n".join(textwrap.wrap(text, width=max_width))
    else:
        # 日本語の場合は指定した文字数ごとに分割
        return "\n".join([text[i:i+max_width] for i in range(0, len(text), max_width)])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def log_documents(docs):
    for doc in docs:
        logger.info(f"Document: {doc.page_content}")
    return docs

def log_prompt(prompt: str) -> str:
    logger.info(f"Overall Prompt: {prompt}")
    return prompt