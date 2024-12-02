import os
import sys
from datetime import datetime
from functools import partial
import logging
from typing import Literal, TypeVar

from pathlib import Path
from tap import Tap
from loguru import logger
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI as LangChainOpenAI
from langchain_core.documents import (
    Document,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings


from sn_providing.construct_query import SpottingDataList

#　(project-root)/.env を読み込む
load_dotenv()

logger.add(
    "logs/{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
    level="DEBUG",
)
# llama_index,langchain のログを標準出力に出す
# TODO loguru.logger に統一したい
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# 型エイリアス 文書スコアの算出方法方法
RetrieverType = Literal["tfidf", "openai-embedding"]

class Arguments(Tap):
    game: str | None = None # useless
    input_file: str
    output_file: str
    retriever_type: RetrieverType = "tfidf"


# constants
MODEL_CONFIG = {
    "model": "gpt-3.5-turbo",
    "temperature": 0,
}

EMBEDDING_CONFIG = {
    "model": "text-embedding-ada-002",
    "chunk_size": 1000,
}

INSTRUCTION = \
"""You are a professional color commentator for a live broadcast of soccer. 
Using the documents below, provide one concise fact, such as player records or team statistics, relevant to the current soccer match. 
The fact should be clear, accurate, and suitable for live commentary. 
The game date will be given as YYYY-MM-DD. Do not use information dated after this.
Generate the comment considering that it follows the previous comments."""

# 知識ベースのデータ保存場所
DOCUMENT_DIR = Path("./data/addinfo_retrieval")

# langchainのデータ構造保存場所
PERSIST_LANGCHAIN_DIR = Path("./storage/langchain-embedding-ada002")



def run_langchain(spotting_data_list: SpottingDataList, output_file: str, retriever_type: RetrieverType):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt_template = \
"""{instruction}

===documents
{documents}
===
{query}

Answer:"""

    retriever = get_retriever_langchain(retriever_type, langchain_store_dir=PERSIST_LANGCHAIN_DIR)

    llm = LangChainOpenAI(
        **MODEL_CONFIG
    )

    prompt = PromptTemplate.from_template(prompt_template)
    
    def log_documents(docs):
        for doc in docs:
            logger.info(f"Document: {doc.page_content}")
        return docs

    def log_prompt(prompt: str) -> str:
        logger.info(f"Overall Prompt: {prompt}")
        return prompt

    rag_chain = (
        {"instruction": lambda _: INSTRUCTION, "documents": retriever | log_documents | format_docs, "query": RunnablePassthrough()}
        | prompt
        | log_prompt
        | llm
        | StrOutputParser()
    )

    result_list = SpottingDataList([])
    for spotting_data in spotting_data_list.spottings:
        logger.info(f"Query: {spotting_data.query}")
        if spotting_data.query is None:
            continue
        
        response = rag_chain.invoke(spotting_data.query)
        spotting_data.generated_text = response
        result_list.spottings.append(spotting_data)
        
        logger.info(f"Response: {response}")
    # save
    result_list.to_jsonline(output_file)


def get_retriever_langchain(type: RetrieverType, langchain_store_dir: Path) -> BaseRetriever:
    if type == "tfidf":
        if not os.path.exists(langchain_store_dir):
            # インデックスの構築
            splits = get_document_splits(DOCUMENT_DIR)
            retriever = TFIDFRetriever.from_documents(splits)
            # 保存
            retriever.save_local(folder_path=langchain_store_dir)
        else:
            # ローカルから読み込み
            retriever = TFIDFRetriever.load_local(
                folder_path=langchain_store_dir, 
                allow_dangerous_deserialization=True
            )
        return retriever
    elif type == "openai-embedding":
        embeddings = OpenAIEmbeddings(**EMBEDDING_CONFIG)
        if not os.path.exists(langchain_store_dir):
            # インデックスの構築
            splits = get_document_splits(DOCUMENT_DIR)
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            # 保存
            vectorstore.save_local(folder_path=langchain_store_dir)
        else:
            # ローカルから読み込み
            vectorstore: FAISS = FAISS.load_local(
                folder_path=langchain_store_dir, 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever()
        return retriever
    else:
        raise ValueError(f"Invalid retriever type: {type}. Use 'tfidf' or 'openai-embedding'.")


def get_document_splits(ducument_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for doc_path in os.listdir(ducument_dir):
        with open(os.path.join(ducument_dir, doc_path), "r") as f:
            doc = Document(page_content=f.read())
            documents.append(doc)
    splits = text_splitter.split_documents(documents)
    return splits


if __name__ == "__main__":
    args = Arguments().parse_args()

    spotting_data_list = SpottingDataList.from_jsonline(args.input_file)
    
    run_langchain(spotting_data_list, args.output_file, args.retriever_type)

