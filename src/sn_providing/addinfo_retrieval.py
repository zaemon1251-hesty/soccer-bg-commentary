import os
import sys
from datetime import datetime
from functools import partial
import logging

from pathlib import Path
from tap import Tap
from loguru import logger
from dotenv import load_dotenv

from llama_index.core import ( 
    SimpleKeywordTableIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage 
)
from llama_index.llms.openai import OpenAI as LLamaIndexOpenAI

from langchain_openai import ChatOpenAI as LangChainOpenAI
from langchain_core.documents import (
    Document,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.prompts import PromptTemplate

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
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


class Arguments(Tap):
    game: str
    input_file: str
    output_file: str


# constants
MODEL_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0,
}

INSTRUCTION = \
"""
Based on Wikipedia, provide interesting events (records, stats) related to the current context query.
Select at least 5 events at random. 
Write each description in a brief one-sentence format. Do not provide any other message.
"""

# 知識ベースのデータ保存場所
DOCUMENT_DIR = Path("./data/addinfo_retrieval")

# llama indexのデータ構造保存場所
PERSIST_LLAMAINDEX_DIR = Path("./storage/llama_index")

# langchainのデータ構造保存場所
PERSIST_LANGCHAIN_DIR = Path("./storage/langchain")


def run_llamaindex(spotting_data_list: SpottingDataList, output_file: str):
    prompt_template = \
    """
    {instruction}
    === query context
    {query}
    """
    # build index
    # check if the index is already stored
    if not os.path.exists(PERSIST_LLAMAINDEX_DIR):
        # store mode
        documents = SimpleDirectoryReader(DOCUMENT_DIR).load_data()
        # SimpleKeywordTableIndex は 単語の正規表現 に基づく
        index = SimpleKeywordTableIndex.from_documents(documents)
        # store to disk
        index.storage_context.persist(persist_dir=PERSIST_LLAMAINDEX_DIR)
    else:
        # load
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_LLAMAINDEX_DIR)
        index = load_index_from_storage(storage_context)

    # 毎回instructionは同じなので、先に入力しておく
    format_prompt_with_query = partial(prompt_template.format, instruction=INSTRUCTION)

    llm = LLamaIndexOpenAI(
        model=MODEL_CONFIG["model_name"],
        temperature=MODEL_CONFIG["temperature"],
    )
    query_engine = index.as_query_engine(llm=llm)

    # query
    result_list = SpottingDataList([])
    for spotting_data in spotting_data_list.spottings[:10]: # run only 10 head for debug
        logger.info(f"Querying: {spotting_data.query}")
        if spotting_data.query is None:
            continue
        
        prompt = format_prompt_with_query(query=spotting_data.query)
        response = query_engine.query(prompt)
        spotting_data.addiofo = str(response.response) if response.response else None
        result_list.spottings.append(spotting_data)
        
        logger.info(f"Response: {response}")
    # save
    result_list.to_jsonline(output_file)


def run_langchain(spotting_data_list: SpottingDataList, output_file: str):
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt_template = \
    """
    {instruction}
    
    === wikipedia documents
    {documents}
    
    === query context
    {query}
    
    Answer:
    """

    # インデックスの構築
    if not os.path.exists(PERSIST_LANGCHAIN_DIR):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = []
        for doc_path in os.listdir(DOCUMENT_DIR):
            with open(os.path.join(DOCUMENT_DIR, doc_path), "r") as f:
                doc = Document(page_content=f.read())
                documents.append(doc)
        splits = text_splitter.split_documents(documents)
        retriever = TFIDFRetriever.from_documents(splits)
        # 保存
        retriever.save_local(folder_path=PERSIST_LANGCHAIN_DIR)
    else:
        # ローカルから読み込み
        retriever = TFIDFRetriever.load_local(
            folder_path=PERSIST_LANGCHAIN_DIR, 
            allow_dangerous_deserialization=True
        )

    llm = LangChainOpenAI(
        model=MODEL_CONFIG["model_name"],
        temperature=MODEL_CONFIG["temperature"],
    )
    
    prompt = PromptTemplate.from_template(prompt_template)
    
    rag_chain = (
        {"instruction": lambda _: INSTRUCTION, "documents": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    result_list = SpottingDataList([])
    for spotting_data in spotting_data_list.spottings[:10]: # run only 10 head for debug
        logger.info(f"Querying: {spotting_data.query}")
        if spotting_data.query is None:
            continue
        
        response = rag_chain.invoke(spotting_data.query)
        spotting_data.addiofo = response
        result_list.spottings.append(spotting_data)
        
        logger.info(f"Response: {response}")
    # save
    result_list.to_jsonline(output_file)


if __name__ == "__main__":
    args = Arguments().parse_args()

    spotting_data_list = SpottingDataList.from_jsonline(args.input_file)
    
    run_langchain(spotting_data_list, args.output_file)


