import os
from datetime import datetime
from functools import partial
import logging

from pathlib import Path
from tap import Tap
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI as LangChainOpenAI
from langchain_core.documents import (
    Document,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings


from sn_providing.entity import SpottingDataList, ReferenceDoc, RetrieverType

#　(project-root)/.env を読み込む
load_dotenv()

# llama_index,langchain のログを標準出力に出す
logging.basicConfig(
    level=logging.DEBUG, 
    filename="logs/{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
)



class Arguments(Tap):
    game: str | None = None # useless
    input_file: str
    output_file: str
    retriever_type: RetrieverType = "tfidf"
    no_retrieval: bool = False
    reference_documents_yaml: str | None = None

# constants
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

INSTRUCTION = \
"""You are a professional color commentator for a live broadcast of soccer. 
Using the documents below, 
provide just one comment with a fact, such as player records or team statistics, relevant to the current soccer match. 
The comment should be short, clear, accurate, and suitable for live commentary. 
The game date will be given as YYYY-MM-DD. Do not use information dated after this.
This comment should be natural comments following the previous comments given to the prompt."""
# No retrievalの場合のプロンプト
prompt_template_no_retrieval = \
"""{instruction}

===
{query}

Comment:"""

# documentが与えられる場合のプロンプト
prompt_template = \
"""{instruction}

===documents
{documents}
===
{query}

Comment:"""

# 知識ベースのデータ保存場所
DOCUMENT_DIR = Path("./data/addinfo_retrieval")

# langchainのデータ構造保存場所
PERSIST_LANGCHAIN_DIR = Path("./storage/langchain-embedding-ada002")



def run_langchain(
    spotting_data_list: SpottingDataList, 
    output_file: str, retriever_type: RetrieverType,
    no_retrieval: bool = False,
    reference_documents_yaml: str | None = None
):
    """
    LangChainを使って付加的情報を生成する
    """

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def log_documents(docs):
        for doc in docs:
            logging.info(f"Document: {doc.page_content}")
        return docs

    def log_prompt(prompt: str) -> str:
        logging.info(f"Overall Prompt: {prompt}")
        return prompt

    retriever = get_retriever_langchain(retriever_type, langchain_store_dir=PERSIST_LANGCHAIN_DIR)

    llm = LangChainOpenAI(
        **MODEL_CONFIG
    )

    # リファレンスドキュメントが与えられた場合使う
    reference_doc_data = None
    get_reference_documents_partial = None

    # チェーンの構築
    if no_retrieval:
        rag_chain = (
            {
                "instruction": lambda _: INSTRUCTION,  
                "query": lambda spotting_data: spotting_data.query}
            | PromptTemplate.from_template(prompt_template_no_retrieval)
            | log_prompt
            | llm
            | StrOutputParser()
        )
    elif reference_documents_yaml is not None:
        # 正解文書の準備
        reference_doc_data = ReferenceDoc.get_list_from_yaml(reference_documents_yaml)
        get_reference_documents_partial = partial(ReferenceDoc.get_reference_documents, reference_documents=reference_doc_data)

        rag_chain = (
            {
                "instruction": lambda _: INSTRUCTION, 
                "documents": lambda spotting_data: get_reference_documents_partial(spotting_data.game, spotting_data.half, spotting_data.game_time),
                "query": lambda spotting_data: spotting_data.query}
            | PromptTemplate.from_template(prompt_template)
            | log_prompt
            | llm
            | StrOutputParser()
        )
    else:
        def process_docs(spotting_data):
            query = lambda _: spotting_data.query # noqa
            docs = query | retriever | log_documents | format_docs
            return docs
        rag_chain = (
            {
                "instruction": lambda _: INSTRUCTION, 
                "documents": lambda spotting_data: process_docs(spotting_data), 
                "query": lambda spotting_data: spotting_data.query}
            | PromptTemplate.from_template(prompt_template)
            | log_prompt
            | llm
            | StrOutputParser()
        )

    # run
    result_list = SpottingDataList([])
    for spotting_data in spotting_data_list.spottings:
        logging.info(f"Query: {spotting_data.query}")
        if spotting_data.query is None:
            continue
        
        if reference_doc_data is not None and \
            get_reference_documents_partial(spotting_data.game, spotting_data.half, spotting_data.game_time) is None:
            # 正解文書がない場合はスキップ
            logging.info(f"skip : {spotting_data.game}, {spotting_data.half}, {spotting_data.game_time}")
            continue
        
        response = rag_chain.invoke(spotting_data)
        spotting_data.generated_text = response
        result_list.spottings.append(spotting_data)
        
        logging.info(f"Response: {response}")
    # save
    result_list.to_jsonline(output_file)


def get_retriever_langchain(
    type: RetrieverType, 
    langchain_store_dir: Path
) -> BaseRetriever:
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
        retriever.k = SEARCH_CONFIG["k"]
        return retriever
    elif type == "openai-embedding":
        embeddings = OpenAIEmbeddings(**EMBEDDING_CONFIG)
        if not os.path.exists(langchain_store_dir):
            # インデックスの構築
            splits = get_document_splits(DOCUMENT_DIR)
            
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': EMBEDDING_CONFIG["score_threshold"], 'k': EMBEDDING_CONFIG["k"]}
            )
            # 保存
            vectorstore.save_local(folder_path=langchain_store_dir)
        else:
            # ローカルから読み込み
            vectorstore: FAISS = FAISS.load_local(
                folder_path=langchain_store_dir, 
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs=SEARCH_CONFIG
            )
        return retriever
    else:
        raise ValueError(f"Invalid retriever type: {type}. Use 'tfidf' or 'openai-embedding'.")


def get_document_splits(ducument_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 100):
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

    if args.no_retrieval:
        INSTRUCTION = INSTRUCTION.replace("Using the documents below,", "")

    run_langchain(
        spotting_data_list, 
        args.output_file, 
        args.retriever_type, 
        args.no_retrieval, 
        args.reference_documents_yaml
    )
