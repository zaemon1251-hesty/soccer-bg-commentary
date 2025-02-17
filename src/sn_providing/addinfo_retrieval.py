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
from sn_providing.constants import (
    MODEL_CONFIG,
    EMBEDDING_CONFIG,
    SEARCH_CONFIG,
    INSTRUCTION,
    prompt_template,
    prompt_template_no_retrieval,
    DOCUMENT_DIR,
    PERSIST_LANGCHAIN_DIR,
    INSTRUCTION_NO_RETRIEVAL,
    INSTRUCTION_JA
)
from sn_providing.util import format_docs, log_documents, log_prompt

# (project-root)/.env を読み込む
load_dotenv()

# llama_index,langchain のログを標準出力に出す
logging.basicConfig(
    level=logging.DEBUG,
    filename="logs/addinfo--{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
)


class Arguments(Tap):
    game: str | None = None  # useless
    input_file: str
    output_file: str
    retriever_type: RetrieverType = "tfidf"
    no_retrieval: bool = False
    reference_documents_yaml: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0
    embedding_model: str = "text-embedding-ada-002"
    chunk_size: int = 1000
    search_k: int = 10
    search_score_threshold: float = 0.7


def run(
    spotting_data_list: SpottingDataList,
    output_file: str,
    retriever_type: RetrieverType,
    no_retrieval: bool = False,
    reference_documents_yaml: str | None = None,
    model_config: dict = MODEL_CONFIG,
    embedding_config: dict = EMBEDDING_CONFIG,
    search_config: dict = SEARCH_CONFIG,
    instruction: str = INSTRUCTION,
):
    """
    LangChainを使って付加的情報を生成する
    """

    # レトリバの取得
    retriever = get_retriever(
        retriever_type,
        langchain_store_dir=PERSIST_LANGCHAIN_DIR,
        embedding_config=embedding_config,
        search_config=search_config,
    )

    llm = LangChainOpenAI(**model_config)

    # チェーンの構築
    rag_chain, reference_doc_data, get_reference_documents_partial = get_rag_chain(
        retriever=retriever,
        llm=llm,
        no_retrieval=no_retrieval,
        reference_documents_yaml=reference_documents_yaml,
        log_documents=log_documents,
        log_prompt=log_prompt,
        format_docs=format_docs,
        instruction=instruction,
    )

    # 実行 (rag_chain は spotting_data を受け取って text を返す)
    result_list = SpottingDataList([])
    for spotting_data in spotting_data_list.spottings:
        logging.info(f"Query: {spotting_data.query}")
        if spotting_data.query is None:
            continue

        if (
            reference_doc_data is not None
            and get_reference_documents_partial(
                spotting_data.game, spotting_data.half, spotting_data.game_time
            )
            is None
        ):
            # 正解文書がない場合はスキップ
            logging.info(
                f"skip : {spotting_data.game}, {spotting_data.half}, {spotting_data.game_time}"
            )
            continue

        response = rag_chain.invoke(spotting_data)
        spotting_data.generated_text = response
        result_list.spottings.append(spotting_data)

        logging.info(f"Response: {response}")
    # save
    result_list.to_jsonline(output_file)


def get_rag_chain(retriever, llm, **kwargs):
    """
    rag_chainを構築する

    return:
        rag_chain: langchainのチェーン
        reference_doc_data: 正解文書のデータ
        get_reference_documents_partial: 正解文書を取得する関数
    """
    instruction = kwargs.get("instruction", INSTRUCTION)
    
    no_retrieval = kwargs.get("no_retrieval", False)
    reference_documents_yaml = kwargs.get("reference_documents_yaml", None)

    # ログ関数
    log_documents = kwargs.get("log_documents", lambda x: x)
    log_prompt = kwargs.get("log_prompt", lambda x: x)
    format_docs = kwargs.get("format_docs", lambda x: x)

    reference_doc_data = None
    get_reference_documents_partial = None

    # チェーンの構築
    if no_retrieval:
        rag_chain = (
            {
                "instruction": lambda _: instruction,
                "query": lambda spotting_data: spotting_data.query,
            }
            | PromptTemplate.from_template(prompt_template_no_retrieval)
            | log_prompt
            | llm
            | StrOutputParser()
        )
    elif reference_documents_yaml is not None:
        # 正解文書の準備
        reference_doc_data = ReferenceDoc.get_list_from_yaml(reference_documents_yaml)
        get_reference_documents_partial = partial(
            ReferenceDoc.get_reference_documents, reference_documents=reference_doc_data
        )

        rag_chain = (
            {
                "instruction": lambda _: instruction,
                "documents": lambda spotting_data: get_reference_documents_partial(
                    spotting_data.game, spotting_data.half, spotting_data.game_time
                ),
                "query": lambda spotting_data: spotting_data.query,
            }
            | PromptTemplate.from_template(prompt_template)
            | log_prompt
            | llm
            | StrOutputParser()
        )
    else:

        def process_docs(spotting_data):
            query = lambda _: spotting_data.query  # noqa
            docs = query | retriever | log_documents | format_docs
            return docs

        rag_chain = (
            {
                "instruction": lambda _: instruction,
                "documents": lambda spotting_data: process_docs(spotting_data),
                "query": lambda spotting_data: spotting_data.query,
            }
            | PromptTemplate.from_template(prompt_template)
            | log_prompt
            | llm
            | StrOutputParser()
        )
    return rag_chain, reference_doc_data, get_reference_documents_partial


def get_retriever(
    type: RetrieverType,
    langchain_store_dir: Path,
    embedding_config: dict = EMBEDDING_CONFIG,
    search_config: dict = SEARCH_CONFIG,
    document_dir: Path = DOCUMENT_DIR,
) -> BaseRetriever:
    if type == "tfidf":
        if not os.path.exists(langchain_store_dir):
            # インデックスの構築
            splits = get_document_splits(document_dir)
            retriever = TFIDFRetriever.from_documents(splits)
            # 保存
            retriever.save_local(folder_path=langchain_store_dir)
        else:
            # ローカルから読み込み
            retriever = TFIDFRetriever.load_local(
                folder_path=langchain_store_dir, allow_dangerous_deserialization=True
            )
        retriever.k = embedding_config["k"]
        return retriever
    elif type == "openai-embedding":
        embeddings = OpenAIEmbeddings(**embedding_config)
        if not os.path.exists(langchain_store_dir):
            # インデックスの構築
            splits = get_document_splits(document_dir)

            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "score_threshold": embedding_config["score_threshold"],
                    "k": embedding_config["k"],
                },
            )
            # 保存
            vectorstore.save_local(folder_path=langchain_store_dir)
        else:
            # ローカルから読み込み
            vectorstore: FAISS = FAISS.load_local(
                folder_path=langchain_store_dir,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold", search_kwargs=search_config
            )
        return retriever
    else:
        raise ValueError(
            f"Invalid retriever type: {type}. Use 'tfidf' or 'openai-embedding'."
        )


def get_document_splits(
    ducument_dir: Path, chunk_size: int = 1000, chunk_overlap: int = 100
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
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
    logging.info("Arguments:")
    logging.info(f"Game: {args.game}")
    logging.info(f"Input File: {args.input_file}")
    logging.info(f"Output File: {args.output_file}")
    logging.info(f"Retriever Type: {args.retriever_type}")
    logging.info(f"No Retrieval: {args.no_retrieval}")
    logging.info(f"Reference Documents: {args.reference_documents_yaml}")

    model_config = {
        "model": args.model,
        "temperature": args.temperature,
    }
    embedding_config = {
        "model": args.embedding_model,
        "chunk_size": args.chunk_size,
    }

    search_config = {
        "k": args.search_k,
        "score_threshold": args.search_score_threshold,
    }

    logging.info(f"Model Config: {model_config}")
    logging.info(f"Embedding Config: {embedding_config}")
    logging.info(f"Search Config: {search_config}")

    instruction = INSTRUCTION
    if args.no_retrieval:
        instruction = INSTRUCTION_NO_RETRIEVAL

    run(
        spotting_data_list,
        args.output_file,
        args.retriever_type,
        args.no_retrieval,
        args.reference_documents_yaml,
        model_config,
        embedding_config,
        search_config,
        instruction=instruction,
    )
