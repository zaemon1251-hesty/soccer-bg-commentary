import os
import sys
from datetime import datetime
from functools import partial
import logging

from loguru import logger
from llama_index.core import ( 
    SimpleKeywordTableIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage 
)
from llama_index.llms.openai import OpenAI

from sn_providing.construct_query import SpottingDataList


logger.add(
    "logs/{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")),
    level="DEBUG",
)
# loguru loggerを標準のloggingモジュールと統合するためのハンドラ
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


# constants
TARGET_GAME = "england_epl/2015-2016/2015-08-16 - 18-00 Manchester City 3 - 0 Chelsea"
INPUT_FILE = f"outputs/{TARGET_GAME}/result_spotting_query.jsonl" # current dir is project root

MODEL_CONFIG = {
    "model_name": "gpt-3.5-turbo",
    "temperature": 0,
}

PROMPT_TEMPLATE = \
"""
{instruction}
===
{query}
"""

INSTRUCTION = \
"""
Based on Wikipedia, provide interesting events (records, stats) related to the current context query.
Select at least 5 events at random. 
Write each description in a brief one-sentence format. Do not provide any other message.
"""

format_prompt_with_query = partial(PROMPT_TEMPLATE.format, instruction=INSTRUCTION)

spotting_data_list = SpottingDataList.from_jsonline(INPUT_FILE)

# build index
# check if the index is already stored
DOCUMENT_DIR = "./data/addinfo_retrieval"
PERSIST_DIR = "./storage/addinfo_retrieval"

if not os.path.exists(PERSIST_DIR):
    # store mode
    documents = SimpleDirectoryReader(DOCUMENT_DIR).load_data()
    # SimpleKeywordTableIndex は 単語の正規表現 に基づく
    index = SimpleKeywordTableIndex.from_documents(documents)
    # store to disk
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

llm = OpenAI(
    model=MODEL_CONFIG["model_name"],
    temperature=MODEL_CONFIG["temperature"],
)
query_engine = index.as_query_engine(llm=llm)

# query
result_list = SpottingDataList([])
for spotting_data in spotting_data_list.spottings[:10]:
    logger.info(f"Querying: {spotting_data.query}")
    if spotting_data.query is None:
        continue
    
    prompt = format_prompt_with_query(query=spotting_data.query)
    response = query_engine.query(prompt)
    spotting_data.addiofo = str(response.response) if response.response else None
    result_list.spottings.append(spotting_data)
    
    logger.info(f"Response: {response}")

# save
OUTPUT_FILE = f"outputs/{TARGET_GAME}/results_addinfo_retrieval.jsonl"
result_list.to_jsonline(OUTPUT_FILE)
