"""
starter code for using the llama_index library
"""

import logging
import sys
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.openai import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# check if the index is already stored
PERSIST_DIR = "./storage/example_storage"
if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("./data/example_data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

llm = OpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")
print(f"{response=}")
