from sn_providing.addinfo_retrieval import (
    get_retriever_langchain, 
    PERSIST_LANGCHAIN_DIR,
    RetrieverType,
)
from sn_providing.entity import CommentDataList
from tap import Tap
import time


class Arguments(Tap):
    game: str
    comment_csv: str
    retriever_type: RetrieverType = "tfidf"


if __name__ == "__main__":
    args = Arguments().parse_args()
    retriever = get_retriever_langchain(
        args.retriever_type, 
        langchain_store_dir=PERSIST_LANGCHAIN_DIR
    )
    comment_data_list = CommentDataList.read_csv(args.comment_csv, args.game)
    
    for comment_data in comment_data_list.comments[:20]:
        time.sleep(2)
        print(f"{comment_data.text=}")
        rdocs = retriever.invoke(comment_data.text)
        for doc in rdocs:
            print(f"{doc.page_content=}")
        print("=============")
