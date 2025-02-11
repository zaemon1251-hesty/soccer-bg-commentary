import logging

logger = logging.getLogger(__name__)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def log_documents(docs):
    for doc in docs:
        logger.info(f"Document: {doc.page_content}")
    return docs

def log_prompt(prompt: str) -> str:
    logger.info(f"Overall Prompt: {prompt}")
    return prompt