from .embeddings import vector_store

from langchain.tools.retriever import create_retriever_tool
retriever = vector_store.as_retriever(search_type="mmr",
    search_kwargs={"k": 2, "score_threshold": 0.5},)
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_documents",
    "Search and return information according to the  query",
)

tools = [retriever_tool]