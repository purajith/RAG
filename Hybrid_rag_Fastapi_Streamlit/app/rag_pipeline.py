import os 
from langchain.text_splitter  import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from file_loader import extracting_text


# --- Retrieval Pipeline ('rag_pipeline.py') ---
def create_retriever(documents_raw: list[str]) -> EnsembleRetriever:
    """
    Creates and returns a hybrid EnsembleRetriever using BM25 and a Vector Store.

    Args:
        documents_raw (List[str]): A list of raw text strings to be indexed.
    
    Returns:
        EnsembleRetriever: The configured hybrid retriever.
    """
    # Use a text splitter to create Document objects
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""],
    )
    docs = text_splitter.create_documents(documents_raw)

    # a) BM25 Retriever (Keyword Search)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # b) Vector Store Retriever (Semantic Search)
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding_function)
    vector_store_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Combine retrievers with EnsembleRetriever and RRF
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_store_retriever],
        weights=[0.5, 0.5],
        c=60,
    )
    return ensemble_retriever


def retrieve_documents(retriever: EnsembleRetriever, query: str) -> list[Document]:
    """
    Performs a hybrid search and retrieves a list of relevant documents.

    Args:
        retriever (EnsembleRetriever): The hybrid retriever instance.
        query (str): The user's search query.

    Returns:
        List[Document]: A list of retrieved documents.
    """
    return retriever.invoke(query)


# # --- Main execution block -
# if __name__ == "__main__":
#     # 1. Get raw data (simulated from 'extract.py')
#     print("Step 1: Extracting documents...")
#     doc_path = "data/Data Retention and Disposal_20240621122021.docx"
#     extracted_data = extracting_text(doc_path)

#     # 2. Create the hybrid retriever
#     print("Step 2: Creating hybrid retriever...")
#     hybrid_retriever = create_retriever(extracted_data)

#     # 3. Define and run the query (retrieval step)
#     # query = "what is temporary files containing personally identifiabl"
#     print(f"Step 3: Running hybrid search for query: '{query}'")
#     retrieved_docs = retrieve_documents(hybrid_retriever, query)

#     # 4. Process retrieved docs (simulated for an 'llm.py' file)
#     print("\n--- Processed for LLM (Retrieved Documents) ---")
#     for i, doc in enumerate(retrieved_docs):
#         print(f"Rank {i+1}:")
#         print(doc.page_content)
#         print("-" * 20)

