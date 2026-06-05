from langchain_chroma import Chroma
import tempfile
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
import os
print(os.getenv("GOOGLE_API_KEY"))

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


docs = [
    Document(
        page_content="""
        LangChain is an open-source framework for building applications powered by large language models.
        It provides components for prompts, chains, agents, memory, and retrieval systems.
        LangChain supports integrations with OpenAI, Gemini, Anthropic, Groq, and many vector databases.
        """,
        metadata={"source": "langchain_docs"}
    ),
    Document(
        page_content="""
        Chroma is an open-source vector database designed for AI applications.
        It stores embeddings and enables semantic search over documents.
        Chroma can be used with LangChain to build Retrieval-Augmented Generation systems.
        """,
        metadata={"source": "chroma_docs"}
    ),
    Document(
        page_content="""
        Retrieval-Augmented Generation, or RAG, combines information retrieval with large language models.
        Documents are converted into embeddings and stored in a vector database.
        When a user asks a question, relevant documents are retrieved and passed to the language model.
        """,
        metadata={"source": "rag_docs"}
    ),
    Document(
        page_content="""
        Vector embeddings are numerical representations of text.
        Similar texts produce vectors that are close together in vector space.
        Embeddings are commonly used for semantic search, recommendation systems, and clustering.
        """,
        metadata={"source": "embedding_docs"}
    ),
]


def langchain_chroma_basic():

    with tempfile.TemporaryDirectory() as tmpdir:
        vectorstore = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")
        
        result = vectorstore.similarity_search(query="What is a vector database?",k=2)
        print(result)
        for i,doc in enumerate(result):
            print(f"{i+1}. {doc.page_content} source:{doc.metadata.get('source')}")

def similarity_search_with_scores():
    vector_store = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")
    
    query="what is vector db and types?"
    result_with_scores=vector_store.similarity_search_with_score(query,k=3)  
    
    for i,(doc,score) in enumerate(result_with_scores):
        print(f"{i+1}. {doc.page_content} score:{score:.4f} source:{doc.metadata.get("source")}")     
            
if __name__ == __name__:        
    # langchain_chroma_basic()
    similarity_search_with_scores()
            