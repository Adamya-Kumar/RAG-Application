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
        LangChain provides document loaders that allow developers to ingest data
        from various sources into LLM applications. Document loaders are commonly
        used in Retrieval-Augmented Generation systems, chatbots, and knowledge bases.
        """,
        metadata={
            "topic": "document_loaders",
            "page": 1,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        PyPDFLoader loads PDF files page by page and preserves metadata such as
        page numbers. It is one of the most commonly used loaders in LangChain.
        """,
        metadata={
            "topic": "pypdfloader",
            "page": 1,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        LangChain supports multiple document loaders including TextLoader,
        CSVLoader, JSONLoader, DirectoryLoader, WebBaseLoader, and YouTubeLoader.
        Each loader is designed for a specific data source.
        """,
        metadata={
            "topic": "loader_types",
            "page": 1,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        After loading documents, text should be split into smaller chunks using
        RecursiveCharacterTextSplitter. Smaller chunks generally improve retrieval
        quality in vector databases.
        """,
        metadata={
            "topic": "text_splitting",
            "page": 2,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        Metadata preservation is important because it allows applications to
        provide citations, page references, and source tracking during retrieval.
        """,
        metadata={
            "topic": "metadata",
            "page": 2,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        LangChain provides modular components including prompt templates,
        output parsers, chains, agents, memory, and vector stores.
        These components help developers build LLM-powered applications.
        """,
        metadata={
            "topic": "langchain_features",
            "page": 2,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        Vector stores are used for semantic search. Documents are converted into
        embeddings and stored in a vector database. Queries are also embedded and
        matched against stored vectors.
        """,
        metadata={
            "topic": "vector_stores",
            "page": 2,
            "source": "langchain_demo.pdf"
        }
    ),

    Document(
        page_content="""
        Document loaders are the entry point for LangChain applications that work
        with external data. They enable robust RAG systems by connecting PDFs,
        websites, databases, and other sources to LLM workflows.
        """,
        metadata={
            "topic": "rag",
            "page": 3,
            "source": "langchain_demo.pdf"
        }
    )
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
        print(f"{i+1}. {doc.page_content} similarity:{1/(1+score):.4f} distance:{score:.4f} source:{doc.metadata.get("source")}")     
   
   
def metdata_filtering():
    vector_store = Chroma.from_documents(documents=docs,embedding=embeddings,persist_directory="./chroma_db")
    
    query = "what is rag?"
    filter_doc={"topic":"rag"}
    result_with_scores=vector_store.similarity_search(query,k=3,filter=filter_doc)  
    
    print(result_with_scores)
    for i,doc in enumerate(result_with_scores):
            print(f"{i+1}. {doc.page_content} source:{doc.metadata.get("source")}") 
   
   
           
if __name__ == __name__:        
    # langchain_chroma_basic()
    #similarity_search_with_scores()
    metdata_filtering()
            