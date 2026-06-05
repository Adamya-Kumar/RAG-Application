import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader,WebBaseLoader,DirectoryLoader,PyPDFLoader)
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

load_dotenv()


def text_loader():
    with tempfile.NamedTemporaryFile(delete=False,suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this sample text file \nthis is file is used testing the doument with TextLoader.")
        temp_file_path = temp_file.name
        
    try:
        loader = TextLoader(temp_file_path)
        document = loader.load()
        
        print(f"Loaded {len(document)} Document(s)")
        print(f"Content Perview: {document[0].page_content[:100]}..")
        print(f"Metadata: {document[0].metadata.get("source")}")
        
        # for doc in document:
        #     print(doc.page_content)
            
    finally:
        os.remove(temp_file_path)




def pdf_loader(pdf_path:str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} Document(s) from PDF")
    for i,doc in enumerate(documents):
        print(f"Document {i+1} Content Perview: {doc.page_content[:100]}")
        print(f"Metadata: {doc.metadata}")
           
        
if __name__ == "__main__":
    #text_loader()
    pdf_loader("./docs/langchain_demo.pdf")
