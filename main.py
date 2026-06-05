from dotenv import load_dotenv
from langchain_core import __version__ as core_version
import langgraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from importlib.metadata import version

load_dotenv()
# config versions
langgraph_ver = version("langgraph")
google_genai_ver = version("langchain-google-genai")
groq_ver = version("langchain-groq")

# Build the structured config of versions.
version_config = f"""
--- Installed Package Versions ---
LangChain Core:        {core_version}
LangGraph:             {langgraph_ver}
LangChain Google GenAI: {google_genai_ver}
LangChain Groq:         {groq_ver}
----------------------------------
"""
print(version_config)


def main():
    gemmi_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)
    
    response = gemmi_llm.invoke("Explain RAG in simple terms if you know they say i know")
    print(response)
    
    
    
    groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

    response = groq_llm.invoke("What is RAG?in one word")
    print(response)


if __name__ == "__main__":
    main()
