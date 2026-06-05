import chromadb

chroma_client = chromadb.Client()

collection= chroma_client.get_or_create_collection(name="test_collection")

documents = [
    {"id":"doc1",'text':"Hello world."},
    {"id":"doc2",'text':"How are you Today?"},
    {"id":"doc3",'text':"GoodBye, see you later!"}
]

for doc in documents:
    collection.upsert(
    ids=[doc["id"]],
    documents=[doc["text"]]
    )


query_text="Hello world"

result= collection.query(
    query_texts=[query_text],
    n_results=3
)

print(result)


