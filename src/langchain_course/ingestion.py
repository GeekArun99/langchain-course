from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os

load_dotenv()

if __name__ == "__main__":
    print("ingestion")

    loader = UnstructuredLoader(
        file_path="src/langchain_course/mediumblog1.txt",
        chunking_strategy="basic",
        max_characters=1000000
    )

    document = loader.load()

    print("splitting")

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    chunks = text_splitter.split_documents(document)

    print(f"created {len(chunks)} chunks")

    print("embeddings")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Storing in pinecone")

    PineconeVectorStore.from_documents(
        chunks,
        embedding=embedding_model,
        index_name=os.environ["INDEX_NAME"]
    )

    print("done")