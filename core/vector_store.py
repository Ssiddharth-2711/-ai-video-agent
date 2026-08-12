from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


VECTOR_STORE_PATH = "vector_store"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_documents(transcript: str):
    """Convert transcript into smaller meaningful chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
        ],
    )

    chunks = splitter.split_text(transcript)

    documents = [
        Document(
            page_content=chunk,
            metadata={
                "chunk_id": i
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    print(f"Created {len(documents)} transcript chunks.")

    return documents


def build_vector_store(transcript: str):

    print("Creating transcript chunks...")

    documents = create_documents(transcript)

    print("Creating embeddings...")

    embeddings = get_embeddings()

    print("Building FAISS vector store...")

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store


def get_retriever(vector_store, k=4):

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k
        }
    )


def save_vector_store(vector_store):

    vector_store.save_local(VECTOR_STORE_PATH)

    print("Vector store saved.")


def load_vector_store():

    embeddings = get_embeddings()

    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store