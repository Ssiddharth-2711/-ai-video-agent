import os

from dotenv import load_dotenv

load_dotenv(override=True)
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever
)


def get_llm():

    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def format_docs(docs):

    return "\n\n".join(
        [
            f"[Transcript Chunk {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ]
    )


def build_rag_chain(transcript: str):

    print("Building vector store...")

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=5
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI video assistant.

Your job is to answer the user's question using ONLY
the transcript context provided below.

Rules:

1. Answer the exact question asked.
2. Use the most relevant information from the transcript.
3. Prefer direct definitions and explanations over secondary examples.
4. Combine information from multiple transcript chunks when necessary.
5. Do not add information that is not supported by the transcript.
6. If the answer cannot be found in the transcript, say exactly:

"I could not find this information in the meeting transcript."

7. Keep answers concise but informative.
8. Do not mention "chunks", "vector database", "retriever",
or internal system details unless the user asks about them.

Transcript context:

{context}
""",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store,
        k=5
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI video assistant.

Answer the user's question using ONLY the transcript
context provided below.

Rules:

1. Answer the exact question.
2. Use the most relevant information.
3. Do not invent information.
4. If the answer is not present, say:

"I could not find this information in the meeting transcript."

5. Be concise and informative.

Transcript context:

{context}
""",
            ),
            (
                "human",
                "{question}"
            ),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_question(rag_chain, question: str) -> str:

    print(f"Question: {question}")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")

    return answer