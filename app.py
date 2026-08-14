import streamlit as st
from dotenv import load_dotenv

from main import run_pipeline
from core.rag_engine import ask_question


# ============================================================
# Configuration
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="AI Meeting Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# Session State
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# Header
# ============================================================

st.title("🤖 AI Meeting Assistant")

st.write(
    "Upload a meeting audio/video file or provide a YouTube URL "
    "to generate a transcript, summary, action items, decisions, "
    "and an AI-powered meeting Q&A."
)


# ============================================================
# Input
# ============================================================

st.subheader("📥 Meeting Input")

source_type = st.radio(
    "Choose input type",
    ["Local File", "YouTube URL"],
    horizontal=True,
)


source = None


if source_type == "YouTube URL":

    source = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

else:

    uploaded_file = st.file_uploader(
        "Upload audio/video",
        type=[
            "mp3",
            "wav",
            "m4a",
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
        ],
    )

    if uploaded_file:

        import os

        os.makedirs("uploads", exist_ok=True)

        file_path = os.path.join(
            "uploads",
            uploaded_file.name
        )

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        source = file_path

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


# ============================================================
# Language
# ============================================================

language = st.selectbox(
    "Meeting language",
    [
        "english",
        "hinglish",
    ],
)


# ============================================================
# Process Meeting
# ============================================================

if st.button(
    "🚀 Process Meeting",
    type="primary",
    use_container_width=True,
):

    if not source:

        st.warning(
            "Please provide a YouTube URL or upload a file."
        )

    else:

        st.session_state.messages = []

        with st.spinner(
            "Processing meeting... This may take some time."
        ):

            try:

                result = run_pipeline(
                    source,
                    language
                )

                st.session_state.result = result

                st.success(
                    "Meeting processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Error while processing meeting: {e}"
                )

                st.exception(e)


# ============================================================
# Display Results
# ============================================================

result = st.session_state.result


if result:

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    st.header(
        f"📌 {result['title']}"
    )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    st.subheader("📋 Meeting Summary")

    st.markdown(
        result["summary"]
    )


    # --------------------------------------------------------
    # Action Items / Decisions / Questions
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.subheader("✅ Action Items")

        st.markdown(
            result["action_items"]
        )


    with col2:

        st.subheader("🔑 Key Decisions")

        st.markdown(
            result["key_decisions"]
        )


    with col3:

        st.subheader("❓ Open Questions")

        st.markdown(
            result["open_questions"]
        )


    # --------------------------------------------------------
    # Transcript
    # --------------------------------------------------------

    st.subheader("📝 Transcript")

    with st.expander(
        "Show full transcript"
    ):

        st.text_area(
            "Transcript",
            result["transcript"],
            height=500,
        )


    # --------------------------------------------------------
    # Meeting Q&A
    # --------------------------------------------------------

    st.header("💬 Ask Questions About the Meeting")

    st.caption(
        "Ask questions based on the meeting transcript."
    )


    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # Chat input

    question = st.chat_input(
        "Ask something about the meeting..."
    )


    if question:

        # User message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)


        # Assistant response

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    answer = ask_question(
                        result["rag_chain"],
                        question
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Error answering question: {e}"
                    )