from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import streamlit as st
import re
import time

llm = ChatOllama(
    model="llama3.1:8b",
    disable_streaming=False
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

prompt = PromptTemplate(
    template="""
       You are a helpful assistant that answers questions about a YouTube video using its transcript.
       Base your answer strictly on the context below. If the answer isn't in the context, say so honestly.
        
        Context:
        {context}

        Question: {question}

    """,
    input_variables= ['context', 'question']
)

st.set_page_config(page_title="YouTube Chatbot", page_icon="▶", layout="wide")

st.title("YouTube Chatbot", text_alignment="center")
st.markdown("A RAG-based chatbot that understands and answers queries from YouTube videos.", text_alignment="center")

def extract_youtube_id(url):
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([^&?/]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

if "messages" not in st.session_state:
    st.session_state.messages=[]

if "video_id" not in st.session_state:
    st.session_state.video_id = None

with st.sidebar:
    st.title("Load a Video")
    url = st.text_input("Paste YouTube URL", placeholder="https://youtu.be/..")
    if st.button("Extract ID", use_container_width=False):
        with st.spinner("Extracting video details...", show_time=True):
            time.sleep(1.0)
            video_id = extract_youtube_id(url)
            
        if video_id:
            st.session_state.video_id = video_id
            st.success("Video is ready for chat.")
            st.toast("Video loaded successfully!")
        else:
            st.error("Invalid YouTube URL")

    if st.session_state.video_id:
        st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
        st.caption(f"Video ID: `{st.session_state.video_id}`")

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()



for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

video_id = st.session_state.get("video_id")

question = st.chat_input("Ask anything about the Video")

if question:
    
    if st.session_state.video_id:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        
        try:
            with st.spinner("Processing transcripts...", show_time=True):
                api = YouTubeTranscriptApi()
                scriptsList = api.fetch(video_id=video_id, languages=['en', 'hi' ])

                easy = " ".join(s.text for s in scriptsList.snippets)
                text_spliter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_spliter.create_documents([easy])

                vector_store = FAISS.from_documents(chunks, embeddings)

                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k':4}
                )

                retrieved_docs = retriever.invoke(question)

                context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

                final_prompt = prompt.invoke({'context': context_text, "question": question})
            
            with st.chat_message("assistant"):
                full_response = ""
                placeholder = st.empty()

                for chunk in llm.stream(final_prompt):
                    full_response += chunk.content
                    placeholder.markdown(full_response)

                placeholder.markdown(full_response)
                        
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except (TranscriptsDisabled, NoTranscriptFound):
            msg = "No caption available for this video."
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        
        except Exception as e:
            msg = f"Something went wrong: {e}"
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})

    else:
        st.error("Access Denied: YouTube ID is Required!")
        st.info("Paste a YouTube URL in the sidebar to get started.")