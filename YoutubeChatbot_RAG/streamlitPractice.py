import streamlit as st
import re
import time
from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="llama3", disable_streaming=False)

# prompt = PromptTemplate(
#     template="""
#     Extract the YouTube video summary title from this link:

#     {link}

#     Rules:
#     - Output only the title.
#     - No extra sentences.
#     - No prefixes.
#     - No suffixes.
#     - No explanation.
#     - No quotation marks.
#     - Output must contain only the raw title text.
#     """,
#     input_variables=["link"]
# )

# query = prompt.invoke({'link': "https://youtu.be/80M9kzbY8Ms?si=fXL3qRNXGhmkltlA"})
# response = model.invoke(query)
# print(response.content)


st.title("Youtube Chatbot", text_alignment="center")

def extract_youtube_id(url):
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([^&?/]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

url = st.text_input("Paste YouTube URL")

if st.button("Extract ID"):
    with st.spinner("Extracting video details...", show_time=True):
        time.sleep(2.0)
        video_id = extract_youtube_id(url)
        
    if video_id:
        st.success("Video is ready for chat.")
        st.toast("Video loaded successfully!")
        ids = f"Your Video ID: {video_id}"
        st.session_state.messages.append({"role": "assistant", "content": ids})
        video = st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.error("Invalid YouTube URL")


if "messages" not in st.session_state:
    st.session_state.messages=[]

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


video_id = extract_youtube_id(url)

prompt = st.chat_input("Ask anything about the Video")

if prompt:
    
    if video_id:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # with st.spinner("Thinking...", show_time=True):
        #     response = model.invoke(prompt)
            
        with st.chat_message("ai"):
            full_response = ""
            placeholder = st.empty()

            for chunk in model.stream(prompt):
                full_response += chunk.content
                placeholder.markdown(full_response)

            placeholder.markdown(full_response)

            # for i in response.content.split():
            #     full_response += i + " "
            #     placeholder.markdown(full_response)
            #     time.sleep(0.05)
                    
        st.session_state.messages.append({"role": "ai", "content": full_response})
    else:
        st.error("Access Denied: YouTube ID is Required!")
