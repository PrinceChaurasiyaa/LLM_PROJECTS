import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

model = ChatOllama(model="llama3")

prompt = PromptTemplate(
    template="""
    Extaract only the YouTube id from Link: {link}
    """,
    input_variables=["link"]
)

# query = prompt.invoke({'link': "https://youtu.be/80M9kzbY8Ms?si=81ioOW41PAKndYgA"})
# response = model.invoke(query)
# print(response.content)

import streamlit as st
import re

st.header("Youtube Chatbot")

def extract_youtube_id(url):
    pattern = r"(?:v=|youtu\.be/|embed/|shorts/)([^&?/]+)"
    match = re.search(pattern, url)
    return match.group(1) if match else None

url = st.text_input("Paste YouTube URL")

if st.button("Extract ID"):
    video_id = extract_youtube_id(url)
    
    if video_id:
        st.success(f"Video ID: {video_id}")
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.error("Invalid YouTube URL")

st.title("ChatBot")

# name = st.text_input("Enter your Name")
# if name:
#     st.write("Hello, ", name)

# youtube_id = st.text_input("Enter Youtube video id")
# if youtube_id:
#     st.write("Video ID:", youtube_id)
#     st.video(f"https://www.youtube.com/watch?v={youtube_id}")






# with st.chat_message("ai"):
#     st.write("Hello")

youtube_id = st.chat_input("Ask anything about the Video")
if youtube_id:
    st.chat_message("user")
    





# if "messages" not in st.session_state:
#     st.session_state.messages=[]

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What is up"):
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     response = f"Echo: {prompt}"

#     with st.chat_message("assistant"):
#         st.markdown(response)
    
#     st.session_state.messages.append({'role':'assistant', 'content':response})
