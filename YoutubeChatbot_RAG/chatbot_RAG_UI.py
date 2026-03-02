# from langchain_ollama import OllamaEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import streamlit as st

st.header("Youtube Chatbot")

with st.chat_message("ai"):
    st.write("Hello")

prompt = st.chat_input("Say Something")
if prompt:
    st.write(f"User has sent the following prompt : {prompt}")




# video_id = "80M9kzbY8Ms"

# try:
#     api = YouTubeTranscriptApi()
#     scriptsList = api.fetch(video_id=video_id, languages=['hi'])
    
# except TranscriptsDisabled:
#     print("No caption available for this video.")



# # Investigating

# print(scriptsList)
# print("\n")
# print(scriptsList.snippets)
# print("\n")
# print(scriptsList.snippets[0])
# print("\n")
# #print(scriptsList.snippets[0].text)




# # Gathering only 'text'

# transcipts = []

# for chunk in scriptsList.snippets:
#     transcipts.append(chunk.text)
# print(transcipts)
# print("\n")

# plain_transcipts = " ".join(transcipts)
# print(plain_transcipts)


# # Alternatives 
# print("\n")

# easy = " ".join(s.text for s in scriptsList.snippets)
# print(easy)




# text_spliter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = text_spliter.create_documents([easy])
# print(chunks)
# print("\n")
# print(len(chunks))
# print("\n")
# print(chunks[100].page_content)



# embeddings = OllamaEmbeddings(model="nomic-embed-text")


# vector_store = FAISS.from_documents(chunks, embeddings)


# vector_store.index_to_docstore_id

# vector_store.get_by_ids(['818a2d2e-b0bd-4196-95ef-1eca9fc63eee'])



# retriever = vector_store.as_retriever(
#     search_type="similarity",
#     search_kwargs={'k':4}
# )


# print(retriever)


# retriever.invoke('')

# llm = ChatOllama(
#     model="llama3.1:8b",
#     temperature=0.2
# )

# prompt = PromptTemplate(
#     template="""
#         आप उपन्यास **"White Nights"** के लेखक की भूमिका निभा रहे हैं।
#         नीचे दिए गए संदर्भ के आधार पर ही उत्तर दें।
        

#         संदर्भ:
#         {context}

#         प्रश्न: {question}

#     """,
#     input_variables= ['context', 'question']
# )


# question = "कहानी की चार रातों में क्या-क्या मुख्य घटनाएँ होती हैं?"
# retrieved_docs = retriever.invoke(question)

# print(retrieved_docs)

# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# print(context_text)

# final_prompt = prompt.invoke({'context': context_text, "question": question})


# print(final_prompt)


# response = llm.invoke(final_prompt)
# print(response.content)