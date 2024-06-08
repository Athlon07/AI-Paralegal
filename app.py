import os
import time

import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import Together

st.set_page_config(page_title="AI PARALEGAL")
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image("BHARAT LEX.jpg")

st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.reset_clicked = False  # Reset the flag

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )

if "reset_clicked" not in st.session_state:
    st.session_state.reset_clicked = False

embeddings = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"},
)

db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})  

prompt_template = """<s>[INST]This is a chat template and As a legal chat bot specializing in Indian Penal Code queries, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])

os.environ["TOGETHER_AI"] = "3b01ba7c029199b51dfa32baa6aff8c3d261a60c4552c05dac17b95b2c7bf964"
TOGETHER_AI_API = os.environ["TOGETHER_AI"]
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.5, max_tokens=1024, together_api_key=f"{TOGETHER_AI_API}"
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, memory=st.session_state.memory, retriever=db_retriever, combine_docs_chain_kwargs={"prompt": prompt}
)

def display_chat_messages(messages):
    for message in messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))


def process_user_input(input_text):
    with st.chat_message("user"):
        st.write(input_text)

    st.session_state.messages.append({"role": "user", "content": input_text})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke(input=input_text)

            message_placeholder = st.empty()

            full_response = "⚠️ **_Note: Information provided may be inaccurate._** \n\n\n"
            for chunk in result["answer"]:
                full_response += chunk
                time.sleep(0.02)

                message_placeholder.markdown(full_response + " ▌")
        st.button("Reset All Chat 🗑️", on_click=reset_conversation)  # Add reset button

        if st.session_state.reset_clicked:
            reset_conversation()  # Call reset function
            # Reset other elements or states if needed

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

def scrape_legal_news():
    news_data = []
    try:
        url = "https://indianexpress.com/section/india/legal/"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            news_items = soup.find_all('div', class_='articles')
            for item in news_items[:5]:  # top 5 news items
                headline_elem = item.find('h2', class_='title')
                summary_elem = item.find('p', class_='description')
                link_elem = item.find('a', class_='story')
                if headline_elem and summary_elem and link_elem:
                    headline = headline_elem.text.strip()
                    summary = summary_elem.text.strip()
                    link = link_elem['href']
                    news_data.append((headline, summary, link))
    except Exception as e:
        st.error(f"An error occurred while scraping legal news: {e}")

    return news_data
    


for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

input_prompt = st.text_input("Say something", key="user_input")

if input_prompt:
    process_user_input(input_prompt)

st.subheader("Legal News Updates")
st.write("Stay informed with the latest legal news!")

legal_news = scrape_legal_news()

if legal_news:
    for headline, summary, link in legal_news:
        st.subheader(headline)
        st.write(summary)
        st.write(f"[Read more]({link})")
        st.write("---")
else:
    st.write("No legal news updates available at the moment.")
