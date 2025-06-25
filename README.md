# ⚖️ AI Paralegal  
*Your Generative AI-powered legal assistant, trained on the Indian Penal Code (IPC)*

![LangChain](https://img.shields.io/badge/LangChain-RAG-blue)  
![Python](https://img.shields.io/badge/Python-3.10-green)  
![TogetherAI](https://img.shields.io/badge/TogetherAI-LLM-orange)  
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

---

### 🧠 What is AI Paralegal?

**AI Paralegal** is an intelligent legal assistant designed to help users query Indian Penal Code (IPC) laws using natural language.  
It supports real-time retrieval, summarization, and contextual legal understanding via a **Retrieval-Augmented Generation (RAG)** pipeline.

Whether you're a law student, developer, or citizen seeking legal guidance—AI Paralegal has your back.

---

### 🔧 Features

- 🔍 Ask legal questions in plain English  
- 📜 Get structured, clause-based IPC responses  
- 🤖 Built using LLMs, LangChain, and TogetherAI API  
- 📁 Custom legal dataset embedded with FAISS  
- 🧠 Real-time RAG with semantic search  
- 🌐 Streamlit-based UI for instant legal Q&A

---

### 🧰 Tech Stack

| Category | Tech Used |
|----------|-----------|
| 💬 LLM | TogetherAI |
| 🔗 Framework | LangChain |
| 🔍 Vector Store | FAISS |
| 📄 Data | IPC Law JSON Corpus |
| 🧠 RAG Engine | Context + Query Chain |
| 🎯 Interface | Streamlit |

---

### 🚀 Demo

![AI Paralegal Demo](demo.gif)  
> *(Insert your demo gif here, or generate one with screen recording tools)*

---

### 🛠️ Setup

```bash
git clone https://github.com/Sujal-py3/AI-Paralegal.git
cd AI-Paralegal
pip install -r requirements.txt
streamlit run app.py
