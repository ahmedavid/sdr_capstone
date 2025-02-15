# Semantic Question and Answering for documents using Large Language Models

Contributers:
- David Ahmadov (Backend, Langchain, Google Gemini Integration, Vector Embeddings)
- Parisa Ahmadlu (Frontend, Ollama integration)

In this project we created an application which user can upload a pdf document and subsequently ask questions about contents of the document.
Application uses concepts of Chunking, Embedding and Context enriching to allow LLMs to provide accurate answers to questions about the contents of the document 
thus avoiding hallucinations. Sometimes documents can be too long that can't fit into LLMs context window. Therefore we need this approach instead of simply copy pasting document into LLMs input.

We have used langchain to facilitate LLM querying. Langchain allows easily to switch between difference LLM providers.
We have used Ollama for local usage and Google Gemini for omline usage.

Tech Stack used

1. Langchain
2. Ollama
3. Deepseek R1
4. Google Gemini 2.0 flash
5. Streamlit
6. Vector Database Indexing
7. Vector Embeddings

Langchain 

## Live Demo
https://huggingface.co/spaces/ahmedavid/pdf_chat_capstone

![image](https://github.com/user-attachments/assets/ce77b194-b05c-4528-928d-aac370954aaa)


---
title: Pdf Chat Capstone
emoji: 👀
colorFrom: red
colorTo: gray
sdk: streamlit
sdk_version: 1.42.0
app_file: app.py
pinned: false
short_description: pdf_chat_capstone
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
