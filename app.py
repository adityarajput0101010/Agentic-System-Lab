import streamlit as st
from dotenv import load_dotenv
import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load .env
load_dotenv()

# Get API Key
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit Title
st.title("🤖 Simple AI Chatbot")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Old Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

# Prompt Template
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

Answer this question:

Question: {question}

Answer:
"""
)

# User Input
user_input = st.chat_input("Ask something...")

if user_input:

    # Store User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Display User Message
    with st.chat_message("user"):
        st.write(user_input)

    # Create Prompt
    final_prompt = prompt.format(question=user_input)

    # AI Response
    response = llm.invoke(final_prompt)

    ai_response = response.content

    # Store AI Response
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

    # Display AI Response
    with st.chat_message("assistant"):
        st.write(ai_response)