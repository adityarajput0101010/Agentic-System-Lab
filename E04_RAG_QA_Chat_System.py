# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E04
# Title: RAG-Based Question Answering Chat System
# ============================================================
# Install: pip install langchain langchain-community langchain-anthropic
#          pip install pypdf faiss-cpu sentence-transformers

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from anthropic import Anthropic

# ── Configuration ──────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # Free local embedding model
TOP_K       = 3                     # Number of chunks to retrieve
# ───────────────────────────────────────────────────────────

client = Anthropic()


def build_knowledge_base(pdf_path=None):
    """Load PDF (or use sample text) and build FAISS vector store."""
    print("🔨 Building knowledge base...")

    if pdf_path and os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        raw_docs = loader.load()
        print(f"   Loaded {len(raw_docs)} page(s) from PDF.")
    else:
        print("   No PDF found — using built-in sample knowledge base.")
        sample_content = """
        Artificial Intelligence (AI) is the simulation of human intelligence processes by machines.
        AI applications include expert systems, natural language processing, speech recognition, and machine vision.

        Machine Learning (ML) is a subset of AI that gives systems the ability to automatically learn 
        and improve from experience without being explicitly programmed. ML focuses on developing 
        computer programs that can access data and use it to learn for themselves.

        Deep Learning uses neural networks with many layers (deep neural networks) to learn representations 
        of data with multiple levels of abstraction. It has revolutionized fields like image recognition, 
        speech recognition, and natural language processing.

        Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.
        Applications include machine translation, chatbots, sentiment analysis, text summarization.

        Retrieval-Augmented Generation (RAG) is an AI framework that retrieves facts from an external 
        knowledge base to improve the accuracy of large language model outputs. RAG reduces hallucination 
        and keeps LLM responses grounded in real data.

        Large Language Models (LLMs) are deep learning models trained on massive text datasets. 
        Examples include GPT-4, Claude, Gemini, and LLaMA. They can generate, summarize, translate, 
        and reason about text.

        Vector databases store high-dimensional embeddings and support similarity search. Popular options 
        include FAISS (Facebook AI), Chroma, Pinecone, Weaviate, and Qdrant.

        Embeddings are dense numerical representations of text that capture semantic meaning. 
        Similar texts have embeddings that are close together in vector space.

        LangChain is a framework for developing applications powered by LLMs. It provides components 
        for prompt management, chains, agents, memory, and retrieval.

        Fine-tuning adapts a pre-trained model to a specific task using a smaller dataset. 
        LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method.
        """
        raw_docs = [Document(page_content=sample_content, metadata={"source": "sample"})]

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(raw_docs)
    print(f"   Created {len(chunks)} chunks.")

    # Build vector store
    print("   Generating embeddings (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Knowledge base ready!\n")
    return vectorstore


def retrieve_context(vectorstore, query, k=TOP_K):
    """Retrieve top-k relevant chunks for a query."""
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(docs)])
    return context, docs


def generate_answer(query, context, chat_history):
    """Generate an answer using Claude with retrieved context."""
    system_prompt = (
        "You are a helpful AI assistant. Answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say 'I don't have enough information to answer that.' "
        "Be concise and accurate."
    )

    # Build messages with chat history
    messages = chat_history.copy()
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=512,
        system=system_prompt,
        messages=messages
    )
    return response.content[0].text


def rag_chat(pdf_path=None):
    """Run the RAG-based chat loop."""
    print("=" * 60)
    print("  E04: RAG-Based Question Answering Chat System")
    print("=" * 60)

    vectorstore = build_knowledge_base(pdf_path)

    print("💬 RAG Chat System Ready!")
    print("   Type your question and press Enter.")
    print("   Type 'quit' or 'exit' to stop.\n")

    chat_history = []

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break

        # Retrieve relevant context
        context, source_chunks = retrieve_context(vectorstore, user_input)

        print(f"\n📚 Retrieved {len(source_chunks)} relevant chunk(s).")

        # Generate answer
        answer = generate_answer(user_input, context, chat_history)

        print(f"Bot: {answer}\n")

        # Update chat history (keep last 6 turns to avoid token overflow)
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": answer})
        if len(chat_history) > 12:
            chat_history = chat_history[-12:]


if __name__ == "__main__":
    PDF_PATH = "sample.pdf"   # ← Replace with your PDF path (optional)
    rag_chat(pdf_path=PDF_PATH if os.path.exists(PDF_PATH) else None)
    print("\n✅ Experiment E04 Completed Successfully!")
