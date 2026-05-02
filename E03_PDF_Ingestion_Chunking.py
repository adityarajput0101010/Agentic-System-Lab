# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E03
# Title: PDF Document Ingestion and Text Chunking for RAG
# ============================================================
# Install: pip install langchain langchain-community pypdf tiktoken

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)


def load_pdf(pdf_path):
    """Load a PDF and return list of Document objects."""
    print(f"\n📄 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} page(s)")
    for i, doc in enumerate(documents):
        print(f"   Page {i+1}: {len(doc.page_content)} characters")
    return documents


def fixed_size_chunking(documents):
    """Chunk using fixed character size."""
    print("\n" + "="*60)
    print("1. FIXED-SIZE CHUNKING (CharacterTextSplitter)")
    print("="*60)

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    print(f"\n--- Chunk 1 ---\n{chunks[0].page_content[:300]}...")
    return chunks


def recursive_chunking(documents):
    """Chunk using recursive character splitting (recommended for RAG)."""
    print("\n" + "="*60)
    print("2. RECURSIVE CHARACTER TEXT SPLITTING")
    print("="*60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")
    print(f"\n--- Chunk 1 ---\n{chunks[0].page_content}")
    print(f"\n--- Chunk 2 ---\n{chunks[1].page_content if len(chunks) > 1 else 'N/A'}")
    return chunks


def analyze_chunks(chunks, label=""):
    """Print statistics about chunk sizes."""
    print(f"\n📊 Chunk Analysis {label}")
    sizes = [len(c.page_content) for c in chunks]
    print(f"   Total chunks   : {len(chunks)}")
    print(f"   Min chunk size : {min(sizes)} chars")
    print(f"   Max chunk size : {max(sizes)} chars")
    print(f"   Avg chunk size : {sum(sizes)//len(sizes)} chars")


def demo_with_sample_text():
    """Demo chunking using a synthetic document (no PDF needed)."""
    print("\n" + "="*60)
    print("3. DEMO WITH SYNTHETIC TEXT (No PDF required)")
    print("="*60)

    from langchain.schema import Document

    sample_text = """
    Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the 
    natural intelligence displayed by animals including humans. AI research has been defined as 
    the field of study of intelligent agents, which refers to any system that perceives its 
    environment and takes actions that maximize its chance of achieving its goals.

    Machine Learning is a method of data analysis that automates analytical model building. 
    It is based on the idea that systems can learn from data, identify patterns and make 
    decisions with minimal human intervention. Machine learning algorithms include supervised 
    learning, unsupervised learning, and reinforcement learning.

    Deep Learning is part of a broader family of machine learning methods based on artificial 
    neural networks with representation learning. Learning can be supervised, semi-supervised 
    or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural 
    networks, and convolutional neural networks have been applied to fields including computer 
    vision, speech recognition, natural language processing, and bioinformatics.

    Natural Language Processing (NLP) is a subfield of linguistics, computer science, and 
    artificial intelligence concerned with the interactions between computers and human language, 
    in particular how to program computers to process and analyze large amounts of natural language 
    data. NLP is used in machine translation, spam detection, information extraction, summarization, 
    medical diagnosis, and question answering.

    Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with 
    text generation. RAG retrieves relevant documents from a knowledge base and uses them as 
    additional context for a language model to generate accurate, factual responses. This approach 
    reduces hallucination and allows LLMs to access up-to-date information.
    """ * 3  # Repeat to create longer document

    docs = [Document(page_content=sample_text, metadata={"source": "synthetic", "page": 0})]

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    print(f"Original text length : {len(sample_text)} characters")
    print(f"Total chunks created : {len(chunks)}")

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)
        print(f"[Metadata: {chunk.metadata}]")

    analyze_chunks(chunks, "(Synthetic Document)")
    return chunks


def chunk_size_comparison(documents):
    """Compare different chunk sizes."""
    print("\n" + "="*60)
    print("4. CHUNK SIZE COMPARISON")
    print("="*60)

    for size in [200, 500, 1000]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size//10)
        chunks = splitter.split_documents(documents)
        print(f"Chunk size {size:4d} | Overlap {size//10:3d} | Total chunks: {len(chunks)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  E03: PDF Ingestion & Text Chunking for RAG")
    print("=" * 60)

    PDF_PATH = "sample.pdf"   # ← Replace with your PDF path

    if os.path.exists(PDF_PATH):
        documents = load_pdf(PDF_PATH)
        chunks_fixed = fixed_size_chunking(documents)
        chunks_recursive = recursive_chunking(documents)
        analyze_chunks(chunks_fixed, "(Fixed Size)")
        analyze_chunks(chunks_recursive, "(Recursive)")
        chunk_size_comparison(documents)
    else:
        print(f"\n⚠️  '{PDF_PATH}' not found. Running demo with synthetic text...\n")
        documents_synthetic = demo_with_sample_text()

    print("\n✅ Experiment E03 Completed Successfully!")
