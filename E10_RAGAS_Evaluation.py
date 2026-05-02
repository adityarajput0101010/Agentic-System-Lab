# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E10
# Title: RAG Evaluation using RAGAS Evaluation Metrics
# ============================================================
# Install: pip install ragas langchain langchain-anthropic
#          pip install langchain-community faiss-cpu sentence-transformers datasets

import os
import json
from datasets import Dataset
from anthropic import Anthropic

# ── RAGAS imports ──────────────────────────────────────────
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

client = Anthropic()

# ─────────────────────────────────────────────
#  KNOWLEDGE BASE (simulated document store)
# ─────────────────────────────────────────────
DOCUMENTS = [
    "RAG stands for Retrieval-Augmented Generation. It is a technique that combines information retrieval with language model generation. RAG retrieves relevant documents from a knowledge base and uses them as context for the LLM to generate accurate, factual responses.",
    "LangChain is an open-source framework for building LLM-powered applications. It provides modules for prompt management, chains, agents, memory, and document retrieval. LangChain supports integration with OpenAI, Anthropic, HuggingFace, and many other LLM providers.",
    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It is widely used for building vector databases in RAG systems. FAISS supports both exact and approximate nearest neighbor search.",
    "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique. Instead of updating all model parameters, LoRA injects trainable low-rank matrices into transformer layers. This reduces trainable parameters by over 90% compared to full fine-tuning.",
    "The Transformer architecture was introduced in the paper 'Attention Is All You Need' by Vaswani et al. in 2017. It uses self-attention mechanisms to process sequences in parallel. Transformers are the foundation of modern LLMs like GPT, BERT, and Claude.",
    "Vector embeddings are dense numerical representations of text in a high-dimensional space. Similar texts have embeddings that are close together in vector space. Embedding models like sentence-transformers convert text to vectors for semantic search.",
    "RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG pipelines. It provides metrics including faithfulness, answer relevancy, context precision, context recall, and answer correctness. RAGAS uses LLMs as evaluators for nuanced scoring.",
    "Hallucination in LLMs refers to generating confident but factually incorrect information. RAG reduces hallucination by grounding responses in retrieved factual context. Faithfulness metric in RAGAS specifically measures hallucination rate.",
    "Chain-of-Thought (CoT) prompting encourages LLMs to reason step-by-step before giving a final answer. Zero-shot CoT adds 'Let's think step by step' to the prompt. Few-shot CoT provides example reasoning chains for the model to follow.",
    "LangGraph is a library built on top of LangChain for building stateful, multi-actor workflows using graph structures. It supports conditional branching, cycles, and state persistence. LangGraph is ideal for complex agentic workflows.",
]


# ─────────────────────────────────────────────
#  SIMPLE RAG PIPELINE (without vector DB for simplicity)
# ─────────────────────────────────────────────
def simple_retriever(query: str, top_k: int = 3) -> list[str]:
    """Retrieve top-k most relevant documents using keyword matching."""
    scores = []
    query_words = set(query.lower().split())
    for doc in DOCUMENTS:
        doc_words = set(doc.lower().split())
        overlap   = len(query_words & doc_words)
        scores.append((overlap, doc))
    scores.sort(reverse=True)
    return [doc for _, doc in scores[:top_k]]


def generate_answer(question: str, contexts: list[str]) -> str:
    """Generate answer using Claude with retrieved contexts."""
    context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"Be concise and accurate.\n\n"
        f"{context_text}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def run_rag_pipeline(questions: list[str]) -> list[dict]:
    """Run RAG pipeline on a list of questions and return results."""
    results = []
    print("\n🔄 Running RAG Pipeline...")
    for q in questions:
        contexts = simple_retriever(q)
        answer   = generate_answer(q, contexts)
        results.append({
            "question": q,
            "answer":   answer,
            "contexts": contexts
        })
        print(f"   Q: {q[:60]}...")
        print(f"   A: {answer[:80]}...\n")
    return results


# ─────────────────────────────────────────────
#  EVALUATION DATASET
# ─────────────────────────────────────────────
QA_PAIRS = [
    {
        "question":     "What is RAG and how does it work?",
        "ground_truth": "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with language model generation to produce accurate, grounded responses."
    },
    {
        "question":     "What is LoRA and why is it used?",
        "ground_truth": "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into transformer layers, reducing trainable parameters by over 90%."
    },
    {
        "question":     "What is LangChain used for?",
        "ground_truth": "LangChain is an open-source framework for building LLM-powered applications with components for prompt management, chains, agents, memory, and document retrieval."
    },
    {
        "question":     "What is RAGAS and what metrics does it provide?",
        "ground_truth": "RAGAS is a framework for evaluating RAG pipelines providing metrics like faithfulness, answer relevancy, context precision, context recall, and answer correctness."
    },
    {
        "question":     "What is the Transformer architecture?",
        "ground_truth": "The Transformer architecture introduced in 2017 uses self-attention mechanisms to process sequences in parallel and is the foundation of modern LLMs like GPT, BERT, and Claude."
    },
]


# ─────────────────────────────────────────────
#  RAGAS EVALUATION
# ─────────────────────────────────────────────
def evaluate_with_ragas(rag_results: list[dict], ground_truths: list[str]) -> dict:
    print("\n" + "="*60)
    print("📊 Running RAGAS Evaluation...")
    print("="*60)

    eval_data = {
        "question":     [r["question"]     for r in rag_results],
        "answer":       [r["answer"]       for r in rag_results],
        "contexts":     [r["contexts"]     for r in rag_results],
        "ground_truth": ground_truths
    }

    dataset = Dataset.from_dict(eval_data)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    print("   Running metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness")
    result = evaluate(dataset, metrics=metrics)
    return result


# ─────────────────────────────────────────────
#  DISPLAY RESULTS
# ─────────────────────────────────────────────
def display_results(result):
    print("\n" + "="*60)
    print("📈 RAGAS EVALUATION RESULTS")
    print("="*60)

    metrics_info = {
        "faithfulness":       ("Is the answer grounded in the retrieved context?", 0.8),
        "answer_relevancy":   ("Is the answer relevant to the question?",          0.8),
        "context_precision":  ("Are retrieved contexts useful (signal-to-noise)?", 0.7),
        "context_recall":     ("Does context contain the necessary information?",  0.7),
        "answer_correctness": ("Is the answer factually correct vs ground truth?", 0.6),
    }

    scores = {}
    for metric_name, (description, threshold) in metrics_info.items():
        try:
            score = result[metric_name]
            scores[metric_name] = score
            status = "✅" if score >= threshold else "⚠️ "
            print(f"\n  {status} {metric_name.upper()}: {score:.4f}")
            print(f"     → {description}")
            print(f"     → Threshold: {threshold} | Status: {'PASS' if score >= threshold else 'NEEDS IMPROVEMENT'}")
        except Exception:
            print(f"\n  ❓ {metric_name}: Not available")

    # Overall score
    if scores:
        overall = sum(scores.values()) / len(scores)
        print(f"\n{'='*60}")
        print(f"  🏆 OVERALL RAG QUALITY SCORE: {overall:.4f}")
        if overall >= 0.75:
            print("  → Excellent RAG pipeline!")
        elif overall >= 0.60:
            print("  → Good RAG pipeline, minor improvements needed.")
        else:
            print("  → RAG pipeline needs significant improvement.")
        print("="*60)

    return scores


# ─────────────────────────────────────────────
#  IMPROVEMENT SUGGESTIONS
# ─────────────────────────────────────────────
def suggest_improvements(scores: dict):
    print("\n" + "="*60)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*60)

    suggestions = {
        "faithfulness": [
            "Use stricter system prompts: 'Answer ONLY from the context'",
            "Reduce LLM temperature to lower hallucination",
            "Add post-processing to verify answer against context",
        ],
        "answer_relevancy": [
            "Improve prompt to focus on the specific question",
            "Use query rewriting before retrieval",
            "Filter out unrelated context chunks",
        ],
        "context_precision": [
            "Use better embedding models (e.g., text-embedding-3-large)",
            "Increase chunk overlap for better boundary handling",
            "Use hybrid search (keyword + semantic)",
        ],
        "context_recall": [
            "Increase top-k retrieved documents",
            "Use hierarchical retrieval (coarse-to-fine)",
            "Add metadata filtering to improve retrieval scope",
        ],
        "answer_correctness": [
            "Fine-tune the model on domain-specific data",
            "Add answer verification step using a second LLM",
            "Include more diverse and comprehensive documents",
        ],
    }

    for metric, score in scores.items():
        if score < 0.75 and metric in suggestions:
            print(f"\n  📌 Improve '{metric}' (score: {score:.4f}):")
            for s in suggestions[metric]:
                print(f"     • {s}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  E10: RAG Evaluation using RAGAS Metrics")
    print("=" * 60)

    questions     = [qa["question"]     for qa in QA_PAIRS]
    ground_truths = [qa["ground_truth"] for qa in QA_PAIRS]

    # Step 1: Run RAG pipeline
    rag_results = run_rag_pipeline(questions)

    # Step 2: Evaluate with RAGAS
    try:
        result = evaluate_with_ragas(rag_results, ground_truths)
        scores = display_results(result)
        suggest_improvements(scores)
    except Exception as e:
        print(f"\n⚠️  RAGAS evaluation error: {e}")
        print("   Displaying raw RAG outputs instead:\n")
        for r in rag_results:
            print(f"   Q: {r['question']}")
            print(f"   A: {r['answer']}\n")

    print("\n✅ Experiment E10 Completed Successfully!")
