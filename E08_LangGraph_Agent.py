# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E08
# Title: Graph-Based Intelligent Agent using LangGraph
# ============================================================
# Install: pip install langgraph langchain langchain-anthropic

from typing import TypedDict, Annotated, Literal
import operator
import json
import math

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

llm = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=512)

# ─────────────────────────────────────────────
#  STATE DEFINITION
# ─────────────────────────────────────────────
class ResearchState(TypedDict):
    query:       str
    search_results: str
    analysis:    str
    summary:     str
    final_report: str
    iteration:   int
    status:      str


# ─────────────────────────────────────────────
#  NODE FUNCTIONS
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "artificial intelligence": "AI is the simulation of human intelligence in machines. Key areas: ML, NLP, Computer Vision, Robotics. Founded at Dartmouth Conference 1956.",
    "machine learning": "ML enables systems to learn from data. Types: Supervised (labeled data), Unsupervised (patterns), Reinforcement (reward-based). Key algorithms: SVM, Decision Trees, Neural Networks.",
    "deep learning": "Deep Learning uses deep neural networks with many layers. Architectures: CNN, RNN, LSTM, Transformer. Breakthroughs: ImageNet, AlphaGo, GPT, BERT.",
    "natural language processing": "NLP enables computers to understand human language. Tasks: tokenization, POS tagging, NER, sentiment analysis, machine translation, QA.",
    "large language models": "LLMs are trained on massive text datasets. Examples: GPT-4, Claude, Gemini, LLaMA. Capabilities: text generation, summarization, reasoning, coding.",
    "computer vision": "CV enables machines to interpret visual data. Tasks: image classification, object detection, segmentation, OCR. Key models: ResNet, YOLO, ViT.",
    "reinforcement learning": "RL trains agents via reward/penalty signals. Components: Agent, Environment, State, Action, Reward. Algorithms: Q-Learning, PPO, A3C.",
}

def search_node(state: ResearchState) -> ResearchState:
    """Node 1: Search for information about the query."""
    print(f"\n🔍 [Node: Search] Searching for: {state['query']}")
    query_lower = state["query"].lower()

    results = []
    for key, value in KNOWLEDGE_BASE.items():
        if any(word in query_lower for word in key.split()) or key in query_lower:
            results.append(f"[{key.upper()}]: {value}")

    if not results:
        results = ["General AI information: AI encompasses machine learning, deep learning, NLP, and computer vision."]

    search_results = "\n\n".join(results)
    print(f"   Found {len(results)} result(s).")
    return {**state, "search_results": search_results, "status": "searched"}


def analysis_node(state: ResearchState) -> ResearchState:
    """Node 2: Analyze the search results."""
    print(f"\n🧠 [Node: Analysis] Analyzing results...")

    prompt = f"""Analyze the following information about '{state['query']}' and extract key insights:

Information:
{state['search_results']}

Provide:
1. Main concepts identified
2. Key relationships between concepts
3. Important technical details
4. Practical applications

Keep analysis concise (150 words max)."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"   Analysis complete.")
    return {**state, "analysis": response.content, "status": "analyzed"}


def quality_check_node(state: ResearchState) -> ResearchState:
    """Node 3: Check if analysis is sufficient."""
    print(f"\n✅ [Node: Quality Check] Checking analysis quality...")

    word_count = len(state["analysis"].split())
    has_key_points = "1." in state["analysis"] or "•" in state["analysis"] or "-" in state["analysis"]
    iteration = state.get("iteration", 0) + 1

    if word_count > 50 and has_key_points and iteration <= 2:
        status = "quality_ok"
        print(f"   Quality OK (words: {word_count}, iteration: {iteration})")
    elif iteration > 2:
        status = "quality_ok"  # Force proceed after 2 iterations
        print(f"   Max iterations reached. Proceeding.")
    else:
        status = "needs_retry"
        print(f"   Quality insufficient (words: {word_count}). Will retry.")

    return {**state, "iteration": iteration, "status": status}


def summarize_node(state: ResearchState) -> ResearchState:
    """Node 4: Summarize findings."""
    print(f"\n📝 [Node: Summarize] Creating summary...")

    prompt = f"""Create a concise 3-bullet-point summary of the research on '{state['query']}':

Analysis:
{state['analysis']}

Format:
• Point 1
• Point 2  
• Point 3"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print(f"   Summary created.")
    return {**state, "summary": response.content, "status": "summarized"}


def report_node(state: ResearchState) -> ResearchState:
    """Node 5: Generate final structured report."""
    print(f"\n📊 [Node: Report] Generating final report...")

    report = f"""
╔══════════════════════════════════════════════════════════╗
   RESEARCH REPORT: {state['query'].upper()}
╚══════════════════════════════════════════════════════════╝

📌 TOPIC: {state['query']}

🔍 RAW INFORMATION:
{state['search_results']}

🧠 ANALYSIS:
{state['analysis']}

📋 SUMMARY:
{state['summary']}

📊 METADATA:
   - Search results found : {len(state['search_results'].split('['))-1}
   - Analysis iterations  : {state['iteration']}
   - Status               : Complete
"""
    print(f"   Report generated.")
    return {**state, "final_report": report, "status": "complete"}


# ─────────────────────────────────────────────
#  CONDITIONAL EDGE
# ─────────────────────────────────────────────
def should_retry(state: ResearchState) -> Literal["analysis", "summarize"]:
    """Decide whether to re-analyze or proceed to summarize."""
    if state["status"] == "needs_retry":
        print("   ↩️  Routing back to analysis node for retry...")
        return "analysis"
    else:
        print("   ➡️  Routing to summarize node...")
        return "summarize"


# ─────────────────────────────────────────────
#  BUILD THE GRAPH
# ─────────────────────────────────────────────
def build_research_graph():
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("search",        search_node)
    graph.add_node("analysis",      analysis_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("summarize",     summarize_node)
    graph.add_node("report",        report_node)

    # Add edges
    graph.set_entry_point("search")
    graph.add_edge("search",        "analysis")
    graph.add_edge("analysis",      "quality_check")
    graph.add_conditional_edges(
        "quality_check",
        should_retry,
        {"analysis": "analysis", "summarize": "summarize"}
    )
    graph.add_edge("summarize", "report")
    graph.add_edge("report",    END)

    return graph.compile()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run_research_pipeline(query: str):
    print(f"\n{'='*60}")
    print(f"  🚀 Starting Research Pipeline")
    print(f"  Query: {query}")
    print("="*60)

    app = build_research_graph()

    initial_state: ResearchState = {
        "query":          query,
        "search_results": "",
        "analysis":       "",
        "summary":        "",
        "final_report":   "",
        "iteration":      0,
        "status":         "start"
    }

    final_state = app.invoke(initial_state)
    print(final_state["final_report"])
    return final_state


if __name__ == "__main__":
    print("=" * 60)
    print("  E08: Graph-Based Agent using LangGraph")
    print("=" * 60)

    queries = [
        "Large Language Models",
        "Reinforcement Learning",
        "Natural Language Processing",
    ]

    for query in queries:
        run_research_pipeline(query)
        print("\n" + "="*60)

    print("\n✅ Experiment E08 Completed Successfully!")
