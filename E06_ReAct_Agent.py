# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E06
# Title: ReAct (Reasoning and Acting) Loop for Autonomous Agents
# ============================================================
# Install: pip install anthropic

import re
import math
import json
import anthropic

client = anthropic.Anthropic()
MODEL  = "claude-3-haiku-20240307"

# ─────────────────────────────────────────────
#  TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────
KNOWLEDGE_BASE = {
    "python":      "Python is a high-level, interpreted programming language known for simplicity and readability. Created by Guido van Rossum in 1991.",
    "machine learning": "Machine Learning is a subset of AI that enables systems to learn from data. Types: supervised, unsupervised, reinforcement learning.",
    "deep learning": "Deep Learning uses multi-layered neural networks. Key architectures: CNN (images), RNN/LSTM (sequences), Transformer (NLP).",
    "langchain":   "LangChain is a framework for building LLM-powered applications with chains, agents, memory, and retrieval components.",
    "rag":         "RAG (Retrieval-Augmented Generation) combines document retrieval with LLM generation for accurate, grounded responses.",
    "transformer": "Transformer is a neural network architecture introduced in 'Attention is All You Need' (2017). Foundation of modern LLMs.",
    "gpu":         "GPU (Graphics Processing Unit) accelerates parallel computations, essential for training deep learning models.",
    "numpy":       "NumPy is a Python library for numerical computing with support for arrays, matrices, and mathematical functions.",
}

def search_knowledge(query: str) -> str:
    query_lower = query.lower()
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower or query_lower in key:
            return value
    return f"No information found for '{query}'. Try a different search term."

def calculator(expression: str) -> str:
    try:
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

def word_count(text: str) -> str:
    return f"Words: {len(text.split())}, Characters: {len(text)}"

TOOLS = {
    "search":     search_knowledge,
    "calculator": calculator,
    "word_count": word_count,
}

TOOL_DESCRIPTIONS = {
    "search":     "Search the knowledge base. Input: a topic or keyword.",
    "calculator": "Evaluate a math expression. Input: expression like '5*8+3' or 'sqrt(144)'.",
    "word_count": "Count words/characters in text. Input: any text string.",
}

# ─────────────────────────────────────────────
#  ReAct SYSTEM PROMPT
# ─────────────────────────────────────────────
def build_react_system_prompt():
    tool_list = "\n".join([f"- {name}: {desc}" for name, desc in TOOL_DESCRIPTIONS.items()])
    return f"""You are an autonomous AI agent that solves tasks using the ReAct framework.
You interleave Thought, Action, and Observation steps until you reach the Final Answer.

Available Tools:
{tool_list}

FORMAT (follow strictly):
Thought: [Your reasoning about the current situation]
Action: [tool_name]("[input]")
Observation: [result will be filled in]
... (repeat Thought/Action/Observation as needed)
Final Answer: [Your complete answer to the original question]

Rules:
- Always start with a Thought.
- Use exactly one Action per step.
- After receiving an Observation, decide whether to continue or give Final Answer.
- Never make up tool results.
"""

# ─────────────────────────────────────────────
#  ReAct PARSER
# ─────────────────────────────────────────────
def parse_action(text: str):
    """Extract tool name and input from an Action line."""
    pattern = r'Action:\s*(\w+)\("?([^"]*)"?\)'
    match = re.search(pattern, text)
    if match:
        return match.group(1), match.group(2)
    # Try alternate format: Action: tool_name(input)
    pattern2 = r'Action:\s*(\w+)\((.+)\)'
    match2 = re.search(pattern2, text)
    if match2:
        return match2.group(1), match2.group(2).strip('"\'')
    return None, None

def extract_final_answer(text: str):
    match = re.search(r'Final Answer:\s*(.+)', text, re.DOTALL)
    return match.group(1).strip() if match else None

# ─────────────────────────────────────────────
#  ReAct AGENT LOOP
# ─────────────────────────────────────────────
def react_agent(question: str, max_steps: int = 8):
    print(f"\n{'='*60}")
    print(f"❓ Question: {question}")
    print("="*60)

    system_prompt = build_react_system_prompt()
    conversation = [{"role": "user", "content": f"Question: {question}"}]
    full_scratchpad = ""

    for step in range(1, max_steps + 1):
        print(f"\n--- Step {step} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=system_prompt,
            messages=conversation
        )
        agent_text = response.content[0].text
        print(agent_text)
        full_scratchpad += agent_text + "\n"

        # Check if Final Answer is reached
        final = extract_final_answer(agent_text)
        if final:
            print(f"\n✅ Final Answer: {final}")
            return final

        # Parse and execute action
        tool_name, tool_input = parse_action(agent_text)
        if tool_name and tool_name in TOOLS:
            observation = TOOLS[tool_name](tool_input)
            obs_text = f"Observation: {observation}"
            print(obs_text)
            full_scratchpad += obs_text + "\n"

            # Add to conversation
            conversation.append({"role": "assistant", "content": agent_text})
            conversation.append({"role": "user",      "content": obs_text})
        else:
            print("⚠️  No valid action found. Stopping.")
            break

    print("⚠️  Max steps reached without Final Answer.")
    return None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  E06: ReAct Loop for Autonomous Decision-Making")
    print("=" * 60)

    questions = [
        "What is LangChain and how many characters are in its description?",
        "What is 15 squared plus the square root of 256?",
        "Search for RAG and then count the words in its description.",
        "What is deep learning? Also calculate 2 to the power of 10.",
    ]

    for q in questions:
        react_agent(q)
        print()

    print("\n✅ Experiment E06 Completed Successfully!")
