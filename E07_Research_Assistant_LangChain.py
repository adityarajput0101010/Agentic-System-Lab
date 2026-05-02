# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E07
# Title: AI-Based Research Assistant using LangChain Framework
# ============================================================
# Install: pip install langchain langchain-anthropic langchain-community
#          pip install wikipedia duckduckgo-search

import os
from langchain_anthropic import ChatAnthropic
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# ── LLM Setup ─────────────────────────────────────────────
llm = ChatAnthropic(model="claude-3-haiku-20240307", max_tokens=1024)

# ─────────────────────────────────────────────
#  TOOL 1: Wikipedia Search
# ─────────────────────────────────────────────
wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

# ─────────────────────────────────────────────
#  TOOL 2: DuckDuckGo Web Search
# ─────────────────────────────────────────────
ddg_search = DuckDuckGoSearchAPIWrapper(max_results=3)

def web_search(query: str) -> str:
    results = ddg_search.results(query, max_results=3)
    formatted = []
    for r in results:
        formatted.append(f"Title: {r.get('title','')}\nSnippet: {r.get('body','')}\nURL: {r.get('href','')}")
    return "\n\n".join(formatted) if formatted else "No results found."

# ─────────────────────────────────────────────
#  TOOL 3: Summarizer
# ─────────────────────────────────────────────
def summarize_text(text: str) -> str:
    """Summarize long text using LangChain summarization chain."""
    docs = [Document(page_content=text)]
    prompt_template = """Write a concise and informative summary of the following text in 3-5 bullet points:

{text}

SUMMARY:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    result = chain.run(docs)
    return result

# ─────────────────────────────────────────────
#  TOOL 4: Research Outline Generator
# ─────────────────────────────────────────────
def generate_outline(topic: str) -> str:
    """Generate a structured research outline for a topic."""
    prompt = PromptTemplate(
        input_variables=["topic"],
        template=(
            "Create a detailed research outline for the topic: '{topic}'\n\n"
            "Include:\n"
            "1. Introduction and Background\n"
            "2. Key Concepts\n"
            "3. Current State of Research\n"
            "4. Applications\n"
            "5. Challenges and Limitations\n"
            "6. Future Directions\n"
            "7. References (suggest 3 key papers/books)\n\n"
            "Outline:"
        )
    )
    chain = prompt | llm
    result = chain.invoke({"topic": topic})
    return result.content if hasattr(result, "content") else str(result)


# ─────────────────────────────────────────────
#  BUILD AGENT WITH TOOLS + MEMORY
# ─────────────────────────────────────────────
tools = [
    Tool(
        name="Wikipedia",
        func=wiki_tool.run,
        description="Search Wikipedia for factual information about people, places, concepts, history, science."
    ),
    Tool(
        name="WebSearch",
        func=web_search,
        description="Search the web using DuckDuckGo for recent news, articles, and information."
    ),
    Tool(
        name="Summarizer",
        func=summarize_text,
        description="Summarize a long piece of text into concise bullet points. Input: the text to summarize."
    ),
    Tool(
        name="ResearchOutline",
        func=generate_outline,
        description="Generate a structured research outline for a given topic. Input: the topic name."
    ),
]

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)


# ─────────────────────────────────────────────
#  DEMO RESEARCH TASKS
# ─────────────────────────────────────────────
def run_demo_tasks():
    print("=" * 60)
    print("  E07: AI Research Assistant using LangChain")
    print("=" * 60)

    tasks = [
        "What is the Transformer architecture in machine learning? Give me a brief overview.",
        "Generate a research outline for 'Applications of Large Language Models in Healthcare'.",
        "Search for recent information about GPT-4 and summarize what you find.",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"Research Task {i}: {task}")
        print("="*60)
        try:
            result = agent.run(task)
            print(f"\n📋 Result:\n{result}")
        except Exception as e:
            print(f"Error: {e}")


# ─────────────────────────────────────────────
#  INTERACTIVE CHAT
# ─────────────────────────────────────────────
def interactive_chat():
    print("\n" + "="*60)
    print("  💬 Interactive Research Assistant")
    print("  Type 'quit' to exit | 'demo' to run demo tasks")
    print("="*60)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Goodbye!")
            break
        if user_input.lower() == "demo":
            run_demo_tasks()
            continue

        try:
            response = agent.run(user_input)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    MODE = "demo"   # Change to "chat" for interactive mode

    if MODE == "demo":
        run_demo_tasks()
    else:
        interactive_chat()

    print("\n✅ Experiment E07 Completed Successfully!")
