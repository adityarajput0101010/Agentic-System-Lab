# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E01
# Title: Implementation of Basic Prompting Techniques for LLMs
# ============================================================

from anthropic import Anthropic

client = Anthropic()

def send_prompt(prompt, system=None):
    """Send a prompt to Claude and return the response."""
    kwargs = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 512,
        "messages": [{"role": "user", "content": prompt}]
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)
    return response.content[0].text


def zero_shot_prompting():
    print("\n" + "="*60)
    print("1. ZERO-SHOT PROMPTING")
    print("="*60)
    prompt = "Translate the following sentence to French: 'Artificial Intelligence is transforming the world.'"
    print(f"Prompt: {prompt}")
    print(f"Response: {send_prompt(prompt)}")


def instructional_prompting():
    print("\n" + "="*60)
    print("2. INSTRUCTIONAL PROMPTING")
    print("="*60)
    prompt = (
        "List 5 applications of Machine Learning in healthcare. "
        "Format your answer as a numbered list with one sentence explanation for each."
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {send_prompt(prompt)}")


def role_based_prompting():
    print("\n" + "="*60)
    print("3. ROLE-BASED PROMPTING")
    print("="*60)
    system = "You are an expert Python programmer with 10 years of experience. Explain concepts simply."
    prompt = "What is a Python decorator and when should I use it?"
    print(f"System Role: {system}")
    print(f"Prompt: {prompt}")
    print(f"Response: {send_prompt(prompt, system=system)}")


def contextual_prompting():
    print("\n" + "="*60)
    print("4. CONTEXTUAL PROMPTING")
    print("="*60)
    prompt = (
        "Context: A student is preparing for a data science interview at a top tech company. "
        "They have 2 weeks left and know basic Python but not machine learning.\n\n"
        "Question: Create a 2-week study plan for this student."
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {send_prompt(prompt)}")


def negative_prompting():
    print("\n" + "="*60)
    print("5. NEGATIVE PROMPTING (What NOT to do)")
    print("="*60)
    prompt = (
        "Explain what a neural network is. "
        "Do NOT use technical jargon. Do NOT write more than 3 sentences. "
        "Do NOT use analogies involving the human brain."
    )
    print(f"Prompt: {prompt}")
    print(f"Response: {send_prompt(prompt)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  E01: Basic Prompting Techniques for LLMs")
    print("=" * 60)

    zero_shot_prompting()
    instructional_prompting()
    role_based_prompting()
    contextual_prompting()
    negative_prompting()

    print("\n✅ Experiment E01 Completed Successfully!")
