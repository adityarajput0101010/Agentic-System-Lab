# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E02
# Title: Advanced Prompting - Chain-of-Thought & Few-Shot Learning
# ============================================================

from anthropic import Anthropic

client = Anthropic()

def send_prompt(prompt):
    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def few_shot_prompting():
    print("\n" + "="*60)
    print("1. FEW-SHOT PROMPTING")
    print("="*60)

    prompt = """Classify the sentiment of each sentence as Positive, Negative, or Neutral.

Examples:
Sentence: "I love this product, it works perfectly!"
Sentiment: Positive

Sentence: "The delivery was late and the packaging was damaged."
Sentiment: Negative

Sentence: "The package arrived today."
Sentiment: Neutral

Sentence: "This is the worst experience I have ever had."
Sentiment: Negative

Now classify:
Sentence: "The camera quality of this phone is absolutely stunning!"
Sentiment:"""

    print("Prompt (Few-Shot):")
    print(prompt)
    print(f"\nModel Response: {send_prompt(prompt)}")


def chain_of_thought_prompting():
    print("\n" + "="*60)
    print("2. CHAIN-OF-THOUGHT (CoT) PROMPTING")
    print("="*60)

    prompt = """Solve the following math problem step by step:

A train travels from City A to City B at 60 km/h. 
The return journey is at 40 km/h. 
If the distance between the cities is 120 km, 
what is the average speed for the entire trip?

Think through this step by step:"""

    print("Prompt (CoT):")
    print(prompt)
    print(f"\nModel Response:\n{send_prompt(prompt)}")


def zero_shot_cot():
    print("\n" + "="*60)
    print("3. ZERO-SHOT CHAIN-OF-THOUGHT")
    print("="*60)

    prompt = """If a shirt costs $25 and is on sale for 20% off, 
and there is an additional 10% coupon, 
what is the final price? 
Let's think step by step."""

    print("Prompt (Zero-Shot CoT):")
    print(prompt)
    print(f"\nModel Response:\n{send_prompt(prompt)}")


def few_shot_cot():
    print("\n" + "="*60)
    print("4. FEW-SHOT CHAIN-OF-THOUGHT")
    print("="*60)

    prompt = """Answer math questions by showing your reasoning step by step.

Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. 
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger starts with 5 tennis balls.
   He buys 2 cans × 3 balls = 6 new tennis balls.
   Total = 5 + 6 = 11 tennis balls.

Q: The cafeteria had 23 apples. They used 20 for lunch and bought 6 more. 
   How many apples do they have?
A: Start with 23 apples.
   Used 20, so 23 - 20 = 3 apples remain.
   Bought 6 more, so 3 + 6 = 9 apples.

Q: A library has 48 books. They receive a donation of 3 boxes, 
   each containing 12 books. Then they lend out 15 books. 
   How many books does the library have now?
A:"""

    print("Prompt (Few-Shot CoT):")
    print(prompt)
    print(f"\nModel Response:\n{send_prompt(prompt)}")


def compare_with_without_cot():
    print("\n" + "="*60)
    print("5. COMPARISON: With vs Without CoT")
    print("="*60)

    question = "John is twice as old as Mary. In 5 years, John will be 1.5 times Mary's age. How old are they now?"

    # Without CoT
    prompt_no_cot = question
    print("Without CoT:")
    print(f"Prompt: {prompt_no_cot}")
    print(f"Response: {send_prompt(prompt_no_cot)}\n")

    # With CoT
    prompt_with_cot = question + "\nLet's think step by step and solve using algebra:"
    print("With CoT:")
    print(f"Prompt: {prompt_with_cot}")
    print(f"Response:\n{send_prompt(prompt_with_cot)}")


if __name__ == "__main__":
    print("=" * 60)
    print("  E02: Chain-of-Thought & Few-Shot Prompting")
    print("=" * 60)

    few_shot_prompting()
    chain_of_thought_prompting()
    zero_shot_cot()
    few_shot_cot()
    compare_with_without_cot()

    print("\n✅ Experiment E02 Completed Successfully!")
