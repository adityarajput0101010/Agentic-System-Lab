# ============================================================
# DKTE Textile & Engineering Institute, Ichalkaranji
# Department: CSE (AI & ML)
# Subject: 01AMP347 - Agentic Systems Lab
# Experiment No: E05
# Title: Tool-Using Intelligent Agent using LLMs
# ============================================================
# Install: pip install anthropic requests

import json
import math
import datetime
import anthropic

client = anthropic.Anthropic()
MODEL  = "claude-3-haiku-20240307"


# ─────────────────────────────────────────────
#  TOOL DEFINITIONS  (describe tools to Claude)
# ─────────────────────────────────────────────
tools = [
    {
        "name": "calculator",
        "description": (
            "Perform arithmetic calculations. Supports: add, subtract, multiply, divide, "
            "power, sqrt, and basic math expressions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate, e.g. '2 + 3 * 4', 'sqrt(144)', '2 ** 10'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Get the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Timezone name, e.g. 'UTC', 'IST'. Default is local time."
                }
            },
            "required": []
        }
    },
    {
        "name": "unit_converter",
        "description": "Convert values between common units (length, weight, temperature).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value":     {"type": "number", "description": "Numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit, e.g. 'km', 'kg', 'celsius'"},
                "to_unit":   {"type": "string", "description": "Target unit, e.g. 'miles', 'pounds', 'fahrenheit'"}
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    },
    {
        "name": "word_counter",
        "description": "Count words, characters, and sentences in a given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to analyze"}
            },
            "required": ["text"]
        }
    }
]


# ─────────────────────────────────────────────
#  TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────
def calculator(expression: str) -> str:
    try:
        # Safe evaluation with math functions available
        safe_env = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_env["sqrt"] = math.sqrt
        result = eval(expression, {"__builtins__": {}}, safe_env)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def get_current_datetime(timezone: str = "local") -> str:
    now = datetime.datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({timezone})"


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    conversions = {
        ("km", "miles"):      value * 0.621371,
        ("miles", "km"):      value * 1.60934,
        ("kg", "pounds"):     value * 2.20462,
        ("pounds", "kg"):     value / 2.20462,
        ("meters", "feet"):   value * 3.28084,
        ("feet", "meters"):   value / 3.28084,
        ("celsius", "fahrenheit"): (value * 9/5) + 32,
        ("fahrenheit", "celsius"): (value - 32) * 5/9,
        ("liters", "gallons"):     value * 0.264172,
        ("gallons", "liters"):     value / 0.264172,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key]
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Conversion from {from_unit} to {to_unit} not supported."


def word_counter(text: str) -> str:
    words      = len(text.split())
    characters = len(text)
    sentences  = text.count('.') + text.count('!') + text.count('?')
    return (
        f"Words: {words} | Characters: {characters} | "
        f"Sentences: {sentences} | Avg word length: {characters/max(words,1):.1f} chars"
    )


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Route tool calls to their implementations."""
    if tool_name == "calculator":
        return calculator(tool_input["expression"])
    elif tool_name == "get_current_datetime":
        return get_current_datetime(tool_input.get("timezone", "local"))
    elif tool_name == "unit_converter":
        return unit_converter(tool_input["value"], tool_input["from_unit"], tool_input["to_unit"])
    elif tool_name == "word_counter":
        return word_counter(tool_input["text"])
    else:
        return f"Unknown tool: {tool_name}"


# ─────────────────────────────────────────────
#  AGENT LOOP
# ─────────────────────────────────────────────
def run_agent(user_query: str):
    """Run the agent loop until a final answer is produced."""
    print(f"\n{'='*60}")
    print(f"User: {user_query}")
    print("="*60)

    messages = [{"role": "user", "content": user_query}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        print(f"\n[Agent thinking... stop_reason={response.stop_reason}]")

        # If Claude is done, print the final text answer
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"\nAgent: {block.text}")
            break

        # If Claude wants to use a tool
        if response.stop_reason == "tool_use":
            # Add Claude's response to history
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n🔧 Tool Called : {block.name}")
                    print(f"   Input       : {json.dumps(block.input, indent=2)}")
                    result = execute_tool(block.name, block.input)
                    print(f"   Result      : {result}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            # Feed tool results back to Claude
            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            print(f"Unexpected stop_reason: {response.stop_reason}")
            break


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  E05: Tool-Using Intelligent Agent")
    print("=" * 60)

    queries = [
        "What is 25 multiplied by 48, and what is the square root of the result?",
        "What is today's date and time?",
        "Convert 100 kilometers to miles and also convert 37 degrees Celsius to Fahrenheit.",
        "Count the words and characters in this sentence: 'Artificial Intelligence is transforming every industry in the modern world.'",
        "If a rectangle has length 15 km and width 8 km, what is its area? Also convert that area value from square km context: just tell me 15 * 8.",
    ]

    for query in queries:
        run_agent(query)
        print()

    print("\n✅ Experiment E05 Completed Successfully!")
