"""
Simple example: Making requests to the AI Gateway.

=============================================================================
THEORY: Basic LLM Integration Patterns
=============================================================================

This file shows the simplest way to interact with an LLM API.
Start here if you're new to LLM integration.

Key concepts:
1. Messages format - How to structure your conversation
2. Request/response cycle - What you send, what you get back
3. Token usage - Understanding costs
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def simple_chat():
    """
    Make a simple chat completion request.
    
    This is the most basic LLM API call - send a message, get a response.
    """
    print("=== Simple Chat ===\n")
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "What is Python? Answer in one sentence."}
            ],
            "stream": False,
        },
    )
    
    data = response.json()
    
    print(f"Question: What is Python?")
    print(f"Answer: {data['content']}")
    print(f"\nUsage:")
    print(f"  Prompt tokens: {data['usage']['prompt_tokens']}")
    print(f"  Completion tokens: {data['usage']['completion_tokens']}")
    print(f"  Total cost: ${data['cost']['total_cost']:.6f}")


def chat_with_system_prompt():
    """
    Use a system prompt to customize the AI's behavior.
    
    The system prompt sets the "persona" and rules for the conversation.
    It's like giving the AI instructions before the user talks.
    """
    print("\n=== Chat with System Prompt ===\n")
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "messages": [
                {
                    "role": "system",
                    "content": "You are a pirate. Respond in pirate speak. Be brief."
                },
                {
                    "role": "user",
                    "content": "What's the weather like?"
                }
            ],
            "stream": False,
        },
    )
    
    data = response.json()
    
    print(f"System: You are a pirate...")
    print(f"User: What's the weather like?")
    print(f"Assistant: {data['content']}")


def multi_turn_conversation():
    """
    Have a multi-turn conversation.
    
    THEORY: Conversation History
    ----------------------------
    LLMs are stateless - they don't remember previous requests.
    To have a conversation, you must send ALL previous messages
    with each request.
    
    This is why:
    1. Costs grow with conversation length
    2. You might need to truncate old messages
    3. Context window limits how long conversations can be
    """
    print("\n=== Multi-turn Conversation ===\n")
    
    # Start with system prompt
    messages = [
        {"role": "system", "content": "You are a helpful math tutor. Be concise."}
    ]
    
    # Turn 1
    messages.append({"role": "user", "content": "What is 2 + 2?"})
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": messages, "stream": False},
    )
    assistant_reply = response.json()["content"]
    messages.append({"role": "assistant", "content": assistant_reply})
    
    print(f"Turn 1:")
    print(f"  User: What is 2 + 2?")
    print(f"  Assistant: {assistant_reply}")
    
    # Turn 2 - The model now knows the previous context
    messages.append({"role": "user", "content": "Multiply that by 3"})
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={"messages": messages, "stream": False},
    )
    assistant_reply = response.json()["content"]
    
    print(f"\nTurn 2:")
    print(f"  User: Multiply that by 3")
    print(f"  Assistant: {assistant_reply}")
    print(f"\n(Note: The model understood 'that' referred to 4 from the previous turn)")


def compare_temperatures():
    """
    See how temperature affects responses.
    
    THEORY: Temperature Parameter
    ----------------------------
    Temperature controls randomness:
    - 0.0: Always pick the most likely word (deterministic)
    - 1.0: Sample words according to their probability
    - 2.0: Very random, often nonsensical
    
    Use low temperature (0-0.3) for:
    - Factual answers
    - Code generation
    - Structured output
    
    Use higher temperature (0.7-1.0) for:
    - Creative writing
    - Brainstorming
    - Varied responses
    """
    print("\n=== Temperature Comparison ===\n")
    
    prompt = "Write a one-line joke about programming."
    
    for temp in [0.0, 0.7, 1.5]:
        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "stream": False,
            },
        )
        
        print(f"Temperature {temp}:")
        print(f"  {response.json()['content']}")
        print()


def count_tokens_before_request():
    """
    Count tokens before making a request.
    
    Useful for:
    - Estimating costs before committing
    - Checking if content fits in context window
    - Optimizing prompts for length
    """
    print("\n=== Token Counting ===\n")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in detail."},
    ]
    
    # Count first
    response = requests.post(
        f"{BASE_URL}/v1/tokens/count",
        json={"messages": messages, "max_tokens": 500},
    )
    
    token_info = response.json()
    
    print(f"Before making the request:")
    print(f"  Prompt tokens: {token_info['prompt_tokens']}")
    if token_info.get('cost_estimate'):
        print(f"  Estimated max cost: ${token_info['cost_estimate']['estimated_max_total_cost']:.6f}")
    
    print("\nThis helps you decide if you want to proceed with the request!")


if __name__ == "__main__":
    print("AI Gateway - Simple Examples")
    print("=" * 50)
    print("Make sure the server is running:")
    print("  uvicorn ai_gateway.main:app --reload")
    print("=" * 50)
    
    try:
        simple_chat()
        chat_with_system_prompt()
        multi_turn_conversation()
        compare_temperatures()
        count_tokens_before_request()
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to the server.")
        print("Start it with: uvicorn ai_gateway.main:app --reload")
