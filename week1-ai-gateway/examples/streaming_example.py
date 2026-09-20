"""
Streaming example: How to consume Server-Sent Events.

=============================================================================
THEORY: Server-Sent Events (SSE) for LLM Streaming
=============================================================================

Why streaming matters:
1. User Experience - See responses as they're generated
2. Time to First Token (TTFT) - Critical UX metric
3. Early Cancellation - Stop generation if going wrong
4. Perceived Performance - Feels faster even if total time is same

SSE Protocol:
-------------
SSE is a simple text-based protocol:

    event: chunk
    data: {"delta": "Hello"}
    
    event: chunk  
    data: {"delta": " World"}
    
    event: complete
    data: {"usage": {...}}
    
    event: done
    data: [DONE]

Key points:
- Each message ends with double newline (\\n\\n)
- 'event:' line specifies the event type
- 'data:' line contains the payload
- Connection stays open until server closes it
"""

import httpx
import asyncio
import json
import sys

BASE_URL = "http://localhost:8000"


# =============================================================================
# Synchronous Streaming (Simple but Blocking)
# =============================================================================

def stream_sync():
    """
    Stream a response synchronously using httpx.
    
    This is simpler but blocks the main thread.
    Good for scripts and simple CLI tools.
    """
    print("=== Synchronous Streaming ===\n")
    print("Question: Explain how computers work in 3 sentences.\n")
    print("Answer: ", end="", flush=True)
    
    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Explain how computers work in 3 sentences."}
                ],
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            buffer = ""
            for chunk in response.iter_text():
                buffer += chunk
                
                # Process complete events
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    
                    event_type = "message"
                    event_data = ""
                    
                    for line in event_text.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()
                    
                    if event_type == "chunk" and event_data:
                        try:
                            data = json.loads(event_data)
                            # Print each token as it arrives
                            print(data.get("delta", ""), end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                    
                    elif event_type == "complete" and event_data:
                        try:
                            data = json.loads(event_data)
                            print("\n")
                            print(f"Tokens: {data['usage']['total_tokens']}")
                            print(f"Cost: ${data['cost']['total_cost']:.6f}")
                            print(f"Latency: {data['latency_ms']:.0f}ms")
                        except json.JSONDecodeError:
                            pass


# =============================================================================
# Asynchronous Streaming (Non-Blocking)
# =============================================================================

async def stream_async():
    """
    Stream a response asynchronously.
    
    THEORY: Why Async for Streaming?
    --------------------------------
    Async streaming allows:
    1. Non-blocking I/O - Other tasks can run while waiting
    2. Multiple streams - Handle many users concurrently
    3. Better resource usage - Event loop manages connections
    
    This is how production servers handle thousands of concurrent streams.
    """
    print("\n=== Asynchronous Streaming ===\n")
    print("Question: Write a haiku about coding.\n")
    print("Answer: ", end="", flush=True)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Write a haiku about coding."}
                ],
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    
                    event_type = "message"
                    event_data = ""
                    
                    for line in event_text.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()
                    
                    if event_type == "chunk" and event_data:
                        try:
                            data = json.loads(event_data)
                            print(data.get("delta", ""), end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                    
                    elif event_type == "complete":
                        print("\n")


# =============================================================================
# Streaming with Token-by-Token Processing
# =============================================================================

async def stream_with_processing():
    """
    Process each token as it arrives.
    
    THEORY: Real-time Token Processing
    ----------------------------------
    You can do useful things as tokens arrive:
    1. Syntax highlighting - Detect and color code blocks
    2. Safety checks - Stop if detecting harmful content
    3. Structured extraction - Parse JSON/XML as it forms
    4. Progress tracking - Count tokens in real-time
    
    This example counts words and characters as they arrive.
    """
    print("\n=== Streaming with Real-time Processing ===\n")
    print("Processing tokens as they arrive...\n")
    
    word_count = 0
    char_count = 0
    full_text = ""
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "List 5 programming languages and one thing each is good for."}
                ],
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            buffer = ""
            async for chunk in response.aiter_text():
                buffer += chunk
                
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    
                    event_type = "message"
                    event_data = ""
                    
                    for line in event_text.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()
                    
                    if event_type == "chunk" and event_data:
                        try:
                            data = json.loads(event_data)
                            delta = data.get("delta", "")
                            
                            # Process each token
                            full_text += delta
                            char_count += len(delta)
                            word_count = len(full_text.split())
                            
                            # Print with live stats
                            print(delta, end="", flush=True)
                            
                        except json.JSONDecodeError:
                            pass
                    
                    elif event_type == "complete":
                        print("\n")
                        print(f"Final stats: {word_count} words, {char_count} characters")


# =============================================================================
# Compare Streaming vs Non-Streaming
# =============================================================================

async def compare_streaming_modes():
    """
    Compare perceived latency between streaming and non-streaming.
    
    THEORY: Time to First Token (TTFT)
    ----------------------------------
    TTFT measures how quickly the user sees the first response token.
    
    Non-streaming: TTFT = Total generation time (user waits for everything)
    Streaming: TTFT = ~100-500ms (user sees response immediately)
    
    For long responses, streaming dramatically improves perceived performance.
    """
    print("\n=== Streaming vs Non-Streaming Comparison ===\n")
    
    import time
    
    prompt = "Write a paragraph about the history of programming."
    
    # Non-streaming
    print("Non-streaming mode:")
    print("-" * 40)
    start = time.time()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
        )
        first_byte = time.time()
        data = response.json()
        done = time.time()
    
    print(f"Time to first byte: {(first_byte - start) * 1000:.0f}ms")
    print(f"Total time: {(done - start) * 1000:.0f}ms")
    print(f"(User sees nothing until {(first_byte - start) * 1000:.0f}ms)")
    
    # Streaming
    print("\nStreaming mode:")
    print("-" * 40)
    start = time.time()
    first_token_time = None
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            headers={"Accept": "text/event-stream"},
        ) as response:
            buffer = ""
            async for chunk in response.aiter_text():
                if first_token_time is None:
                    first_token_time = time.time()
                buffer += chunk
                
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
    
    done = time.time()
    
    print(f"Time to first token: {(first_token_time - start) * 1000:.0f}ms")
    print(f"Total time: {(done - start) * 1000:.0f}ms")
    print(f"(User starts seeing response at {(first_token_time - start) * 1000:.0f}ms!)")


# =============================================================================
# Main
# =============================================================================

async def main():
    print("AI Gateway - Streaming Examples")
    print("=" * 50)
    print("Make sure the server is running:")
    print("  uvicorn ai_gateway.main:app --reload")
    print("=" * 50)
    
    try:
        stream_sync()
        await stream_async()
        await stream_with_processing()
        await compare_streaming_modes()
    except httpx.ConnectError:
        print("\nError: Cannot connect to the server.")
        print("Start it with: uvicorn ai_gateway.main:app --reload")


if __name__ == "__main__":
    asyncio.run(main())
