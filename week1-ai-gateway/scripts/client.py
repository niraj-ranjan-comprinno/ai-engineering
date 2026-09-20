#!/usr/bin/env python3
"""
Interactive CLI Client for the AI Gateway.

=============================================================================
THEORY: Building LLM Client Applications
=============================================================================

A good LLM client needs to handle:
1. Streaming responses - Display text as it arrives
2. Connection management - Handle disconnects gracefully
3. Error handling - Retry on transient failures
4. User experience - Show progress, costs, latency

This client demonstrates:
- SSE consumption with httpx-sse
- Rich terminal UI with the Rich library
- Async patterns for non-blocking I/O
- Proper resource cleanup

Libraries used:
- httpx: Modern async HTTP client (better than requests)
- httpx-sse: Server-Sent Events support for httpx
- rich: Beautiful terminal formatting
- asyncio: Python's async runtime

Run with: python scripts/client.py
"""

import asyncio
import json
import sys
from typing import Optional

import httpx
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

# Base URL for the AI Gateway
BASE_URL = "http://localhost:8000"

console = Console()


# =============================================================================
# SSE Client
# =============================================================================

async def stream_chat(
    messages: list[dict],
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> tuple[str, dict]:
    """
    Stream a chat completion from the gateway.
    
    THEORY: Consuming Server-Sent Events
    -------------------------------------
    SSE is consumed differently than regular HTTP:
    
    1. Open connection (doesn't close immediately)
    2. Read events as they arrive
    3. Parse each event's type and data
    4. Handle connection termination
    
    Event format:
        event: chunk
        data: {"delta": "Hello"}
        
        event: complete
        data: {"usage": {...}, "cost": {...}}
        
        event: done
        data: [DONE]
    
    httpx-sse handles the low-level parsing for us.
    
    Returns:
        Tuple of (full_content, metadata_dict)
    """
    full_content = ""
    metadata = {}
    
    request_body = {
        "messages": messages,
        "stream": True,
    }
    if model:
        request_body["model"] = model
    if provider:
        request_body["provider"] = provider
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json=request_body,
            headers={"Accept": "text/event-stream"},
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise Exception(f"Request failed: {response.status_code} - {error_text}")
            
            # Read SSE events
            buffer = ""
            current_event = None
            
            async for chunk in response.aiter_text():
                buffer += chunk
                
                # Process complete events (separated by double newline)
                while "\n\n" in buffer:
                    event_text, buffer = buffer.split("\n\n", 1)
                    
                    # Parse event
                    event_type = "message"  # default
                    event_data = ""
                    
                    for line in event_text.split("\n"):
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            event_data = line[5:].strip()
                    
                    if not event_data:
                        continue
                    
                    # Handle different event types
                    if event_type == "chunk":
                        try:
                            data = json.loads(event_data)
                            delta = data.get("delta", "")
                            full_content += delta
                            # Print delta immediately for streaming effect
                            console.print(delta, end="")
                        except json.JSONDecodeError:
                            pass
                    
                    elif event_type == "complete":
                        try:
                            metadata = json.loads(event_data)
                        except json.JSONDecodeError:
                            pass
                    
                    elif event_type == "done":
                        break
                    
                    elif event_type == "error":
                        try:
                            error = json.loads(event_data)
                            raise Exception(f"Stream error: {error.get('message', 'Unknown error')}")
                        except json.JSONDecodeError:
                            raise Exception(f"Stream error: {event_data}")
    
    return full_content, metadata


# =============================================================================
# Non-Streaming Request
# =============================================================================

async def chat(
    messages: list[dict],
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict:
    """
    Make a non-streaming chat completion request.
    
    THEORY: When to Use Non-Streaming
    ---------------------------------
    Non-streaming is simpler and better for:
    - Batch processing (no user watching)
    - When you need the complete response to proceed
    - Testing and debugging
    - Lower overhead for very short responses
    
    Returns:
        Full response dictionary
    """
    request_body = {
        "messages": messages,
        "stream": False,
    }
    if model:
        request_body["model"] = model
    if provider:
        request_body["provider"] = provider
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{BASE_URL}/v1/chat/completions",
            json=request_body,
        )
        response.raise_for_status()
        return response.json()


# =============================================================================
# Utility Functions
# =============================================================================

async def get_providers() -> dict:
    """Fetch available providers from the gateway."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/v1/providers")
        response.raise_for_status()
        return response.json()


async def count_tokens(messages: list[dict], model: str = None) -> dict:
    """Count tokens for messages without making an LLM call."""
    request_body = {"messages": messages}
    if model:
        request_body["model"] = model
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/v1/tokens/count",
            json=request_body,
        )
        response.raise_for_status()
        return response.json()


def display_usage(metadata: dict):
    """
    Display usage statistics in a nice table.
    
    THEORY: UX for LLM Applications
    -------------------------------
    Users should see:
    1. What they got (content length, quality indicators)
    2. What it cost (tokens, dollars)
    3. How long it took (latency)
    
    This transparency builds trust and helps users
    optimize their usage.
    """
    usage = metadata.get("usage", {})
    cost = metadata.get("cost", {})
    
    table = Table(title="Request Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", metadata.get("model", "unknown"))
    table.add_row("Provider", metadata.get("provider", "unknown"))
    table.add_row("Latency", f"{metadata.get('latency_ms', 0):.0f}ms")
    table.add_row("", "")
    table.add_row("Prompt Tokens", str(usage.get("prompt_tokens", 0)))
    table.add_row("Completion Tokens", str(usage.get("completion_tokens", 0)))
    table.add_row("Total Tokens", str(usage.get("total_tokens", 0)))
    
    if cost:
        table.add_row("", "")
        table.add_row("Input Cost", f"${cost.get('input_cost', 0):.6f}")
        table.add_row("Output Cost", f"${cost.get('output_cost', 0):.6f}")
        table.add_row("Total Cost", f"${cost.get('total_cost', 0):.6f}")
    
    console.print()
    console.print(table)


# =============================================================================
# Interactive Chat Loop
# =============================================================================

async def interactive_chat():
    """
    Run an interactive chat session.
    
    THEORY: Conversation State Management
    -------------------------------------
    Chat applications maintain conversation history:
    
    Turn 1: User → "Hi"
    Turn 2: Assistant → "Hello!"
    Turn 3: User → "What's 2+2?"
    Turn 4: Assistant → "4"
    
    Each request includes ALL previous messages so the model
    has context. This is why token costs grow with conversation length.
    
    Strategies to manage this:
    1. Truncate old messages (sliding window)
    2. Summarize history periodically
    3. Reset conversation when topic changes
    """
    console.print(Panel.fit(
        "[bold green]AI Gateway Interactive Client[/bold green]\n\n"
        "Commands:\n"
        "  /quit, /exit - Exit the client\n"
        "  /clear - Clear conversation history\n"
        "  /model <name> - Switch model\n"
        "  /provider <name> - Switch provider\n"
        "  /providers - List available providers\n"
        "  /tokens - Count tokens in current conversation\n"
        "  /stream on|off - Toggle streaming mode\n"
        "  /help - Show this help\n",
        title="Welcome",
    ))
    
    # Check if server is running
    try:
        providers = await get_providers()
        console.print(f"[green]Connected to AI Gateway[/green]")
        console.print(f"Default provider: {providers['default_provider']}")
    except Exception as e:
        console.print(f"[red]Error connecting to AI Gateway: {e}[/red]")
        console.print("[yellow]Make sure the server is running: uvicorn ai_gateway.main:app --reload[/yellow]")
        return
    
    # Conversation state
    messages = []
    current_model = None
    current_provider = None
    streaming = True
    
    # System prompt
    system_prompt = Prompt.ask(
        "\n[cyan]System prompt (press Enter for default)[/cyan]",
        default="You are a helpful AI assistant.",
    )
    messages.append({"role": "system", "content": system_prompt})
    
    console.print("\n[dim]Chat started. Type your message and press Enter.[/dim]\n")
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()
                
                if cmd[0] in ["/quit", "/exit"]:
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                
                elif cmd[0] == "/clear":
                    messages = [messages[0]]  # Keep system prompt
                    console.print("[yellow]Conversation cleared.[/yellow]")
                    continue
                
                elif cmd[0] == "/model" and len(cmd) > 1:
                    current_model = cmd[1]
                    console.print(f"[yellow]Model set to: {current_model}[/yellow]")
                    continue
                
                elif cmd[0] == "/provider" and len(cmd) > 1:
                    current_provider = cmd[1]
                    console.print(f"[yellow]Provider set to: {current_provider}[/yellow]")
                    continue
                
                elif cmd[0] == "/providers":
                    providers = await get_providers()
                    for name, info in providers["providers"].items():
                        console.print(f"\n[cyan]{name}[/cyan]")
                        console.print(f"  Default: {info['default_model']}")
                        console.print(f"  Models: {', '.join(info['models'][:5])}...")
                    continue
                
                elif cmd[0] == "/tokens":
                    token_info = await count_tokens(messages, current_model)
                    console.print(f"[yellow]Conversation tokens: {token_info['prompt_tokens']}[/yellow]")
                    continue
                
                elif cmd[0] == "/stream":
                    if len(cmd) > 1 and cmd[1] in ["on", "off"]:
                        streaming = cmd[1] == "on"
                        console.print(f"[yellow]Streaming: {'on' if streaming else 'off'}[/yellow]")
                    else:
                        console.print(f"[yellow]Streaming is currently {'on' if streaming else 'off'}[/yellow]")
                    continue
                
                elif cmd[0] == "/help":
                    console.print(Panel.fit(
                        "Commands:\n"
                        "  /quit, /exit - Exit\n"
                        "  /clear - Clear history\n"
                        "  /model <name> - Switch model\n"
                        "  /provider <name> - Switch provider\n"
                        "  /providers - List providers\n"
                        "  /tokens - Count tokens\n"
                        "  /stream on|off - Toggle streaming",
                    ))
                    continue
                
                else:
                    console.print("[red]Unknown command. Type /help for help.[/red]")
                    continue
            
            # Add user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Get response
            console.print("\n[bold green]Assistant[/bold green]: ", end="")
            
            if streaming:
                content, metadata = await stream_chat(
                    messages,
                    model=current_model,
                    provider=current_provider,
                )
            else:
                response = await chat(
                    messages,
                    model=current_model,
                    provider=current_provider,
                )
                content = response["content"]
                metadata = response
                console.print(content)
            
            # Add assistant response to history
            messages.append({"role": "assistant", "content": content})
            
            # Display usage stats
            display_usage(metadata)
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type /quit to exit.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            # Remove failed user message from history
            if messages and messages[-1]["role"] == "user":
                messages.pop()


# =============================================================================
# Demo Mode
# =============================================================================

async def run_demo():
    """
    Run a quick demo of the gateway features.
    
    Useful for:
    - Verifying the setup works
    - Demonstrating features
    - Quick testing
    """
    console.print(Panel.fit(
        "[bold]AI Gateway Demo[/bold]\n"
        "This will make a few requests to demonstrate the gateway.",
        title="Demo Mode",
    ))
    
    # Check connection
    console.print("\n[cyan]1. Checking server connection...[/cyan]")
    try:
        providers = await get_providers()
        console.print(f"   ✓ Server is running")
        console.print(f"   ✓ Available providers: {list(providers['providers'].keys())}")
    except Exception as e:
        console.print(f"   [red]✗ Cannot connect: {e}[/red]")
        console.print("   [yellow]Start the server with: uvicorn ai_gateway.main:app --reload[/yellow]")
        return
    
    # Token counting
    console.print("\n[cyan]2. Token counting...[/cyan]")
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    token_info = await count_tokens(test_messages)
    console.print(f"   ✓ Test prompt: {token_info['prompt_tokens']} tokens")
    
    # Streaming request
    console.print("\n[cyan]3. Streaming chat completion...[/cyan]")
    console.print("   Response: ", end="")
    content, metadata = await stream_chat(test_messages)
    display_usage(metadata)
    
    console.print("\n[green]Demo complete![/green]")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Gateway CLI Client")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a quick demo instead of interactive mode",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the AI Gateway (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    if args.demo:
        asyncio.run(run_demo())
    else:
        asyncio.run(interactive_chat())


if __name__ == "__main__":
    main()
