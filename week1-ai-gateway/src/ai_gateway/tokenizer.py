"""
Token Counting and Cost Calculation.

=============================================================================
THEORY: Tokenization Deep Dive
=============================================================================

What is tokenization?
---------------------
Tokenization converts text into numerical tokens that models understand.
Modern LLMs use subword tokenization (BPE - Byte Pair Encoding).

How BPE works:
1. Start with individual characters
2. Find most frequent adjacent pairs
3. Merge them into a new token
4. Repeat until vocabulary size is reached

Example: "lowest" might become ["low", "est"]
         "newer" might become ["new", "er"]

Why BPE?
- Handles rare/unknown words by breaking them down
- Balances vocabulary size vs sequence length
- Works across languages

tiktoken:
---------
OpenAI's tokenizer library. Fast (Rust-based) and accurate.
Each model uses a specific encoding:
- gpt-4, gpt-4o: cl100k_base
- gpt-3.5-turbo: cl100k_base  
- text-davinci-003: p50k_base

Anthropic uses a different tokenizer, but we estimate with cl100k_base.
"""

from functools import lru_cache
import tiktoken


# =============================================================================
# Model Pricing (USD per 1M tokens) - Updated September 2024
# =============================================================================
# Keep this updated! Prices change frequently.
# Format: model_name: (input_price_per_1M, output_price_per_1M)

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI Models
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4-turbo-preview": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    "gpt-3.5-turbo-0125": (0.50, 1.50),
    
    # Anthropic Models
    "claude-3-5-sonnet-20240620": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-opus-latest": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}

# Default pricing for unknown models
DEFAULT_PRICING = (1.00, 3.00)


@lru_cache(maxsize=10)
def get_encoding(model: str) -> tiktoken.Encoding:
    """
    Get the tokenizer encoding for a model.
    
    THEORY: Encoding Selection
    --------------------------
    Different model families use different tokenizers:
    
    - cl100k_base: GPT-4, GPT-3.5-turbo, text-embedding-ada-002
    - p50k_base: text-davinci-003, code-davinci-002
    - r50k_base: GPT-3 davinci, curie, etc.
    
    Using the wrong tokenizer gives inaccurate counts!
    We cache encodings because loading them takes ~100ms.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base for unknown models
        # This is a good approximation for most modern models
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to tokenize
        model: The model to use for tokenization
        
    Returns:
        Number of tokens
        
    Example:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("def fibonacci(n):\\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)")
        28
    """
    encoding = get_encoding(model)
    return len(encoding.encode(text))


def count_message_tokens(messages: list[dict], model: str = "gpt-4o-mini") -> int:
    """
    Count tokens for a list of chat messages.
    
    THEORY: Chat Message Token Overhead
    ------------------------------------
    Chat completions have additional token overhead:
    
    Each message adds special tokens for:
    - Role marker (e.g., <|im_start|>assistant)
    - Content separator
    - Message end marker
    
    OpenAI's formula (approximately):
    tokens = sum(tokens_per_message for each message)
           + tokens_per_name if name present
           + 3 (for priming assistant response)
    
    The exact overhead varies by model, but ~4 tokens per message
    is a reasonable approximation for GPT-4 class models.
    """
    encoding = get_encoding(model)
    
    # Token overhead per message (model-specific, using GPT-4 values)
    tokens_per_message = 4  # <|im_start|>{role}\n{content}<|im_end|>
    tokens_per_name = -1    # Omitting name saves 1 token
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    
    # Every reply is primed with <|im_start|>assistant
    num_tokens += 3
    
    return num_tokens


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> dict[str, float]:
    """
    Calculate the cost for a completion request.
    
    Args:
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        model: Model name for pricing lookup
        
    Returns:
        Dictionary with input_cost, output_cost, total_cost
        
    THEORY: Cost Calculation
    ------------------------
    LLM pricing is per-token, usually quoted per 1M tokens.
    
    Formula:
        input_cost = (prompt_tokens / 1_000_000) * input_price_per_M
        output_cost = (completion_tokens / 1_000_000) * output_price_per_M
        total_cost = input_cost + output_cost
    
    Important insights:
    1. Output tokens cost 2-4x more than input tokens
    2. A verbose system prompt compounds over many requests
    3. max_tokens limits output cost but not input cost
    """
    input_price, output_price = MODEL_PRICING.get(model, DEFAULT_PRICING)
    
    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price
    
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
    }


def estimate_max_context(model: str) -> int:
    """
    Get the maximum context window for a model.
    
    THEORY: Context Windows
    -----------------------
    The context window is the maximum total tokens (input + output)
    a model can handle in a single request.
    
    Context window evolution:
    - GPT-3: 4,096 tokens
    - GPT-3.5-turbo: 4,096 → 16,384 tokens
    - GPT-4: 8,192 → 128,000 tokens
    - Claude 3: 200,000 tokens
    
    Larger context windows enable:
    - Longer conversations
    - Processing entire documents
    - More few-shot examples
    
    But beware:
    - Cost increases linearly with tokens
    - Very long contexts can reduce quality
    - Attention mechanism is O(n²) in sequence length
    """
    context_windows = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "claude-3-5-sonnet-20240620": 200000,
        "claude-3-opus-20240229": 200000,
        "claude-3-sonnet-20240229": 200000,
        "claude-3-haiku-20240307": 200000,
    }
    return context_windows.get(model, 8192)
