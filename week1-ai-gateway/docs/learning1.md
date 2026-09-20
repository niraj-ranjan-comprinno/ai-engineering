# Learning 1: AWS Bedrock Integration Challenges

## Date: September 20, 2026

This document captures the real-world problems encountered while integrating AWS Bedrock as an LLM provider and the solutions applied.

---

## Problem 1: Bedrock API Keys Don't Work for Model Invocation

### What Happened
Initially thought Bedrock's "API Keys" feature (visible in AWS Console → Bedrock → API keys) would work like OpenAI's API keys - just paste and use.

### The Error
```
403 Forbidden for url 'https://bedrock-runtime.us-east-1.amazonaws.com/model/.../invoke'
```

### Root Cause
Bedrock API keys are designed for specific features like **Bedrock Agents**, NOT for direct model invocation. Model invocation requires **AWS Signature V4 authentication**.

### Solution
Use AWS credentials via `~/.aws/credentials` with boto3:

```bash
# Set up AWS credentials securely
aws configure --profile ai-learning

# Then use in code
session = boto3.Session(profile_name='ai-learning')
client = session.client('bedrock-runtime', region_name='us-east-1')
```

### Key Takeaway
> Not all "API keys" work the same way. Always check the authentication method required by the specific API endpoint.

---

## Problem 2: Model Version End of Life

### What Happened
Used model IDs from documentation/examples that were outdated.

### The Error
```
ResourceNotFoundException: This model version has reached the end of its life. 
Please refer to the AWS documentation for more details.
```

### Root Cause
AWS retires older model versions. Model IDs like `anthropic.claude-3-5-sonnet-20240620-v1:0` become unavailable over time.

### Solution
Always query for current available models:

```bash
# List available Claude models
aws bedrock list-foundation-models \
  --query "modelSummaries[?contains(modelId, 'claude')].{modelId:modelId, modelName:modelName}" \
  --output table
```

### Key Takeaway
> LLM models have lifecycles. Build your system to handle model deprecation gracefully - use aliases that can be remapped.

---

## Problem 3: On-Demand Invocation Not Supported

### What Happened
Found the correct, current model ID but still couldn't invoke it.

### The Error
```
ValidationException: Invocation of model ID anthropic.claude-sonnet-4-6 
with on-demand throughput isn't supported. 
Retry your request with the ID or ARN of an inference profile that contains this model.
```

### Root Cause
AWS requires **Inference Profiles** for many newer models. An inference profile is a managed configuration that handles routing, throughput, and regional availability.

### Solution
1. Find available inference profiles:
```bash
aws bedrock list-inference-profiles \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId, 'claude')].{id:inferenceProfileId, name:inferenceProfileName}" \
  --output table
```

2. Use the inference profile ID instead of the model ID:
```python
# ❌ Wrong - Model ID
model_id = "anthropic.claude-sonnet-4-6"

# ✅ Correct - Inference Profile ID
model_id = "us.anthropic.claude-sonnet-4-6"
```

### Key Takeaway
> AWS Bedrock has an extra layer of abstraction (Inference Profiles) between you and the models. This enables cross-region routing and managed throughput.

---

## Problem 4: Model Type Detection Failed

### What Happened
Code that detected model type by checking the prefix stopped working.

### The Error
```
ValueError: Unsupported model: claude-3-haiku
```

### Root Cause
Old code checked `model_id.startswith("anthropic.")` but inference profiles start with `us.` or `global.`:

```python
# Model ID format:      anthropic.claude-sonnet-4-6
# Inference Profile:    us.anthropic.claude-sonnet-4-6
#                       ^^^--- Different prefix!
```

### Solution
Use substring matching instead of prefix matching:

```python
# ❌ Old code - breaks with inference profiles
def _is_claude_model(self, model: str) -> bool:
    model_id = self._get_model_id(model)
    return model_id.startswith("anthropic.")

# ✅ Fixed code - works with any format
def _is_claude_model(self, model: str) -> bool:
    model_id = self._get_model_id(model)
    return "anthropic" in model_id or "claude" in model.lower()
```

### Key Takeaway
> When dealing with identifiers that might have prefixes or namespaces, use `in` (substring matching) rather than `startswith()` for more robust detection.

---

## Problem 5: Finish Reason Validation Error

### What Happened
Streaming worked but failed at the end when parsing the finish reason.

### The Error
```
ValidationError: 1 validation error for StreamChunk
finish_reason
  Input should be 'stop', 'length' or 'error' [type=literal_error, input_value='end_turn']
```

### Root Cause
Different providers use different terminology:
- **OpenAI**: `"stop"`, `"length"`
- **Anthropic/Bedrock**: `"end_turn"`, `"max_tokens"`

Our Pydantic model was too strict:
```python
finish_reason: Optional[Literal["stop", "length", "error"]]
```

### Solution
Accept any string and normalize if needed:

```python
# ✅ Flexible - accepts any provider's format
finish_reason: Optional[str]
```

Or normalize the values:
```python
def normalize_finish_reason(reason: str) -> str:
    mapping = {
        "end_turn": "stop",
        "max_tokens": "length",
    }
    return mapping.get(reason, reason)
```

### Key Takeaway
> When building multi-provider systems, design for the **union** of all possible values, not the intersection. Normalize at the edges of your system.

---

## Summary: Key Learnings for AWS Bedrock

| Aspect | What to Remember |
|--------|------------------|
| **Authentication** | Use AWS credentials (boto3), not API keys for model invocation |
| **Model IDs** | Check current availability; models retire over time |
| **Inference Profiles** | Newer models require inference profile IDs, not raw model IDs |
| **Regional Prefixes** | Profile IDs have prefixes like `us.` or `global.` |
| **Provider Differences** | Different providers return different field values; normalize them |

## Useful Commands

```bash
# List all Claude models
aws bedrock list-foundation-models \
  --query "modelSummaries[?contains(modelId, 'claude')]" \
  --output table

# List inference profiles for Claude
aws bedrock list-inference-profiles \
  --query "inferenceProfileSummaries[?contains(inferenceProfileId, 'claude')]" \
  --output table

# Test a model invocation
aws bedrock-runtime invoke-model \
  --model-id "us.anthropic.claude-sonnet-4-6" \
  --body '{"anthropic_version":"bedrock-2023-05-31","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}' \
  --content-type "application/json" \
  output.json
```

---

*Documented during Week 1: AI Gateway project*
