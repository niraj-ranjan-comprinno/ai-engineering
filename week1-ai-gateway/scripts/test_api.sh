#!/bin/bash
#
# Quick API tests using curl.
#
# =============================================================================
# THEORY: Testing LLM APIs
# =============================================================================
#
# When testing LLM APIs, verify:
# 1. Basic connectivity (health check)
# 2. Input validation (bad requests)
# 3. Happy path (valid request)
# 4. Streaming (SSE format)
# 5. Error handling (provider errors)
#
# curl flags used:
# -X POST: HTTP method
# -H: Headers
# -d: Request body
# -s: Silent mode
# -N: Disable buffering (important for streaming!)
#
# =============================================================================

BASE_URL="${AI_GATEWAY_URL:-http://localhost:8000}"

echo "Testing AI Gateway at: $BASE_URL"
echo "========================================"

# Test 1: Health Check
echo -e "\n1. Health Check"
echo "---------------"
curl -s "$BASE_URL/v1/health" | python3 -m json.tool
echo ""

# Test 2: List Providers
echo -e "\n2. List Providers"
echo "-----------------"
curl -s "$BASE_URL/v1/providers" | python3 -m json.tool
echo ""

# Test 3: Token Counting
echo -e "\n3. Token Counting"
echo "-----------------"
curl -s -X POST "$BASE_URL/v1/tokens/count" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }' | python3 -m json.tool
echo ""

# Test 4: Non-Streaming Chat
echo -e "\n4. Non-Streaming Chat Completion"
echo "---------------------------------"
echo "Request:"
echo '{"messages": [{"role": "user", "content": "Say hello in 5 words or less."}], "stream": false}'
echo ""
echo "Response:"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in 5 words or less."}],
    "stream": false
  }' | python3 -m json.tool
echo ""

# Test 5: Streaming Chat (raw output)
echo -e "\n5. Streaming Chat Completion (SSE)"
echo "-----------------------------------"
echo "Request:"
echo '{"messages": [{"role": "user", "content": "Count from 1 to 5."}], "stream": true}'
echo ""
echo "Response (SSE events):"
# -N disables buffering which is critical for SSE!
curl -s -N -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": true
  }'
echo -e "\n"

# Test 6: Metrics
echo -e "\n6. Metrics"
echo "----------"
curl -s "$BASE_URL/metrics" | python3 -m json.tool
echo ""

echo "========================================"
echo "Tests complete!"
