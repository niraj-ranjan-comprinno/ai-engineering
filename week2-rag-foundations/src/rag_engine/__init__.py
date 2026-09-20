"""
RAG Engine - Week 2: RAG Foundations

A Retrieval-Augmented Generation system that:
1. Ingests documents (PDF, Markdown, Text)
2. Chunks them into smaller pieces
3. Creates embeddings using AWS Bedrock Titan
4. Stores in ChromaDB vector database
5. Retrieves relevant context for queries
6. Generates answers using Week 1's AI Gateway
"""

__version__ = "0.1.0"
