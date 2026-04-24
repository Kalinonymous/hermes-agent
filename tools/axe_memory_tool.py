#!/usr/bin/env python3
"""
AXE Central Memory Tool - Hermes Memory Integration

Stores and retrieves memories from Cloudflare Vectorize via the memory worker.
This gives Hermes persistent memory across conversations.

Flow:
  1. Store: Hermes calls axe_memory_store with text → memory worker → Vectorize
  2. Retrieve: Hermes calls axe_memory_retrieve with query → Vectorize search
  3. Context: Hermes loads memory context before responding to any user

Environment (from ~/.hermes/.env):
  MEMORY_WORKER_URL=https://memory.westgategroupofschools.ke
  MEMORY_API_KEY=<api-key>
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

MEMORY_WORKER_URL = os.environ.get("MEMORY_WORKER_URL", "https://memory.westgategroupofschools.ke")
MEMORY_API_KEY = os.environ.get("MEMORY_API_KEY", "")


def _make_request(method: str, endpoint: str, data: dict = None) -> dict:
    """Make HTTP request to memory worker. Uses urllib (built-in, no new packages)."""
    url = f"{MEMORY_WORKER_URL}{endpoint}"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MEMORY_API_KEY}",
        "User-Agent": "AXE-Hermes/1.0",
    }
    
    body = json.dumps(data).encode("utf-8") if data else None
    
    try:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return {"ok": True, "data": json.loads(resp.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        return {"ok": False, "error": f"HTTP {e.code}: {body}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def axe_memory_store(
    text: str,
    tenant_id: str,
    category: str = "general",
    access_level: str = "admin",
    urgency_score: int = 5,
    description: str = None,
) -> dict:
    """
    Store a memory in the central Vectorize system.

    Args:
        text: The content to remember (will be embedded and stored)
        tenant_id: Which tenant this memory belongs to (required for isolation)
        category: finance|students|staff|academic|operations|compliance|marketing|general
        access_level: admin|staff|teacher|public
        urgency_score: 1-10 (10=critical business rules that must not be ignored)
        description: Short summary of what this memory is about

    Returns:
        {"ok": True, "memory_id": "mem-xxx", "stored": True}
    """
    if not tenant_id:
        return {"ok": False, "error": "tenant_id is required"}
    
    if not text:
        return {"ok": False, "error": "text is required"}
    
    payload = {
        "text": text,
        "metadata": {
            "tenant_id": tenant_id,
            "category": category,
            "access_level": access_level,
            "urgency_score": urgency_score,
            "description": description or text[:100],
        }
    }
    
    result = _make_request("POST", "/memory", data=payload)
    
    if result.get("ok"):
        return {
            "ok": True,
            "memory_id": result.get("data", {}).get("id", "unknown"),
            "stored": True,
            "category": category,
        }
    else:
        return {"ok": False, "error": result.get("error", "Unknown error")}


def axe_memory_retrieve(
    query: str,
    tenant_id: str,
    category: str = None,
    limit: int = 5,
) -> dict:
    """
    Search and retrieve relevant memories from Vectorize.

    Args:
        query: Search query to find relevant memories
        tenant_id: Which tenant's memories to search (mandatory filter)
        category: Optional filter by category
        limit: Max memories to return (default 5)

    Returns:
        {"ok": True, "memories": [{"text": "...", "score": 0.9, "metadata": {...}}, ...]}
    """
    if not tenant_id:
        return {"ok": False, "error": "tenant_id is required"}
    
    if not query:
        return {"ok": False, "error": "query is required"}
    
    # Build filter - tenant_id is ALWAYS applied (hard-coded safety)
    filter_obj = {"tenant_id": tenant_id}
    if category:
        filter_obj["category"] = category
    
    payload = {
        "text": query,
        "top_k": limit,
        "filter": filter_obj,
    }
    
    result = _make_request("POST", "/query", data=payload)
    
    if result.get("ok"):
        matches = result.get("data", {}).get("matches", [])
        return {
            "ok": True,
            "memories": [
                {
                    "text": m.get("metadata", {}).get("text", "") or m.get("text", ""),
                    "score": m.get("score", 0),
                    "metadata": m.get("metadata", {}),
                }
                for m in matches
            ],
            "count": len(matches),
        }
    else:
        return {"ok": False, "error": result.get("error", "Unknown error")}


def axe_memory_context(
    tenant_id: str,
    query: str = None,
    limit: int = 10,
) -> str:
    """
    Load memory context for an agent. Returns formatted string of relevant memories.

    This is the PRIMARY function Hermes calls before responding to any user.
    It loads all relevant memories and formats them as context.

    Args:
        tenant_id: Tenant to load context for
        query: Optional specific query, or loads all recent memories if None
        limit: Max memories to load

    Returns:
        Formatted string like "[Memory: Student fee balance - KES 45,000 - finance]"
    """
    if not query:
        query = "recent important memories business rules"
    
    result = axe_memory_retrieve(query=query, tenant_id=tenant_id, limit=limit)
    
    if not result.get("ok"):
        return f"[Memory system unavailable: {result.get('error')}]"
    
    memories = result.get("memories", [])
    
    if not memories:
        return "[No relevant memories found]"
    
    parts = []
    for mem in memories:
        meta = mem.get("metadata", {})
        desc = meta.get("description", mem.get("text", "")[:80])
        cat = meta.get("category", "general")
        score = mem.get("score", 0)
        parts.append(f"[{cat.upper()} score:{score:.2f}] {desc}")
    
    return "\n".join(parts)


# --- Tool Schema ---

AXE_MEMORY_STORE_SCHEMA = {
    "name": "axe_memory_store",
    "description": "Store information in AXE central memory so the agent remembers it across conversations. Use when: user tells you something important about their business, you learn a rule/preference/fact, you want to save key information.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The exact information to remember. Be specific and complete."},
            "tenant_id": {"type": "string", "description": "Tenant identifier (e.g. 'westgate_001'). Required for isolation."},
            "category": {"type": "string", "enum": ["finance", "students", "staff", "academic", "operations", "compliance", "marketing", "general"], "description": "What type of information.", "default": "general"},
            "access_level": {"type": "string", "enum": ["admin", "staff", "teacher", "public"], "description": "Who can see this.", "default": "admin"},
            "urgency_score": {"type": "integer", "minimum": 1, "maximum": 10, "description": "How critical? 10=critical business rules.", "default": 5},
            "description": {"type": "string", "description": "Short summary (under 100 chars)."},
        },
        "required": ["text", "tenant_id"],
    },
}


AXE_MEMORY_RETRIEVE_SCHEMA = {
    "name": "axe_memory_retrieve",
    "description": "Search AXE central memory for relevant information. tenant_id filter is ALWAYS applied automatically.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query to find relevant memories."},
            "tenant_id": {"type": "string", "description": "Tenant identifier. Required."},
            "category": {"type": "string", "enum": ["finance", "students", "staff", "academic", "operations", "compliance", "marketing", "general"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max memories to return.", "default": 5},
        },
        "required": ["query", "tenant_id"],
    },
}


AXE_MEMORY_CONTEXT_SCHEMA = {
    "name": "axe_memory_context",
    "description": "Load memory context before responding to user. Call at START of every conversation turn. Returns formatted context string.",
    "parameters": {
        "type": "object",
        "properties": {
            "tenant_id": {"type": "string", "description": "Tenant identifier. Required."},
            "query": {"type": "string", "description": "Optional specific query."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Max memories to load.", "default": 10},
        },
        "required": ["tenant_id"],
    },
}


def check_memory_requirements() -> bool:
    """Check if memory worker is reachable."""
    try:
        result = _make_request("GET", "/")
        return result.get("ok", False)
    except Exception:
        return False


# --- Registry ---

from tools.registry import registry

registry.register(
    name="axe_memory_store",
    toolset="axe_memory",
    schema=AXE_MEMORY_STORE_SCHEMA,
    handler=lambda args, **kw: axe_memory_store(
        text=args.get("text", ""),
        tenant_id=args.get("tenant_id", ""),
        category=args.get("category", "general"),
        access_level=args.get("access_level", "admin"),
        urgency_score=args.get("urgency_score", 5),
        description=args.get("description"),
    ),
    check_fn=check_memory_requirements,
    emoji="💾",
)

registry.register(
    name="axe_memory_retrieve",
    toolset="axe_memory",
    schema=AXE_MEMORY_RETRIEVE_SCHEMA,
    handler=lambda args, **kw: axe_memory_retrieve(
        query=args.get("query", ""),
        tenant_id=args.get("tenant_id", ""),
        category=args.get("category"),
        limit=args.get("limit", 5),
    ),
    check_fn=check_memory_requirements,
    emoji="🔍",
)

registry.register(
    name="axe_memory_context",
    toolset="axe_memory",
    schema=AXE_MEMORY_CONTEXT_SCHEMA,
    handler=lambda args, **kw: axe_memory_context(
        tenant_id=args.get("tenant_id", ""),
        query=args.get("query"),
        limit=args.get("limit", 10),
    ),
    check_fn=check_memory_requirements,
    emoji="🧠",
)
