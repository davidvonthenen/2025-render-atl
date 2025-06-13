# coordinator_agent.py
"""
Coordinator agent that calls a local MCP Ticketmaster server to fetch
Los-Angeles (CA) concert listings for a given artist / band and (optionally)
suggests related merch.

Key updates
-----------
1. The Ticketmaster helper now **passes only the search term** (keyword)
   to the MCP server—no markup, state, or format parameters.
2. State selection is hard-coded inside the MCP server itself.
"""

from __future__ import annotations

import os
import uuid
import json
import requests
import openai
from flask import Flask, request, jsonify
from common.types import (  # type: ignore
    A2ARequest,
    SendTaskRequest,
    SendTaskResponse,
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart,
    TaskSendParams,
)

openai.api_key = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TICKETMASTER_URL = os.getenv("TICKETMASTER_URL", "http://127.0.0.1:6274/events")
SHOPPING_URL = os.getenv("SHOPPING_URL", "http://127.0.0.1:7274/recommendation")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_ticketmaster_events",
            "description": "Look up concerts in Los Angeles for a given artist / band.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_recommendations",
            "description": "Suggest products related to the given keyword.",
            "parameters": {
                "type": "object",
                "properties": {"keyword": {"type": "string"}},
                "required": ["keyword"],
            },
        },
    },
]

app = Flask("CoordinatorAgent")


# --------------------------- HTTP entry ---------------------------
@app.post("/tasks")
def receive_task():
    """JSON-RPC tasks/send endpoint."""
    req_json = request.get_json(force=True)
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        return _rpc_error(req_json.get("id"), f"Invalid JSON-RPC: {exc}")

    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_error(rpc_req.id, "Only tasks/send supported")

    params = rpc_req.params
    parts = params.message.parts or []
    if len(parts) < 2:
        return _task_fail(rpc_req.id, params.id, "Need two parts: [context, question]")

    context, question = parts[0].text, parts[1].text

    messages = [
        {
            "role": "system",
            "content": (
                "You orchestrate tasks by calling the provided functions. "
                "If the context mentions a musician or band, call "
                "`search_ticketmaster_events(keyword=…)`. Ask if they would like to purchase a ticket at one of these concerts. The wording of this result should be suitable for doing Text-to-Speech."
                "If you are looking for general purchasing recommendations, call "
                "`search_recommendations(keyword=…)`. Ask if they would like to purchase one of these recommended products. The wording of this result should be suitable for doing Text-to-Speech."
                "\n\n"
                "Based on the results, you are providing recommendations for concerts to attend or things to purchase. Provide a concise answer and appropriate response."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
    ]

    while True:
        try:
            resp = openai.chat.completions.create(
                model=OPENAI_MODEL, messages=messages, tools=TOOLS, temperature=0
            )
        except Exception as exc:
            return _task_fail(rpc_req.id, params.id, f"OpenAI error: {exc}")

        choice = resp.choices[0]

        # ------------------ Tool invocation -----------------------
        if choice.finish_reason == "tool_calls":
            for call in choice.message.tool_calls:
                args = json.loads(call.function.arguments)

                if call.function.name == "search_ticketmaster_events":
                    # ONLY the keyword gets forwarded
                    tool_out = _delegate_worker(TICKETMASTER_URL, [args["keyword"]])
                elif call.function.name == "search_recommendations":
                    tool_out = _delegate_worker(SHOPPING_URL, [args["keyword"]])
                else:
                    tool_out = f"No handler for function {call.function.name}"

                messages.extend(
                    [
                        choice.message,
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": tool_out,
                        },
                    ]
                )
            # Let the model think again with new info
            continue

        # ------------------ Final answer -------------------------
        final_text = choice.message.content.strip()
        result_task = Task(
            id=params.id,
            status=TaskStatus(
                state=TaskState.COMPLETED,
                message=Message(role="agent", parts=[TextPart(text=final_text)]),
            ),
        )
        return jsonify(SendTaskResponse(id=rpc_req.id, result=result_task).model_dump())


# ---------------------- helper functions --------------------------
def _delegate_worker(url: str, texts: list[str]) -> str:
    """
    Send a sub-task to a worker (Ticketmaster or shopping) and return its text
    reply. Falls back to raw response text on JSON-RPC decode failure.
    """
    sub = TaskSendParams(
        id=uuid.uuid4().hex,
        message=Message(role="user", parts=[TextPart(text=t) for t in texts]),
    )
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tasks/send",
        "params": sub.model_dump(),
    }

    r = requests.post(url, json=payload, timeout=40)
    r.raise_for_status()

    try:
        result_json = r.json()
        return result_json["result"]["status"]["message"]["parts"][0]["text"]
    except Exception:
        return r.text.strip()


def _rpc_error(rpc_id, msg):
    return jsonify({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32600, "message": msg}})


def _task_fail(rpc_id, task_id, msg):
    failed = Task(
        id=task_id,
        status=TaskStatus(
            state=TaskState.FAILED,
            message=Message(role="agent", parts=[TextPart(text=msg)]),
        ),
    )
    return jsonify(SendTaskResponse(id=rpc_id, result=failed).model_dump())


if __name__ == "__main__":
    app.run(port=5050)
