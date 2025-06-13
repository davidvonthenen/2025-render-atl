"""
MCP worker that talks directly to Ticketmaster via the `ticketpy` SDK.

• Exposes a JSON-RPC `/tasks` endpoint.
• Accepts **only** the artist/band search term in `message.parts[0].text`.
• Hard-codes search to California (state_code="CA").
• Returns a concise, human-readable list of upcoming events (max 5 lines).

Environment variables
---------------------
TICKETMASTER_API_KEY : Ticketmaster Discovery API key
"""

import os
import json
import datetime as _dt
from typing import List

from flask import Flask, request, jsonify
from common.types import (
    A2ARequest,
    SendTaskRequest,
    SendTaskResponse,
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart,
)

import openai
import ticketpy

openai.api_key  = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

TM_API_KEY = os.environ["TICKETMASTER_API_KEY"]
tm_client = ticketpy.ApiClient(TM_API_KEY)

app = Flask("TicketmasterAgent")


# --------------------------- HTTP entry ---------------------------
@app.post("/events")
def handle_tasks():
    req_json = request.get_json(force=True)
    try:
        rpc_req: A2ARequest = A2ARequest.validate_python(req_json)
    except Exception as exc:
        return _rpc_err(req_json.get("id"), f"Invalid JSON-RPC: {exc}")

    if not isinstance(rpc_req, SendTaskRequest):
        return _rpc_err(rpc_req.id, "Unsupported method")

    params = rpc_req.params
    if not params.message.parts:
        return _rpc_err(rpc_req.id, "No search term supplied")

    keyword = params.message.parts[0].text.strip()
    try:
        answer = _search_events(keyword)
    except Exception as exc:
        answer = f"Ticketmaster lookup failed: {exc}"

    done_task = Task(
        id=params.id,
        status=TaskStatus(
            state=TaskState.COMPLETED,
            message=Message(role="agent", parts=[TextPart(text=answer)]),
        ),
    )
    return jsonify(SendTaskResponse(id=rpc_req.id, result=done_task).model_dump())


# ----------------------- helpers ----------------------------------
def _search_events(keyword: str) -> str:
    """
    Query Ticketmaster for the next few events in California.
    Returns a readable summary string.
    """
    pages = tm_client.events.find(keyword=keyword, state_code="CA").limit()

    events: List = list(pages)

    if not events:
        return

    def _fmt_date(ev):
        date = getattr(ev, "local_start_date", None)  # e.g. 2025-09-14
        time = getattr(ev, "local_start_time", None)  # e.g. 20:00:00
        if date:
            try:
                d = _dt.datetime.strptime(date, "%Y-%m-%d").strftime("%b %d %Y")
            except ValueError:
                d = date
        else:
            d = "TBA"
        return f"{d} {time or ''}".strip()

    lines = [f"Upcoming events in California for “{keyword}”:"]
    for ev in events:
        # print(f"Processing event: {ev}")
        # print("\n\n")

        name = getattr(ev, "name", "Unnamed Event")

        # venue
        venue = getattr(ev, "venues", None)
        if venue and len(venue) > 0:
            venue_name = getattr(venue[0], "name", "Unknown Venue")
            city = getattr(venue[0], "city", "")
        else:
            venue_name = ""
            city = ""

        lines.append(f"• {_fmt_date(ev)} – {name} @ {venue_name} ({city})")

    results = "\n".join(lines)
    # print(results)

    prompt = f"An AI Agent has requested you find all concert information for {keyword}. Here are the results:\n\n{results}\n\nProvide a concise summary."
    resp = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=32768
    )

    return resp.choices[0].message.content.strip()


def _rpc_err(rpc_id, msg):
    return jsonify(
        {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "error": {"code": -32600, "message": msg},
        }
    )


if __name__ == "__main__":
    # Same port the coordinator expects
    app.run(port=6274)
