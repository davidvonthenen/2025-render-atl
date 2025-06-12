import os
import json
import datetime as _dt
from typing import List

from dataclasses import dataclass
from typing import List
import requests
import xml.etree.ElementTree as ET

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

API_ENDPOINT = "https://api.bestbuy.com/v1/products"
BESTBUY_API_KEY = os.getenv("BESTBUY_API_KEY")

app = Flask("BestBuyAgent")

###############################################################################
# Model
###############################################################################

@dataclass
class Product:
    name: str
    sale_price: float | None
    add_to_cart_url: str | None


###############################################################################
# Agent
###############################################################################
@app.post("/recommendation")
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
        answer = _search_recommendations(keyword)
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
def _build_url(search: str,
               page: int = 1,
               page_size: int = 5,
               sort: str = "salePrice",
               fields: str = "name,salePrice,addToCartUrl") -> str:
    """
    Construct the fully-qualified request URL.

    Parameters mirror Best Buy's documentation:
    https://developer.bestbuy.com/apis
    """
    # The Products API requires the search expression inside parentheses
    query = f"(name={search}*&name!=Plan*&name!=Refurb*&salePrice>1000&salePrice<5000&preowned=false)"

    # need to HTTP-encode the entire query string
    query = requests.utils.quote(query)
    # print(f"Query: {query}")

    params = {
        "apiKey": BESTBUY_API_KEY,
        "sort": sort,
        "show": fields,
        "format": "xml",          # We want XML, not JSON.
        "page": str(page),
        "pageSize": str(page_size),
    }
    # Manually glue params because parentheses in the path confuse requests
    param_str = "&".join(f"{k}={v}" for k, v in params.items())
    return f"{API_ENDPOINT}{query}?{param_str}"


def _parse_products(xml_text: str) -> List[Product]:
    """
    Parse <products>…</products> XML into a list of Product dataclass instances.
    """
    root = ET.fromstring(xml_text)
    products: list[Product] = []

    # Each child <product> is flat; grab the three nodes of interest.
    for p in root.iterfind("./product"):
        name         = p.findtext("name")
        price_text   = p.findtext("salePrice")          # May be missing/null.
        cart_url     = p.findtext("addToCartUrl")       # May be missing/null.
        sale_price   = float(price_text) if price_text else None
        products.append(Product(name=name,
                                sale_price=sale_price,
                                add_to_cart_url=cart_url))
    return products


def fetch_products(search: str,
                   *,
                   api_key: str | None = None,
                   page: int = 1,
                   page_size: int = 5) -> List[Product]:
    """
    High-level helper: query the API and return a list of Product objects.
    """
    url = _build_url(search)
    # print(f"Fetching products from: {url}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return _parse_products(resp.text)


def _search_recommendations(keyword: str) -> str:
    """
    Query Ticketmaster for the next few events in California.
    Returns a readable summary string.
    """
    products = fetch_products(keyword)

    pos = 0
    lines = [f"Recommended products based on term '{keyword}':"]
    for p in products:
        price = f"${p.sale_price:,.2f}"
        # print(f"{p.name}\n  {price}\n  {p.add_to_cart_url}\n")

        lines.append(f"• {p.name} – {price} @ {p.add_to_cart_url}")
        lines.append(f"• {p.name} – {price}")

        # exit after 2 products
        pos += 1
        if pos >= 2:
            break

    results = "\n".join(lines)
    # print(results)

    prompt = f"An AI Agent has requested you find recommended products to purchase for {keyword} from Best Buy. Here are the results:\n\n{results}\n\nProvide a concise summary and the output should be suitable for doing."
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
    app.run(port=7274)
