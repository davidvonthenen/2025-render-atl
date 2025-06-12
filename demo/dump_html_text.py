"""
dump_html_and_start_flask.py

1. Acts as a mitmproxy addon to strip and dump readable HTML text + entities.
2. Spawns a Flask server in a background thread, so you can visit http://127.0.0.1:5000/
   to see ChatGPT-generated recommendations based on `products.txt`.

Run it via:
mitmdump -q \
         --listen-port 8080 \
         --ignore-hosts "^(localhost|127\.0\.0\.1):6000$" \
         -s "dump_html_text.py"

mitmdump \
         --listen-port 8080 \
         --ignore-hosts "^(localhost|127\.0\.0\.1):6000$" \
         -s "dump_html_text.py"

Flask will start automatically (no need to use `python script.py` separately).
"""

import re
import threading
import os
import openai

from mitmproxy import http

# -------------------------- mitmproxy ADDON --------------------------

# Attempt to import BeautifulSoup; otherwise, fallback to a regex stripper
try:
    from bs4 import BeautifulSoup  # pip install beautifulsoup4
    def strip_tags(html: str) -> str:
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
except ModuleNotFoundError:
    tag_re     = re.compile(r"<[^>]+>")
    script_re  = re.compile(r"(?is)<(script|style).*?>.*?</\1>")
    def strip_tags(html: str) -> str:
        html = script_re.sub(" ", html)
        return tag_re.sub(" ", html)

MAX_CHARS = 500
count = 0

def response(flow: http.HTTPFlow) -> None:
    """
    mitmproxy hook: called on each HTTP response.
    1. If it's HTML, strip tags and dump plain text.
    2. Skip trivial or unwanted snippets.
    3. Save to <count>.txt and extract PRODUCT entities via spaCy.
    4. Append the most common product to products.txt.
    """
    global count

    ctype = flow.response.headers.get("content-type", "").lower()
    if "text/html" not in ctype:
        return

    # Get plain text
    text = strip_tags(flow.response.text)

    # Skip very short or known boilerplate
    if len(text) < 100:
        return
    if text.startswith("The video showcases the product in use."):
        return
    if "Check each product page for other buying options." in text:
        return
    if "Results Filters" in text:
        return

    # Print a snippet to stdout
    print("\n" + "=" * 80)
    print(f"URL   : {flow.request.url}")
    print(f"Length: {len(text)} characters")
    print("-" * 80)
    print(text[:MAX_CHARS] + ("…" if len(text) > MAX_CHARS else ""))
    print("=" * 80)

    # Save full text + URL to <count>.txt
    # with open(f"{count}.txt", "w", encoding="utf-8") as f:
    #     f.write(flow.request.url + "\n\n")
    #     f.write(text)

    # Try spaCy NER to extract PRODUCT entities
    try:
        import spacy
        from collections import Counter

        # Load the model once per invocation (if performance becomes an issue,
        # you could load nlp globally instead of inside each call).
        nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)
        product_entities = []

        print("\nNamed Entities:")
        for ent in doc.ents:
            # with open(f"{count}_entities.txt", "a", encoding="utf-8") as f_e:
            #     f_e.write(f"{ent.text} ({ent.label_})\n")

            # Only keep PRODUCT entities longer than 9 chars
            if ent.label_ == "PRODUCT" and len(ent.text) >= 10:
                product_entities.append(ent.text)

        if not product_entities:
            print("No product entities found.")
        else:
            counter = Counter(product_entities)
            most_common, freq = counter.most_common(1)[0]
            print(f"Most common entity: {most_common} (count: {freq})")
            # Append that “winner” to products.txt
            with open("products.txt", "a", encoding="utf-8") as f_p:
                f_p.write(f"{most_common}\n")

    except ImportError:
        print("spaCy not installed—skipping named entity extraction.")

    count += 1
