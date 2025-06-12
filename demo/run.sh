#!/usr/bin/env bash

# also needs:
# python -m spacy download en_core_web_sm

# Finally, run mitmdump with the ignore-hosts regex and the addon script:
mitmdump -q \
         --ssl-insecure \
         --listen-port 8080 \
         --ignore-hosts "^(localhost|127\.0\.0\.1):(6000|5050|6274|7274)$|^api\.openai\.com:443$|^app\.ticketmaster\.com:443$|^api\.bestbuy\.com:443$|^huggingface.\.co:443$|^api\.deepgram\.com:443$" \
         -s "dump_html_text.py"

# mitmdump \
#          --listen-port 8080 \
#          --ignore-hosts "^(localhost|127\.0\.0\.1):6000$" \
#          -s "dump_html_text.py"
