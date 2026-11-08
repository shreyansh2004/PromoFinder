"""Quick connectivity check for the configured LLM provider (does not print the key)."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
c = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL") or None)
models = sys.argv[1:] or [os.environ.get("LLM_MODEL", "")]
for m in models:
    for json_mode in (True, False):
        kw = dict(model=m, messages=[{"role": "user", "content": 'Reply with the JSON object {"ok": true}'}])
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        try:
            r = c.chat.completions.create(**kw)
            print(m, "json_mode" if json_mode else "plain", "OK", (r.choices[0].message.content or "")[:40])
        except Exception as e:
            print(m, "json_mode" if json_mode else "plain", "ERR", str(e)[:200])
