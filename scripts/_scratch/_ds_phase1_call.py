import json, urllib.request, sys, time, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

CONSULT = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_ds_phase1_consult.md"
OUT_FILE = r"C:\Users\victor morrow\Git-projects\Pondr\scripts\_scratch\_ds_phase1_response.md"

prompt = open(CONSULT, encoding="utf-8").read()

payload = json.dumps({
    "model": "deepseek-v4-pro:cloud",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 16384,
    "temperature": 0.7,
    "stream": False,
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:11434/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)

last_err = None
for attempt in (1, 2):
    try:
        t0 = time.time()
        print(f"[attempt {attempt}] POSTing to ollama deepseek-v4-pro:cloud (timeout=1800s)...", file=sys.stderr, flush=True)
        resp = urllib.request.urlopen(req, timeout=1800)
        raw = resp.read()
        dt = time.time() - t0
        print(f"[attempt {attempt}] got response in {dt:.1f}s, {len(raw)} bytes", file=sys.stderr, flush=True)
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        print(content, flush=True)
        sys.exit(0)
    except Exception as e:
        last_err = e
        print(f"[attempt {attempt}] FAILED: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        time.sleep(5)

print("BOTH ATTEMPTS FAILED", file=sys.stderr)
print(f"LAST ERROR: {type(last_err).__name__}: {last_err}", file=sys.stderr)
sys.exit(1)