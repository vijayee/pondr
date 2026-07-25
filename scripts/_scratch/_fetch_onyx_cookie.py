"""One-off: fetch Onyx sessions via COOKIE auth (fastapiusersauth), not Bearer.

This Onyx deployment uses fastapi-users cookie transport. The Bearer API keys
are separate synthetic users (``api_key__claude*@...``) that own ZERO chat
sessions, so ``scripts/fetch_onyx_sessions.py`` (Bearer) returns 0 sessions.
The browser authenticates as the real user ("Sir") via the ``fastapiusersauth``
cookie. This script reuses the fetcher's PURE pipeline (``fetch_all``,
``write_jsonl``) with a cookie-authed ``get_json`` seam so the listing + detail
fetch see Sir's sessions.

SECURITY -- the cookie is a full-access-as-Sir secret. Env-var ONLY
(``ONYX_COOKIE``): never written to disk, never printed, never logged. The
output ``sessions.jsonl`` is Sir's chat data -- LOCAL + GITIGNORED, never
uploaded (per user directive: Onyx data must be sanitized before any external
upload). This script lives under the gitignored ``_scratch/`` and is never
committed.
"""
import os
import sys
from pathlib import Path

# scripts/ is the parent of _scratch/, and scripts import each other as
# top-level modules (no __init__.py), so put scripts/ on the path and import
# the fetcher by its module name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests  # noqa: E402

from fetch_onyx_sessions import fetch_all, write_jsonl  # noqa: E402


def make_get_json_cookie(base_url: str, cookie: str):
    """Cookie-authed ``get_json(url, params)`` seam for the fetcher pipeline."""
    api_base = base_url.rstrip("/") + "/api"
    session = requests.Session()
    session.cookies.set("fastapiusersauth", cookie)  # the fastapi-users cookie
    session.headers.update({"Accept": "application/json"})
    # No Authorization header -- this deploy uses cookie transport, not Bearer.

    def get_json(url, params=None):
        full = api_base + "/" + url.lstrip("/")
        resp = session.get(full, params=params or {}, timeout=30)
        if not resp.ok:
            body = resp.text[:500]  # url carries no secret; cookie is not in it
            raise RuntimeError(
                f"Onyx API {resp.status_code} at {url} (params={params}): {body}"
            )
        return resp.json()

    return get_json


def main() -> int:
    base = os.environ.get("ONYX_BASE_URL", "").strip().rstrip("/")
    cookie = os.environ.get("ONYX_COOKIE", "").strip()
    if not base or not cookie:
        print("ERROR: set ONYX_BASE_URL and ONYX_COOKIE env "
              "(the cookie is a SECRET -- never printed)", file=sys.stderr)
        return 1

    g = make_get_json_cookie(base, cookie)

    # Auth sanity: confirm we are the real user (not a synthetic api_key user).
    # Print only role + email prefix -- no token, no full email if it looks keyed.
    try:
        me = g("/me")
        email = str(me.get("email", "?"))
        role = me.get("role", "?")
        sup = me.get("is_superuser", "?")
        print(f"authed as: email={email}  role={role}  is_superuser={sup}",
              flush=True)
        if email.startswith("api_key__"):
            print("ERROR: cookie auth still resolved to a synthetic api_key user "
                  "-- the cookie did not authenticate as the real user.",
                  file=sys.stderr)
            return 1
    except Exception as e:  # noqa: BLE001
        print(f"WARN: /me check failed ({e}); continuing to the fetch",
              file=sys.stderr)

    out = "data/training/strm_graduation/sessions.jsonl"
    try:
        # include_failed=True: an "errored message" just means one tool call
        # failed mid-thread; the surrounding user/assistant turns are still
        # valid serve-trace material. The long coding chats (the richest serve
        # data) carry such errors, so excluding them would lose exactly what we
        # want. Downstream serve-trace capture filters to user/assistant text
        # turns regardless.
        records = fetch_all(g, query=None, page_size=50, limit=None,
                            include_failed=True, verbose=True)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: fetch failed -- {e}", file=sys.stderr)
        return 1

    write_jsonl(records, out)
    n_msg = sum(len(r["messages"]) for r in records)
    print(f"DONE. {len(records)} sessions, {n_msg} messages -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())