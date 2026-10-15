"""Anti X-Fakes — core package (no LLM logic lives here).

Responsibilities, exactly as listed in the project plan:
  1. URL validation / numeric post-id extraction.
  2. SQLite cache of verdicts (Python standard library only).
  3. External service clients:
       - FxTwitter API          (primary post fetcher, no key needed)
       - X oEmbed endpoint      (fallback post fetcher, no key needed)
       - Google Fact Check API  (ClaimSearch, key from .env)
       - GDELT DOC 2.0 API      (global news cross-reference, no key needed)
       - Wikipedia REST API     (background/entity check, no key needed)

Every HTTP call shares one requests.Session with sane timeouts, and every
failure degrades gracefully (empty result) instead of crashing the pipeline.
"""

import json
import os
import re
import sqlite3

import requests

# Cache database lives next to the code; created on first use.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verdicts.db")

# The five verdicts the program may ever return (plan section "Verdicts").
VALID_VERDICTS = ("TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE", "OPINION")

# One shared session => connection pooling/reuse across all API calls.
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "AntiXFakes/1.0"
TIMEOUT = 15  # seconds, applied to every outbound request


# ---------------------------------------------------------------------------
# URL validation / post id extraction
# ---------------------------------------------------------------------------

# Matches x.com / twitter.com (www/mobile optional) status URLs and captures
# the numeric post id. Anything else is rejected before network calls happen.
POST_ID_RE = re.compile(
    r"(?:https?://)?(?:www\.|mobile\.)?(?:twitter\.com|x\.com)/"
    r"[A-Za-z0-9_]+/status/(\d+)",
    re.IGNORECASE,
)


def extract_post_id(url):
    """Return the numeric post id found in an X/Twitter status URL, or None."""
    if not url:
        return None
    match = POST_ID_RE.search(url.strip())
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# SQLite cache (standard library only, used ONLY as a cache of verdicts)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS verdicts (
    post_id     TEXT PRIMARY KEY,
    post_url    TEXT,
    created_at  TEXT,
    payload     TEXT
);
"""


def _connect():
    """Open the cache database and make sure the table exists."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def save_result(post_id, post_url, created_at, payload):
    """Insert or overwrite the cached verdict for a post id."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO verdicts (post_id, post_url, created_at, payload)"
            " VALUES (?, ?, ?, ?)",
            (post_id, post_url, created_at, json.dumps(payload)),
        )


def load_result(post_id):
    """Return the cached verdict dict for a post id, or None if absent."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT post_id, post_url, created_at, payload FROM verdicts WHERE post_id = ?",
            (post_id,),
        ).fetchone()
    if not row:
        return None
    data = json.loads(row[3])
    # Attach the DB columns so templates/PDF can show link and timestamp.
    data["_post_id"], data["_post_url"], data["_created_at"] = row[0], row[1], row[2]
    return data


# ---------------------------------------------------------------------------
# Fetch stage — FxTwitter primary, X oEmbed fallback (one retry)
# ---------------------------------------------------------------------------

FXTWITTER_URL = "https://api.fxtwitter.com/status/{post_id}"
OEMBED_URL = "https://publish.twitter.com/oembed"


def fetch_post(post_id, post_url):
    """Fetch post content. Returns dict(text, author, handle, date) or None.

    Order per plan: try FxTwitter first; if it fails, retry once against the
    official oEmbed endpoint; if both fail return None (caller shows
    "Post not found or unavailable").
    """
    # Primary source: FxTwitter (no API key required).
    try:
        resp = HTTP.get(FXTWITTER_URL.format(post_id=post_id), timeout=TIMEOUT)
        if resp.status_code == 200:
            tweet = (resp.json().get("tweet") or {})
            if tweet.get("text"):
                author = tweet.get("author") or {}
                return {
                    "text": tweet["text"],
                    "author": author.get("name", "Unknown"),
                    "handle": author.get("screen_name", ""),
                    "date": tweet.get("created_at", ""),
                }
    except (requests.RequestException, ValueError):
        pass  # fall through to the oEmbed retry

    # Fallback: X oEmbed returns an HTML snippet; strip tags to plain text.
    try:
        resp = HTTP.get(OEMBED_URL, params={"url": post_url}, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", data.get("html", ""))).strip()
            if text:
                return {
                    "text": text,
                    "author": data.get("author_name", "Unknown"),
                    "handle": data.get("author_url", "").rstrip("/").split("/")[-1],
                    "date": "",  # oEmbed does not expose the original date
                }
    except (requests.RequestException, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Evidence stage — ClaimSearch + GDELT + Wikipedia (top 3 results each)
# ---------------------------------------------------------------------------

CLAIMSEARCH_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
WIKI_SEARCH = "https://en.wikipedia.org/api/rest_v1/page/search"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Each evidence item is normalized to this shape so downstream stages and
# templates treat all three APIs uniformly: {api, title, url, snippet}.
TOP_RESULTS = 3


def search_fact_checks(claim, api_key):
    """Google Fact Check Tools API (ClaimSearch). Returns up to 3 items."""
    if not api_key:
        return []  # feature silently degrades if the key is missing
    try:
        resp = HTTP.get(CLAIMSEARCH_ENDPOINT, timeout=TIMEOUT, params={
            "query": claim[:500],   # API limit on query length
            "key": api_key,
            "pageSize": TOP_RESULTS,
        })
        if resp.status_code != 200:
            return []
        results = []
        for c in (resp.json().get("claims") or [])[:TOP_RESULTS]:
            source = next(iter(c.get("sourceClaims") or []), {})
            rating = (source.get("reviewRating") or {}).get("rating", "")
            results.append({
                "api": "Google Fact Check",
                "title": c.get("claimText", claim),
                "url": source.get("url") or c.get("reviewURL", ""),
                "snippet": f"{rating} — {c.get('reviewURL', '')}".strip(" —"),
            })
        return results
    except (requests.RequestException, ValueError):
        return []


def search_gdelt(claim):
    """GDELT DOC 2.0 API — global news cross-reference. Returns up to 3 items."""
    try:
        resp = HTTP.get(GDELT_API, timeout=TIMEOUT, params={
            "query": claim[:300],
            "mode": "ArtList",     # article list format
            "maxrecords": TOP_RESULTS,
            "format": "json",
            "sort": "Latest",
        })
        if resp.status_code != 200:
            return []
        return [
            {
                "api": "GDELT News",
                "title": a.get("title", ""),
                "url": a.get("url", ""),
                "snippet": a.get("domain", ""),
            }
            for a in (resp.json().get("articles") or [])[:TOP_RESULTS]
        ]
    except (requests.RequestException, ValueError):
        return []


def search_wikipedia(claim):
    """Wikipedia REST API — background/entity info. Returns up to 3 items.

    Two calls per hit: search to find page titles, then summary for extract +
    canonical URL (falling back to the constructed wiki URL if summary fails).
    """
    try:
        resp = HTTP.get(WIKI_SEARCH, timeout=TIMEOUT,
                        params={"query": claim[:200], "limit": TOP_RESULTS})
        if resp.status_code != 200:
            return []
        results = []
        for page in (resp.json().get("pages") or [])[:TOP_RESULTS]:
            title = page.get("title", "")
            if not title:
                continue
            quoted = requests.utils.quote(title)
            extract = page.get("excerpt", "")
            url = f"https://en.wikipedia.org/wiki/{quoted}"
            try:
                s = HTTP.get(WIKI_SUMMARY.format(title=quoted), timeout=TIMEOUT)
                if s.status_code == 200:
                    sj = s.json()
                    extract = sj.get("extract", extract)
                    url = (sj.get("content_urls", {}).get("desktop", {}).get("page") or url)
            except (requests.RequestException, ValueError):
                pass  # keep the constructed URL + search excerpt
            results.append({"api": "Wikipedia", "title": title, "url": url, "snippet": extract})
        return results
    except (requests.RequestException, ValueError):
        return []


def gather_evidence(claim):
    """Query the three evidence APIs for one claim; top 3 results from each.

    The ClaimSearch key is read lazily from the environment so this module
    keeps working even when that optional source returns nothing.
    """
    key = os.environ.get("CLAIMSEARCH_API_KEY", "")
    return (search_fact_checks(claim, key)
            + search_gdelt(claim)
            + search_wikipedia(claim))
