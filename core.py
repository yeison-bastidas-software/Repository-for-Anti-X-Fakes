"""Anti X-Fakes — core package (no LLM, no DB)."""
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
TIMEOUT = 30

POST_ID_RE = re.compile(
    r"(?:https?://)?(?:www\.|mobile\.)?(?:twitter\.com|x\.com)/"
    r"[A-Za-z0-9_]+/status/(\d+)", re.IGNORECASE
)

def extract_post_id(url):
    if not url: return None
    match = POST_ID_RE.search(url.strip())
    return match.group(1) if match else None

def fetch_post(post_id, post_url):
    try:
        resp = HTTP.get(f"https://api.fxtwitter.com/status/{post_id}", timeout=TIMEOUT)
        if resp.status_code == 200:
            tweet = resp.json().get("tweet") or {}
            if tweet.get("text"):
                author = tweet.get("author") or {}
                return {"text": tweet["text"], "author": author.get("name", "Unknown"), 
                        "handle": author.get("screen_name", ""), "date": tweet.get("created_at", "")}
    except (requests.RequestException, ValueError): pass

    try:
        resp = HTTP.get("https://publish.twitter.com/oembed", params={"url": post_url}, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", data.get("html", ""))).strip()
            if text:
                return {"text": text, "author": data.get("author_name", "Unknown"), 
                        "handle": data.get("author_url", "").rstrip("/").split("/")[-1], "date": ""}
    except (requests.RequestException, ValueError): pass
    return None

# --- Evidence Stage ---
CLAIMSEARCH_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

_STOPWORDS = {"a","an","the","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","shall","should","can","could","may","might","must",
    "of","on","in","to","for","with","by","at","from","as","and","or","but","not","no",
    "it","its","this","that","these","those","there","here","then","than","into","over",
    "under","about","after","before","between","during","per","via","el","la","los","las",
    "un","una","unos","unas","de","del","al","en","con","por","para","que","es","son",
    "fue","eran","ha","han","se","su","sus","lo","y","o","ni"}

def _keywords(text, limit=120):
    words = re.findall(r"[A-Za-z0-9áéíóúñÁÉÍÓÚÑ']+", text.lower())
    kept = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return (" ".join(kept) or text.strip())[:limit]

def search_fact_checks(claim, api_key):
    if not api_key: return []
    try:
        resp = HTTP.get(CLAIMSEARCH_ENDPOINT, timeout=TIMEOUT, params={
            "query": _keywords(claim), 
            "key": api_key, 
            "pageSize": 3
        })
        if resp.status_code != 200: 
            return []
        
        results = []
        for c in (resp.json().get("claims") or [])[:3]:
            claim_text = c.get("text", claim)
            source = next(iter(c.get("sourceClaims") or []), {})
            rating = (source.get("reviewRating") or {}).get("textualRating", "")
            publisher = (source.get("publisher") or {}).get("name", "")
            url = source.get("url") or c.get("reviewURL", "")
            
            results.append({
                "api": "Google Fact Check", 
                "title": claim_text,
                "url": url, 
                "snippet": f"{publisher}: {rating}" if rating else publisher
            })
        return results
    except (requests.RequestException, ValueError):
        return []

def search_tavily(claim):
    """Search the web using Tavily API (optimized for AI agents)."""
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return []
    try:
        resp = HTTP.post("https://api.tavily.com/search", timeout=TIMEOUT, json={
            "query": _keywords(claim),
            "max_results": 3,
            "search_depth": "basic",
            "api_key": api_key
        })
        if resp.status_code != 200:
            print(f"Tavily Search error: {resp.status_code} - {resp.text[:200]}")
            return []
        results = []
        for item in (resp.json().get("results") or [])[:3]:
            results.append({
                "api": "Tavily Search",
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", "")
            })
        return results
    except (requests.RequestException, ValueError) as e:
        print(f"Tavily Search exception: {e}")
        return []

def gather_evidence(claim, translated_claim=None):
    """Query FactCheck and Tavily Search in parallel."""
    key = os.environ.get("CLAIMSEARCH_API_KEY", "")
    query = translated_claim if translated_claim else claim
    
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(search_fact_checks, query, key),
            pool.submit(search_tavily, query),
        ]
        combined = []
        for future in as_completed(futures):
            try:
                combined.extend(future.result())
            except Exception:
                pass
    
    return combined