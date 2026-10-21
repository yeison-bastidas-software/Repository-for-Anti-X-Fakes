"""Anti X-Fakes — Gemini LLM stage."""
import json
import os
import re
import time
from google import genai

class QuotaExceededError(RuntimeError): pass

_MODEL_NAME = "gemini-3.8-flash"
_client = None
MAX_CLAIMS = 3  # Optimized: 3 claims is the sweet spot for accuracy vs token cost.

def _get_client():
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key: raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=key)
    return _client

def _call_with_retry(fn, max_retries=3):
    for attempt in range(max_retries):
        try: return fn()
        except Exception as exc:
            msg = str(exc).upper()
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                raise QuotaExceededError(str(exc)) from exc
            if attempt == max_retries - 1: raise
            if "503" in msg or "UNAVAILABLE" in msg:
                time.sleep(2 * (2 ** attempt))
            else: raise

def _ask(prompt):
    def call(): return _get_client().models.generate_content(model=_MODEL_NAME, contents=prompt)
    return (_call_with_retry(call).text or "").strip()

def _parse_json(text):
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match: raise ValueError("no JSON in LLM response")
    return json.loads(match.group(0))

def extract_claims(post_text):
    prompt = (
        "You are a fact-checking analyst. Extract up to 3 ATOMIC VERIFIABLE CLAIMS from the post below. "
        "Ignore opinions, jokes, or calls to action.\n\nPost:\n" + post_text + "\n\n"
        'Reply with ONLY valid JSON: {"claims": ["claim 1", "claim 2"]}'
    )
    data = _parse_json(_ask(prompt))
    return [str(c).strip() for c in data.get("claims", []) if str(c).strip()][:MAX_CLAIMS]

def batch_translate_claims(claims):
    """Translate a list of claims to English in a single API call."""
    if not claims:
        return []
    claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims, 1))
    prompt = (
        "Translate the following claims from Spanish (or any language) to English. "
        "Preserve the original meaning. Return ONLY a valid JSON array of strings.\n\n"
        f"Claims:\n{claims_text}\n\n"
        'Reply with ONLY: ["translation 1", "translation 2", ...]'
    )
    data = _parse_json(_ask(prompt))
    translations = [str(t).strip() for t in data if str(t).strip()]
    return translations[:len(claims)]

def judge_all_claims(claims, all_evidence):
    """Judges ALL claims in a SINGLE API call to maximize efficiency."""
    evidence_blocks = []
    for i, (claim, evidence) in enumerate(zip(claims, all_evidence), 1):
        ev_text = "\n".join(f"  [{j}] ({e['api']}) {e['title']} | {e['snippet']} | {e['url']}" for j, e in enumerate(evidence, 1)) or "  (no external evidence found)"
        evidence_blocks.append(f"Claim {i}: {claim}\nEvidence:\n{ev_text}")

    prompt = (
        "You are a rigorous fact-checking judge. Judge the following claims STRICTLY on the provided evidence.\n\n"
        + "\n\n".join(evidence_blocks) + "\n\n"
        "Return ONLY a valid JSON array of objects with this exact shape:\n"
        '[{"claim": "...", "label": "TRUE|FALSE|MISLEADING|UNVERIFIABLE", "confidence": 0-100, "reasoning": "...", "sources": ["url1"]}]'
    )
    
    data = _parse_json(_ask(prompt))
    results = []
    for i, item in enumerate(data):
        label = str(item.get("label", "")).upper()
        if label not in ("TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE"): label = "UNVERIFIABLE"
        try: confidence = max(0, min(100, int(float(item.get("confidence", 0)))))
        except: confidence = 0
        
        results.append({
            "claim": claims[i] if i < len(claims) else "Unknown",
            "label": label,
            "confidence": confidence,
            "reasoning": str(item.get("reasoning", "")),
            "sources": [str(s) for s in item.get("sources", [])],
            "evidence": all_evidence[i] if i < len(all_evidence) else []
        })
    return results

def aggregate_verdict(claim_results):
    labels = [r.get("label", "UNVERIFIABLE") for r in claim_results]
    if not labels:
        return "UNVERIFIABLE"
    
    true_count = labels.count("TRUE")
    false_count = labels.count("FALSE")
    unverifiable_count = labels.count("UNVERIFIABLE")
    total = len(labels)
    
    if true_count == total:
        return "TRUE"
    
    if false_count == total:
        return "FALSE"
    
    if unverifiable_count == total:
        return "UNVERIFIABLE"
    
    return "MISLEADING"

def extended_analysis(post, claim_results, verdict):
    summary = "\n".join(f"- Claim: {r['claim']}\n  Verdict: {r['label']} ({r['confidence']}%)\n  Reasoning: {r['reasoning']}" for r in claim_results) or "- (opinion post)"
    prompt = (
        f"Original post by @{post['handle']} ({post['author']}):\n{post['text']}\n\n"
        f"Per-claim findings:\n{summary}\n\nOverall verdict: {verdict}\n\n"
        "Write an extended analysis in clear English prose with EXACTLY these 4 section titles, each on its own separate line:\n"
        "1. Deeper context of the topic\n"
        "2. Additional evidence discussion\n"
        "3. Related background\n"
        "4. Final nuanced conclusion\n\n"
        "IMPORTANT RULES:\n"
        "- Each section title MUST be on its own separate line (do not put content on the same line as the title).\n"
        "- Every sentence MUST end with a period (.).\n"
        "- Use proper punctuation and NORMAL spacing between words.\n"
        "- Plain text ONLY. NO markdown symbols (no asterisks, no bold, no hashtags).\n"
        "- Total length: 250-450 words."
    )
    return _ask(prompt)