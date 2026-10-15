"""Anti X-Fakes — Gemini LLM stage.

Gemini (free tier) is the ONLY LLM used, per the project plan. It powers:
  1. extract_claims     — split post text into atomic verifiable claims.
  2. judge_claim        — verdict per claim grounded in retrieved evidence.
  3. aggregate_verdict  — combine claim labels into one post-level verdict.
  4. extended_analysis  — deeper write-up injected into the PDF template.

All prompts demand strict JSON back (except the PDF prose) and every reply is
parsed defensively so a malformed LLM answer can never crash the pipeline.
"""

import json
import os
import re

from google import genai

_MODEL_NAME = "gemini-3.8-flash"  # Gemini free-tier model (gemini-2.0-flash was retired by Google: the API returns 404 NOT_FOUND for it)
_client = None                   # lazily created singleton (one client per app)


def _get_client():
    """Build the Gemini client once, reading GEMINI_API_KEY from .env."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=key)
    return _client


def _ask(prompt):
    """Send one prompt to Gemini and return the plain-text reply."""
    response = _get_client().models.generate_content(
        model=_MODEL_NAME, contents=prompt
    )
    return (response.text or "").strip()


def _parse_json(text):
    """Extract the first JSON object/array embedded in an LLM reply.

    Models sometimes wrap JSON in prose or code fences; the greedy regex
    grabs everything between the outermost brackets and parses it.
    """
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON in LLM response")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Extract stage: split post text into atomic verifiable claims
# ---------------------------------------------------------------------------

MAX_CLAIMS = 8  # hard cap so a rambling post cannot explode the API budget


def extract_claims(post_text):
    """Return a list of atomic verifiable claim strings ([] if opinions only)."""
    prompt = (
        "You are a fact-checking analyst. From the social media post below, "
        "extract every ATOMIC VERIFIABLE CLAIM (a statement that can be "
        "checked against real-world evidence). Ignore pure opinions, jokes or "
        "calls to action.\n\n"
        "Post:\n" + post_text + "\n\n"
        'Reply with ONLY valid JSON: {"claims": ["claim 1", "claim 2"]} — '
        "an empty list if the post contains no verifiable claims."
    )
    data = _parse_json(_ask(prompt))
    claims = [str(c).strip() for c in data.get("claims", []) if str(c).strip()]
    return claims[:MAX_CLAIMS]


# ---------------------------------------------------------------------------
# Judge stage: one verdict per claim, grounded in retrieved evidence
# ---------------------------------------------------------------------------

CLAIM_LABELS = ("TRUE", "FALSE", "MISLEADING", "UNVERIFIABLE")  # per-claim set


def judge_claim(claim, evidence):
    """Ask Gemini to label one claim using ONLY the supplied evidence.

    Returns {claim, label, confidence, reasoning, sources, evidence}.
    Invalid labels fall back to UNVERIFIABLE; confidence is clamped to 0-100.
    """
    if evidence:
        ev_block = "\n".join(
            f"[{i}] ({e['api']}) {e['title']} | {e['snippet']} | {e['url']}"
            for i, e in enumerate(evidence, 1)
        )
    else:
        ev_block = "(no external evidence found)"

    prompt = (
        "You are a rigorous fact-checking judge. Judge the claim STRICTLY on "
        "the evidence provided; do not invent sources.\n\n"
        f"Claim: {claim}\n\nEvidence:\n{ev_block}\n\n"
        "Return ONLY valid JSON with this exact shape:\n"
        '{"label": "TRUE|FALSE|MISLEADING|UNVERIFIABLE", "confidence": 0-100, '
        '"reasoning": "short reasoning", "sources": ["url1", "url2"]}\n\n'
        "Rules: use UNVERIFIABLE when the evidence is insufficient to judge. "
        "List in 'sources' only URLs from the evidence actually used."
    )
    data = _parse_json(_ask(prompt))

    label = str(data.get("label", "")).upper()
    if label not in CLAIM_LABELS:
        label = "UNVERIFIABLE"
    try:
        confidence = max(0, min(100, int(float(data.get("confidence", 0)))))
    except (TypeError, ValueError):
        confidence = 0

    return {
        "claim": claim,
        "label": label,
        "confidence": confidence,
        "reasoning": str(data.get("reasoning", "")),
        "sources": [str(s) for s in data.get("sources", [])],
        "evidence": evidence,  # kept so the UI can cite every source used
    }


def aggregate_verdict(claim_results):
    """Collapse per-claim labels into one post-level verdict.

    Precedence fixed by the plan: any FALSE -> FALSE; else any MISLEADING ->
    MISLEADING; else TRUE if something checked out; else UNVERIFIABLE.
    """
    labels = {r["label"] for r in claim_results}
    if "FALSE" in labels:
        return "FALSE"
    if "MISLEADING" in labels:
        return "MISLEADING"
    if "TRUE" in labels:
        return "TRUE"
    return "UNVERIFIABLE"


# ---------------------------------------------------------------------------
# PDF stage: extended analysis for the print template
# ---------------------------------------------------------------------------

def extended_analysis(post, claim_results, verdict):
    """Generate the deeper report text (context, evidence, background, conclusion).

    Called only when the user clicks Download PDF; receives the cached findings
    so the model expands on real evidence instead of guessing anew.
    """
    summary = "\n".join(
        f"- Claim: {r['claim']}\n  Verdict: {r['label']} (confidence {r['confidence']}%)\n"
        f"  Reasoning: {r['reasoning']}"
        for r in claim_results
    ) or "- (no verifiable claims: opinion post)"

    prompt = (
        "You are writing an extended fact-checking report for a PDF document.\n\n"
        f"Original post by @{post['handle']} ({post['author']}):\n{post['text']}\n\n"
        f"Per-claim findings:\n{summary}\n\n"
        f"Overall verdict: {verdict}\n\n"
        "Write an extended analysis in clear English prose with these sections:\n"
        "1. Deeper context of the topic\n"
        "2. Additional evidence discussion\n"
        "3. Related background\n"
        "4. Final nuanced conclusion\n"
        "Plain text only, no markdown symbols. 250-450 words total."
    )
    return _ask(prompt)
