import os
import threading
import time
import re
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import core
import llm
from llm import QuotaExceededError

# Flask app and in-memory global job store (no database)
app = Flask(__name__)
app.secret_key = "anti-x-fakes"
JOBS = {}
_JOBS_LOCK = threading.Lock()

# Thread-safe accessors for the shared job dictionary
def _new_job(token):
    with _JOBS_LOCK:
        JOBS[token] = {"stage": 0, "total": 4, "status": "Starting investigation…", "done": False, "error": None, "post_id": None, "started_at": time.time()}

def _update_job(token, **changes):
    with _JOBS_LOCK:
        if token in JOBS: JOBS[token].update(changes)

def _get_job(token):
    with _JOBS_LOCK:
        return dict(JOBS.get(token, {}))

# Maps raw technical errors into clear, user-facing messages
FRIENDLY_ERRORS = {
    "429": "Gemini's daily quota is exhausted. Please try again tomorrow (resets at midnight PT).",
    "RESOURCE_EXHAUSTED": "Gemini's daily quota is exhausted. Please try again tomorrow.",
    "Invalid URL": "That doesn't look like a valid X post link. Try something like: https://x.com/user/status/1234567890",
    "Post not found": "We couldn't access that post. It may have been deleted or is private.",
    "GEMINI_API_KEY": "The AI service is not configured. Contact the administrator.",
    "CLAIMSEARCH_API_KEY": "Fact-checking service is not configured. Contact the administrator.",
}

def _friendly_error(raw_error):
    msg = str(raw_error)
    for key, friendly in FRIENDLY_ERRORS.items():
        if key in msg:
            return friendly
    return f"Something went wrong: {msg}"

# Main pipeline: runs in a background thread and reports progress by stage
# (validate URL -> fetch post -> extract claims -> translate -> gather
# evidence -> judge -> aggregate verdict)
def run_pipeline(token, post_url):
    try:
        _update_job(token, stage=1, status="Validating URL and locating the post…")
        post_id = core.extract_post_id(post_url)
        if not post_id:
            _update_job(token, done=True, error="Invalid URL. Paste a valid X post link.")
            return

        _update_job(token, stage=2, status="Fetching post content…")
        post = core.fetch_post(post_id, post_url)
        if not post:
            _update_job(token, done=True, error="Post not found or unavailable.")
            return

        _update_job(token, stage=3, status="Extracting verifiable claims with the LLM…")
        claims = llm.extract_claims(post["text"]) or []

        # Post without verifiable facts: labeled OPINION, no further search
        if not claims:
            payload = {"verdict": "OPINION", "confidence": 100, "reasoning": "No verifiable factual claims found.", "sources": [], "claims": [], "post": post, "elapsed_seconds": round(time.time() - _get_job(token)["started_at"], 1)}
            with _JOBS_LOCK: JOBS[token].update({"done": True, "post_id": post_id, "payload": payload})
            return

        _update_job(token, total=6, stage=4, status="Translating claims to English for better search coverage…")
        try:
            translated_claims = llm.batch_translate_claims(claims)
        except QuotaExceededError:
            _update_job(token, done=True, error="Gemini daily quota exhausted (429). Please try again tomorrow.")
            return
        except Exception as exc:
            traceback.print_exc()
            translated_claims = claims

        _update_job(token, stage=5, status="Gathering evidence for all claims in parallel…")
        all_evidence = [
            core.gather_evidence(original, translated)
            for original, translated in zip(claims, translated_claims)
        ]

        _update_job(token, stage=6, status="Judging claims against retrieved evidence…")

        try:
            results = llm.judge_all_claims(claims, all_evidence)
        except QuotaExceededError:
            _update_job(token, done=True, error="Gemini daily quota exhausted (429). Please try again tomorrow.")
            return
        except Exception as exc:
            traceback.print_exc()
            _update_job(token, done=True, error=f"LLM judging failed: {exc}")
            return

        # Final consolidation: overall verdict, average confidence, combined
        # reasoning and deduplicated sources
        verdict = llm.aggregate_verdict(results)
        confidence = round(sum(r["confidence"] for r in results) / len(results))
        reasoning = " ".join(f"{r['claim']} → {r['reasoning']}" for r in results)

        sources, seen = [], set()
        for r in results:
            for e in r.get("evidence", []):
                if e["url"] and e["url"] not in seen:
                    seen.add(e["url"])
                    sources.append(e)

        payload = {
            "verdict": verdict, "confidence": confidence, "reasoning": reasoning,
            "sources": sources, "claims": results, "post": post,
            "elapsed_seconds": round(time.time() - _get_job(token)["started_at"], 1)
        }

        with _JOBS_LOCK:
            JOBS[token].update({"done": True, "post_id": post_id, "payload": payload})

    except RuntimeError as exc:
        _update_job(token, done=True, error=f"Configuration error: {exc}. Check .env keys.")
    except Exception as exc:
        traceback.print_exc()
        _update_job(token, done=True, error=f"Verification failed: {exc}")

# Web routes: initial form, processing screen, progress polling and result page
@app.route("/")
def home(): return render_template("home.html")

@app.route("/check", methods=["POST"])
def check():
    post_url = (request.form.get("url") or "").strip()
    token = os.urandom(8).hex()
    _new_job(token)
    threading.Thread(target=run_pipeline, args=(token, post_url), daemon=True).start()
    session["last_token"] = token
    return redirect(url_for("processing", token=token))

@app.route("/processing")
def processing(): return render_template("processing.html", token=request.args.get("token", ""))

@app.route("/progress")
def progress():
    job = _get_job(request.args.get("token", ""))
    if not job: return jsonify({"percent": 0, "status": "", "done": True, "error": "Session expired.", "post_id": None})
    stage, total = job["stage"], max(1, job["total"])
    percent = 100 if (job["done"] and not job["error"]) else min(99, round(stage / total * 100))
    return jsonify({"percent": percent, "status": job["status"], "done": job["done"], "error": job["error"], "post_id": job["post_id"]})

@app.route("/result")
def result():
    job = _get_job(session.get("last_token", ""))
    payload = job.get("payload") if job else None
    if not payload: return render_template("error.html", message="No result available yet.")
    return render_template("result.html", data=payload, post_id=job.get("post_id", ""))

# PDF report: generates an extended LLM analysis, parses the 4 expected
# sections from the text, renders the pdf.html template and converts it
# with WeasyPrint
@app.route("/pdf/<post_id>")
def pdf(post_id):
    job = next((j for j in JOBS.values() if j.get("post_id") == post_id and j.get("done")), None)
    if not job or not job.get("payload"):
        return render_template("error.html", message="Post data expired from memory.")

    from weasyprint import HTML
    payload = job["payload"]

    extended_raw = llm.extended_analysis(payload["post"], payload["claims"], payload["verdict"])

    key_map = {
        "deeper context": "1. Deeper context of the topic",
        "context of the topic": "1. Deeper context of the topic",
        "additional evidence": "2. Additional evidence discussion",
        "evidence discussion": "2. Additional evidence discussion",
        "related background": "3. Related background",
        "background": "3. Related background",
        "final nuanced": "4. Final nuanced conclusion",
        "nuanced conclusion": "4. Final nuanced conclusion",
        "final conclusion": "4. Final nuanced conclusion"
    }

    sections = {key: "" for key in key_map.values()}
    current_section = None

    for line in extended_raw.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        line_lower = line_stripped.lower()
        matched_key = None

        for keyword, canonical_key in key_map.items():
            if keyword in line_lower and len(line_stripped) < 80:
                matched_key = canonical_key
                break

        if matched_key:
            current_section = matched_key
            if ":" in line_stripped:
                content = line_stripped.split(":", 1)[1].strip()
                if len(content) > 10:
                    sections[current_section] += content + " "
        elif current_section:
            sections[current_section] += line_stripped + " "

    extended_html = ""
    for section_key, content in sections.items():
        title = section_key.split(". ", 1)[1] if ". " in section_key else section_key
        if content.strip():
            extended_html += f"<h3>{title}</h3><p>{content.strip()}</p>"
        else:
            extended_html = f"<p>{extended_raw}</p>"
            break

    html = render_template(
        "pdf.html",
        data=payload,
        extended=extended_html,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    )

    # Output filename derived from the post (author + first words + id)
    post_text = payload["post"]["text"]
    handle = payload["post"]["handle"]

    words = post_text.split()[:4]
    snippet = " ".join(words)
    snippet = re.sub(r'[^a-zA-Z0-9áéíóúñÁÉÍÓÚÑ\s]', '', snippet)
    slug = "-".join(snippet.split()).lower()[:40]
    handle_clean = re.sub(r'[^a-zA-Z0-9_]', '', handle).lower()

    filename = f"anti-x-fakes_{slug}_{handle_clean}_{post_id}.pdf"

    return Response(
        HTML(string=html).write_pdf(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)
