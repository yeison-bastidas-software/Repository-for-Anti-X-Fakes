"""Anti X-Fakes — Flask application entry point.

Serves the pages and routes defined in the project plan (and nothing else):
  GET  /            homepage with the URL input form and the Verify button
  POST /check       starts the RAG verification pipeline
  GET  /processing  processing screen (progress bar, ETA countdown, stage text)
  GET  /progress    JSON endpoint polled by the processing screen (real status)
  GET  /result      verdict card page (reads the cached result from SQLite)
  GET  /pdf/<id>    extended PDF report (Gemini + Jinja + WeasyPrint)

The heavy work of /check runs in a background thread so the browser can poll
/progress and show genuine per-stage progress instead of a fake animation.
"""

import os
import threading
import time
from datetime import datetime, timezone

# Load .env BEFORE importing modules that read environment variables.
from dotenv import load_dotenv
load_dotenv()

from flask import (
    Flask, render_template, request, redirect, url_for, session, Response, jsonify,
)
# Imported here (after load_dotenv) so both modules see the loaded keys.
import core   # URL validation, SQLite cache, external API clients
import llm    # Gemini: claim extraction, judging, extended PDF analysis

app = Flask(__name__)
# Local-only secret: the session just carries the id of the last job so the
# result page knows which verification to display. No login system exists.
app.secret_key = "anti-x-fakes"

# ---------------------------------------------------------------------------
# In-memory job registry: lets the Processing screen poll real progress.
# Keyed by a short-lived random token generated when a check is submitted.
# Value shape: {stage, total, status, done, error, post_id, started_at}
# ---------------------------------------------------------------------------
JOBS = {}
_JOBS_LOCK = threading.Lock()  # the pipeline thread writes, requests read


def _new_job(token):
    """Create a fresh job record for a just-submitted verification."""
    with _JOBS_LOCK:
        JOBS[token] = {
            "stage": 0, "total": 5, "status": "Starting investigation…",
            "done": False, "error": None, "post_id": None,
            "started_at": time.time(),
        }


def _update_job(token, **changes):
    """Patch fields of an existing job record (called from the worker thread)."""
    with _JOBS_LOCK:
        job = JOBS.get(token)
        if job:
            job.update(changes)


def _get_job(token):
    """Read a consistent snapshot of a job; empty dict if unknown/expired."""
    with _JOBS_LOCK:
        job = JOBS.get(token)
        return dict(job) if job else {}


# ---------------------------------------------------------------------------
# Verification pipeline (internal flow steps 2-8 of the plan)
# ---------------------------------------------------------------------------

def run_pipeline(token, post_url):
    """Execute one full verification and store the outcome in the SQLite cache.

    Progress convention: stages 1-3 are fixed (validate, fetch, extract);
    once the claim count N is known, each claim gets two sub-stages
    (evidence search + judging), so total = 3 + 2*N, then save = total.
    """
    total = 5  # final stage number; recomputed below once claims are known
    try:
        # Step 2: validate the URL and extract the numeric post id.
        _update_job(token, stage=1, status="Validating URL and locating the post…")
        post_id = core.extract_post_id(post_url)
        if not post_id:
            _update_job(token, done=True, error=(
                "Invalid URL. Paste a link to an X post "
                "(e.g. https://x.com/user/status/1234567890)."
            ))
            return

        # Step 3: fetch content — FxTwitter first, one retry via X oEmbed.
        _update_job(token, stage=2,
                    status="Fetching post content (FxTwitter, fallback: oEmbed)…")
        post = core.fetch_post(post_id, post_url)
        if not post:
            _update_job(token, done=True, error="Post not found or unavailable")
            return

        # Step 4: split the post into atomic verifiable claims with Gemini.
        _update_job(token, stage=3, status="Extracting verifiable claims with the LLM…")
        claims = llm.extract_claims(post["text"])

        if claims:
            total = 3 + 2 * len(claims)
            _update_job(token, total=total)
            results = []
            for i, claim in enumerate(claims, 1):
                # Step 5: evidence search (ClaimSearch + GDELT + Wikipedia, top 3 each).
                _update_job(token, stage=3 + 2 * (i - 1), status=(
                    f"Searching evidence for claim {i}/{len(claims)}: “{claim[:60]}”"
                ))
                evidence = core.gather_evidence(claim)

                # Step 6: Gemini judges the claim against its retrieved evidence.
                _update_job(token, stage=3 + 2 * (i - 1) + 1, status=(
                    f"Judging claim {i}/{len(claims)} against the retrieved evidence…"
                ))
                results.append(llm.judge_claim(claim, evidence))

            # Step 7: aggregate per-claim labels into one post-level verdict.
            verdict = llm.aggregate_verdict(results)
            confidence = round(sum(r["confidence"] for r in results) / len(results))
            reasoning = " ".join(f"{r['claim']} → {r['reasoning']}" for r in results)
        else:
            # Opinion-only post: verdict OPINION, jump straight to saving (step 8).
            verdict, confidence, results = "OPINION", 100, []
            reasoning = ("The Gemini extractor found no verifiable factual claims "
                         "in this post; it contains opinions only.")

        # Every source cited anywhere is collected (deduplicated by URL) so the
        # result page can credit all evidence used to form the verdict.
        sources, seen = [], set()
        for r in results:
            for e in r["evidence"]:
                if e["url"] and e["url"] not in seen:
                    seen.add(e["url"])
                    sources.append(e)

        elapsed = round(time.time() - _get_job(token)["started_at"], 1)

        payload = {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "sources": sources,
            "claims": results,
            "post": post,
            "elapsed_seconds": elapsed,
        }

        # Step 8: persist to the SQLite cache keyed by post id, then finish.
        _update_job(token, stage=total - 1, status="Saving verdict to cache and rendering result…")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        core.save_result(post_id, post_url, now, payload)
        _update_job(token, done=True, post_id=post_id)

    except RuntimeError as exc:
        # Missing/misconfigured API key (raised by the Gemini client).
        _update_job(token, done=True, error=f"Configuration error: {exc}. Check your .env keys.")
    except Exception as exc:  # any network/parse failure ends the job cleanly
        _update_job(token, done=True, error=f"Verification failed: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    """Homepage: program name, description, URL field, Paste and Verify buttons."""
    return render_template("home.html")


@app.route("/check", methods=["POST"])
def check():
    """Receive the submitted URL and launch the pipeline in a worker thread."""
    post_url = (request.form.get("url") or "").strip()
    token = os.urandom(8).hex()          # opaque id for this verification run
    _new_job(token)

    threading.Thread(target=run_pipeline, args=(token, post_url), daemon=True).start()

    session["last_token"] = token        # remembered so /result can find the job
    return redirect(url_for("processing", token=token))


@app.route("/processing")
def processing():
    """Processing screen: progress bar, ETA countdown, current-stage text."""
    return render_template("processing.html", token=request.args.get("token", ""))


@app.route("/progress")
def progress():
    """Polling endpoint: percent (0-100), stage text, done flag, error, post id."""
    job = _get_job(request.args.get("token", ""))
    if not job:
        return jsonify({"percent": 0, "status": "", "done": True,
                        "error": "This verification is no longer active. Please start again.",
                        "post_id": None})

    stage, total = job["stage"], max(1, job["total"])
    if job["done"] and not job["error"]:
        percent = 100                     # success always lands on 100%
    else:
        percent = min(99, round(stage / total * 100))  # cap below 100 while running
    return jsonify({"percent": percent, "status": job["status"],
                    "done": job["done"], "error": job["error"], "post_id": job["post_id"]})


@app.route("/result")
def result():
    """Verdict card: conclusion, confidence, reasoning, sources, research time."""
    post_id = (_get_job(session.get("last_token", "")).get("post_id")
               or request.args.get("post_id", ""))
    data = core.load_result(post_id) if post_id else None
    if not data:
        return render_template("error.html",
                               message="No result available yet. Please verify a post first.")
    return render_template("result.html", data=data, post_id=post_id)


@app.route("/pdf/<post_id>")
def pdf(post_id):
    """Extended PDF report: cached verdict + Gemini extended analysis + WeasyPrint."""
    data = core.load_result(post_id)
    if not data:
        return render_template("error.html", message="Post not found or unavailable")

    # Ask Gemini for a deeper write-up, inject it into the print template,
    # then convert the rendered HTML+CSS into a PDF returned as a download.
    from weasyprint import HTML  # heavy import deferred: only needed here
    extended = llm.extended_analysis(data["post"], data["claims"], data["verdict"])
    html = render_template(
        "pdf.html", data=data, extended=extended,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    )
    pdf_bytes = HTML(string=html).write_pdf()
    return Response(
        pdf_bytes, mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="anti-x-fakes-{post_id}.pdf"'},
    )


if __name__ == "__main__":
    # Local development server (the plan targets running on your own machine).
    app.run(debug=True, port=5000)
