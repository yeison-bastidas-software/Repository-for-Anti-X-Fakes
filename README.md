# Anti X-Fakes

Web application for automated fact-checking of posts on X (Twitter). The user pastes a link to any public post and the system extracts its verifiable claims, cross-references them against published fact-checks and global news sources using retrieval-augmented generation (RAG), and returns an evidence-grounded verdict (TRUE, FALSE, MISLEADING, UNVERIFIABLE or OPINION) with cited sources and a downloadable PDF report.

## Purpose

Misinformation spreads quickly on social media, and manually verifying every claim is slow and requires access to specialized fact-checking tools. Anti X-Fakes automates that workflow end to end: from reading the post, to decomposing it into atomic factual claims, searching real external evidence, and producing an explainable verdict where every judgment is backed by the retrieved sources rather than by the language model alone.

## Features

- URL validation and automatic extraction of the X/Twitter post ID (supports twitter.com and x.com, including mobile and www variants).
- Two-level post retrieval with provider fallback (fxtwitter API first, Twitter oEmbed as backup).
- Claim extraction limited to atomic verifiable facts; opinion-only posts are detected and labeled without further searching.
- Automatic translation of claims to English to maximize search coverage over non-English posts.
- Parallel evidence gathering from Google Fact Check Tools and Tavily Search.
- Single-call batch judging of all claims to minimize LLM latency and token cost.
- Real-time progress screen driven by genuine backend stage status (no fake timers).
- Results page with overall verdict, average confidence, per-claim breakdown, reasoning, and deduplicated source list.
- Extended analysis report in PDF format (WeasyPrint) with academic layout, four-section deep analysis, and formal references.
- Friendly error mapping: technical failures (quota exhausted, invalid URLs, missing keys) are translated into clear user-facing messages.
- Built-in retry policy with exponential backoff for transient LLM errors and explicit handling of daily quota exhaustion.

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Backend framework | Flask 3.x | Routing, sessions, template rendering |
| Language model | Google Gemini (google-genai SDK) | Claim extraction, translation, evidence judging, extended analysis |
| Fact-checking API | Google Fact Check Tools API | Professional fact-check reviews and ratings |
| Web search API | Tavily | General web evidence to complement fact-checks |
| Post retrieval | fxtwitter API / Twitter oEmbed | Fetching post text, author and date without official X API access |
| PDF generation | WeasyPrint | HTML-to-PDF report conversion |
| Configuration | python-dotenv | API keys loaded from a local .env file |
| Frontend | Bootstrap 5 (CDN), custom CSS, vanilla JavaScript | Dark themed responsive UI, clipboard paste, progress polling |
| Concurrency | Python threading + ThreadPoolExecutor | Background verification pipeline and parallel evidence searches |
| Persistence | In-memory job store (dict + lock) | No database required; results live for the app process lifetime |

## Architecture

```
app.py     Flask entry point: routes, background pipeline, job state, PDF endpoint
core.py    Non-LLM stage: URL parsing, post fetching, evidence search (Fact Check + Tavily in parallel)
llm.py     Gemini stage: claim extraction, batch translation, claim judging, verdict aggregation,
           extended analysis writing, retry/quota handling
templates/ Jinja2 screens: home, processing, result, error, shared base, print-ready pdf.html
static/    style.css: dark palette and component styling
```

Request flow:

1. `POST /check` starts a daemon thread running the verification pipeline and redirects to `/processing`.
2. The pipeline calls `core.extract_post_id` and `core.fetch_post` to locate the post content.
3. `llm.extract_claims` decomposes the text into up to 3 atomic claims; if none exist the verdict is OPINION.
4. `llm.batch_translate_claims` translates claims to English in one call.
5. `core.gather_evidence` queries Google Fact Check Tools and Tavily concurrently for each claim.
6. `llm.judge_all_claims` evaluates every claim against its evidence in a single LLM call.
7. Verdict, confidence, reasoning and deduplicated sources are stored in the job payload.
8. The browser polls `/progress` until completion and then renders `/result`.
9. `/pdf/<post_id>` optionally generates the extended analysis (`llm.extended_analysis`) and renders the PDF report.

## Requirements

- Python 3.10 or newer
- A Google Cloud project with the Fact Check Tools API enabled (free key)
- A Gemini API key (Google AI Studio)
- A Tavily API key
- System libraries required by WeasyPrint (Pango, Cairo, GDK-PixBuf); on Debian/Ubuntu:
  `sudo apt install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0`

## Installation

```bash
git clone https://github.com/yeison-bastidas-software/Repository-for-Anti-X-Fakes.git
cd Repository-for-Anti-X-Fakes

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # then edit .env with your real keys
```

## Configuration

Create a `.env` file at the project root (see `.env.example`):

| Variable | Provider | Required | Purpose |
|---|---|---|---|
| `GEMINI_API_KEY` | Google AI Studio | Yes | Drives all LLM stages (extraction, translation, judging, PDF analysis) |
| `CLAIMSEARCH_API_KEY` | Google Cloud | Recommended | Enables professional fact-check lookups; without it only Tavily evidence is used |
| `TAVILY_API_KEY` | Tavily | Recommended | Enables general web evidence search; without it only fact-checks are used |

The application runs even with optional keys missing, but the quality and quantity of retrieved evidence decreases accordingly.

## Usage

```bash
python app.py
```

Open http://localhost:5000, paste a public X post link (or use the Paste button), and click Verify. The progress screen shows each real stage of the investigation; when it finishes you land on the verdict page, where you can review the per-claim analysis and sources, or download the full PDF report.

Typical run time is between 15 and 60 seconds depending on network latency and LLM response times.

## Project Structure

```
.
├── app.py              # Flask routes, background pipeline orchestration, PDF endpoint
├── core.py             # Post retrieval and parallel evidence search (no LLM involved)
├── llm.py              # Gemini prompts, parsing, retries and verdict aggregation
├── templates/
│   ├── base.html       # Shared skeleton: navbar, footer disclaimer
│   ├── home.html       # URL input, clipboard paste, verify form
│   ├── processing.html # Live progress bar fed by /progress polling
│   ├── result.html     # Verdict card, per-claim breakdown, sources
│   ├── error.html      # Generic failure screen
│   └── pdf.html        # Print-ready academic report layout
├── static/style.css    # Dark theme palette and components
├── requirements.txt
└── .env.example
```

## Design Decisions and Limitations

- **No database.** Job state lives in memory behind a thread lock, which keeps deployment trivial. Results disappear when the server restarts; persisting them (e.g., Redis or SQLite) is a natural next step.
- **Maximum 3 claims per post.** Empirically the sweet spot between analytical depth, token cost and end-to-end latency.
- **Batched LLM calls.** Translation and judging of all claims happen in single requests instead of per-claim loops, cutting both cost and wall-clock time.
- **Quota-aware.** Gemini 429 responses surface immediately as a friendly "try again tomorrow" message instead of burning retries; transient 503 errors retry with exponential backoff.
- **Verdict transparency.** The judge prompt forces the model to reason strictly over retrieved evidence and cite source URLs, and the UI always displays those sources next to each claim.
- **Disclaimer everywhere.** Every screen and the PDF carry the notice that verdicts are AI-assisted, not absolute truth; human verification is recommended for high-stakes decisions.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
