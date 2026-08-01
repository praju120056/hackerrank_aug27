# HackerRank Orchestrate — Message Notification Router

AI-powered notification routing engine built for WhatsApp. It determines whether to `notify` the user immediately, group into a `digest` for later, or `mute` low-value/suspicious alerts based on multimodal inputs (text, images, and voice notes) and rich, relational behavioral statistics.

---

## How It Works (Pipeline Architecture)

The system enforces a clean, deterministic-first execution path to guarantee reliability, optimize Gemini API rate limit usage, and scale under strict free-tier quotas:

```
Incoming Message
      │
      ▼
Media understanding ───────→ Deduplicated pre-pass processes every unique image
(Gemini 3.5)                 or audio exactly once and caches in-memory.
      │
      ▼
Context Building ──────────→ Merges all 12 dataset CSVs into a structured context
                             object. No raw CSV rows are ever sent to LLM.
      │
      ▼
Rule Engine
├── Absolute Rules ────────→ Deterministically handles prompt injections, domain mismatches,
│                            and OTP scams. Skips the LLM entirely on a match.
└── Preference Signals ────→ Stackable biases (DND, muted group, opt-outs, fatigue ratio)
                             inject context enrichment biases (clamped to [-0.5, +0.5]).
      │
      ▼
Evidence Retrieval ────────→ Tier-ranked retrieval (sender > group > business) with recency
                             and engagement scoring to load the top-3 historical interactions.
      │
      ▼
Decision Engine ───────────→ Batches remaining items (10 per call) to Gemini 3.5 Flash Lite
(Gemini 3.5)                 using strict JSON output. Supports checkpoint save/resume.
      │
      ▼
Output Validation & Calib. → Validates output structure, repairs malformed fields,
                             applies preference biases to confidence, and clamps to [0.55, 0.95].
      │
      ▼
output.csv
```

---

## Repository Layout

```text
.
├── AGENTS.md                         # Rules for AI coding tools + transcript logging
├── problem_statement.md              # Full challenge statement
├── README.md                         # You are here
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for secret config
├── output.csv                        # Final predictions (written by main.py)
├── code/
│   ├── main.py                       # Pipeline Orchestrator (entry point)
│   ├── models.py                     # Shared dataclasses (MediaSummary, RoutingPrediction)
│   ├── context_builder.py            # Aggregates users/groups/history CSV records
│   ├── rule_engine.py                # Enforces absolute bypass rules + stackable preference biases
│   ├── evidence.py                   # Rank-retrieves top-3 historical messages
│   ├── media_processor.py            # Gemini 3.5 multimodal analysis & semantic summaries
│   ├── llm_router.py                 # Batched routing decision driver + checkpointing
│   ├── evaluation.py                 # Validation suite comparing sample_messages.csv to labels
│   └── prompts/
│       └── few_shot_examples.txt     # Labelled JSONL few-shot cases
├── tests/
│   ├── test_output_schema.py         # Validates schema structure of final output.csv
│   ├── test_context_builder.py       # Validates context dict construction & DND checks
│   ├── test_rule_engine.py           # Evaluates stackable signals and absolute rules
│   ├── test_evidence.py              # Evaluates tier-based historical retrieval
│   └── test_llm_parser.py            # Verifies JSON cleaning, repairing, and retrying
└── dataset/
    ├── messages.csv                  # Incoming messages to route
    └── media/                        # Multimodal raw images and audio files
```

---

## Setup and Quick Start

### 1. Install Dependencies
Run under **Python 3.10+**:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and fill in your Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
ROUTING_MODEL=gemini-3.5-flash-lite
MEDIA_MODEL=gemini-3.5-flash-lite
```

### 3. Run Pipeline Inference
```bash
python code/main.py
```
This executes the pipeline end-to-end for all 110 messages in `dataset/messages.csv` and outputs `output.csv`. It supports **checkpoint recovery**: if interrupted, re-running will resume processing from the last successfully written batch.

---

## Advanced Verification and Debug Modes

### Run Unit Tests (Offline / No Key Needed)
Our 50-test suite runs locally without hitting the Gemini API:
```bash
python tests/test_rule_engine.py
python tests/test_context_builder.py
python tests/test_evidence.py
python tests/test_llm_parser.py
```

### Run Evaluation Script (Requires API Key)
Executes inference on the solved `dataset/sample_messages.csv` (stripping answers during inference) and reports accuracy, confusion matrices, and detailed prediction mismatches:
```bash
python code/evaluation.py
```

### Enable Media Debug Mode (`DEBUG_MEDIA=true`)
To audit what semantic summaries Gemini is generating for your images and voice notes, set `DEBUG_MEDIA=true` in your environment before running `main.py` or `evaluation.py`:
```bash
# Windows PowerShell
$env:DEBUG_MEDIA="true"
python code/main.py

# Windows Command Prompt
set DEBUG_MEDIA=true
python code/main.py

# Linux/macOS
DEBUG_MEDIA=true python code/main.py
```
This processes all unique media and writes a clean JSON audit log named `media_debug.json` containing:
```json
{
  "media_id": "vn_008",
  "media_type": "voice",
  "summary": "A notification regarding a bank account block and requesting an OTP for immediate verification.",
  "category": "urgent",
  "urgency": "high"
}
```

---

## Submission Checklist

Ensure you include the following three files when submitting your solution:
1. **Code zip** (`code.zip` containing `code/`, `tests/`, `requirements.txt`, `README.md`, etc.).
2. **Final output** (`output.csv` written in the repository root).
3. **Chat transcript** (`log.txt` generated automatically by your coding agent in `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt`).
