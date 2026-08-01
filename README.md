# HackerRank Orchestrate

Starter repository for the **HackerRank Orchestrate** 24-hour hackathon.

## Message Notification Router

Build an AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

The system must reason over multimodal messages, including text messages, image posters/screenshots, and voice notes.

WhatsApp is noisy. A user can receive family chats, society notices, school updates, co-worker messages, business account promotions, image posters, voice notes, and scams in the same message stream. Treating every message the same creates two bad outcomes: important messages get missed, and unwanted or risky messages interrupt the user.

Read [`problem_statement.md`](./problem_statement.md) for the full task spec, input/output schema, allowed values, and submission format.

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
│   ├── main.py                       # Entry point — orchestrates the full pipeline
│   ├── context_builder.py            # Loads all CSVs, builds per-message context dict
│   ├── rule_engine.py                # Absolute rules + preference signals
│   ├── evidence.py                   # Key-match retrieval from message_history
│   ├── media_processor.py            # Gemini inline image/audio description
│   ├── llm_router.py                 # Gemini Flash batching, prompt, JSON parse
│   └── prompts/
│       └── few_shot_examples.txt     # 30 labelled sample rows for LLM context
├── tests/
│   ├── test_output_schema.py         # Validate output.csv schema and row count
│   ├── test_context_builder.py       # Validate context dict construction
│   ├── test_rule_engine.py           # Unit tests for all rules and signals
│   ├── test_evidence.py              # Evidence retrieval tests
│   └── test_llm_parser.py            # LLM JSON parsing + confidence clamping
└── dataset/
    ├── messages.csv                  # Messages to route
    ├── output.csv                    # Blank submission template
    ├── sample_messages.csv           # Solved examples
    ├── users.csv                     # User notification behavior
    ├── groups.csv                    # Group metadata
    ├── group_members.csv             # User-group relationships
    ├── business_accounts.csv         # Business sender metadata
    ├── user_business_history.csv     # User-business history
    ├── message_history.csv           # Historical messages
    ├── message_events.csv            # User reactions to historical messages
    ├── images.csv                    # Image IDs and media file paths
    ├── voice_notes.csv               # Voice note IDs and media file paths
    ├── daily_notification_summary.csv
    └── media/
        ├── images/
        └── audio/
```

---

## Setup and Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your Gemini API key

```bash
cp .env.example .env
# Open .env and replace "your_key_here" with your real GEMINI_API_KEY
```

### 3. Run the pipeline

```bash
python code/main.py
```

`output.csv` will be written to the repo root with exactly 110 rows (one per message).

---

## Run Tests (no API key needed for tests 2-5)

```bash
python tests/test_rule_engine.py      # Rule engine unit tests
python tests/test_context_builder.py  # Context builder validation
python tests/test_evidence.py         # Evidence retrieval tests
python tests/test_llm_parser.py       # LLM parser + confidence clamping
python tests/test_output_schema.py    # Output CSV schema validation (run after main.py)
```
5. Evaluate your approach on the solved sample rows before submitting.

You may use any language or runtime. Python, JavaScript, and TypeScript are all reasonable choices.

---

## Requirements

Your solution must:

- be runnable from the terminal
- read the provided files from `dataset/`
- produce a valid `output.csv`
- include one prediction for every `message_id` in `dataset/messages.csv`
- not use organizer-only files or hardcoded labels

If you use API keys or secrets, read them from environment variables. Never hardcode secrets in the repo.

---

## Evaluation

Your `output.csv` will be compared against hidden ground-truth labels.

The scoring will consider:

- correctness of `action`
- correctness of `message_type`
- usefulness and consistency of `reason`
- whether `evidence_message_ids` point to relevant historical messages
- reasonable confidence calibration

Strong systems will combine retrieval, structured metadata, behavioral history, safety checks, OCR/ASR handling, and contextual reasoning.

---

## Chat Transcript Logging

This repo includes an [`AGENTS.md`](./AGENTS.md) file for AI coding tools. It asks compatible tools to append conversation summaries to:

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

Upload this log as your chat transcript at submission time. Do not paste secrets into the chat.

---

## Submission

Submit the following files as instructed by HackerRank:

1. **Code zip**: full runnable solution, prompts/configs, README, and any evaluation files.
2. **Predictions CSV**: final `output.csv` for all rows in `dataset/messages.csv`.
3. **Chat transcript**: the `log.txt` described above.

Before submitting, confirm:

- `output.csv` has one row per row in `dataset/messages.csv`.
- `output.csv` has the exact required columns in the exact required order.
- Your runnable code and setup instructions are included in `code.zip`.
