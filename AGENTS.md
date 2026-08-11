# Project Working Rules

- Treat the current code and tests as the source of truth. When documentation conflicts with them, follow the implementation and call out the discrepancy.
- Work only within the requested Stage and Step; do not pre-implement later work.
- Read the relevant real code and tests before editing.
- Unless explicitly requested, do not call real APIs and do not commit or push.
- Read API keys only from environment variables. Never hard-code, persist, log, or expose them.
- Never place original image bytes, Base64, or Data URLs in Generation Jobs, Context, or Trace data.
- Build exactly one `ContextBundle` per Generation execution. Derive deterministic Analyzer, Tool, and Final projections from that same snapshot; consumers do not all need the full Bundle.
- The current user query is the only task for the current turn. Historical context is reference material for disambiguation and condition recovery, not a queue of tasks.
- Preserve continuous coverage with Summary + unsummarized Bridge + Recent history; do not create context gaps.
- Implement deterministic engineering rules in code, not in Analyzer prompts or model judgment.
- Run relevant targeted tests, then one full regression. Avoid repetitive test loops without new evidence.
- Report debugging as: error → cause → minimal fix.
