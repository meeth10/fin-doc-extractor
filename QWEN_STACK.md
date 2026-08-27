# Qwen financial analysis stack

```text
PDF
  -> existing deterministic extractor
  -> financial fact normalization
  -> deterministic fact retrieval
  -> Qwen3-Embedding 0.6B (only for narrative retrieval)
  -> Qwen3 8B document analyst (only when semantic retrieval is needed)
  -> Python financial calculations
  -> Qwen3 14B financial analyst
  -> Qwen3 8B controller/critic
```

Simple financial fact questions use a deterministic fast path. Embeddings are cached and semantic models are only invoked for interpretation-heavy questions.

Models:
- qwen3-embedding:0.6b — narrative retrieval infrastructure
- qwen3:8b — document/evidence analyst and controller
- qwen3:14b — financial reasoning analyst
- qwen3-vl:4b — optional later fallback for visually difficult pages

For a 24 GB machine, keep models sequential and use:

```bash
export OLLAMA_KEEP_ALIVE=0
export OLLAMA_CONTEXT=12288
```

Install:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:8b
ollama pull qwen3:14b
```

Run:

```bash
python -m agent.qwen_cli path/to/report.pdf \
  --question "What was total debt?" \
  --out output
```

The extractor remains model-independent. `extractor/financial_facts.py` is only a compatibility import; the implementation lives in `agent/financial_facts.py`.
