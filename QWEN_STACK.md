# Qwen financial analysis stack

The local stack is intentionally small, sequential, and evidence-first for a 24 GB RAM workstation:

```text
PDF / extracted tables + narrative
        |
        v
Financial fact graph
  (period + column + scope + provenance)
        |
        +--> Qwen3-Embedding 0.6B
        |      hybrid semantic retrieval
        |
        +--> Qwen3 4B Planner
        |      question -> accounting plan
        |
        v
Deterministic accounting engine
  reported facts / aligned comparisons / formulas
        |
        +--> Qwen3 8B Analyst
        |      only for ambiguity or narrative reasoning
        |
        +--> Qwen3 4B Verifier
               independent audit of answer + evidence
```

## Model choice

| Role | Model | Why |
|---|---|---|
| Embedding | `qwen3-embedding:0.6b` | Cheap document/fact retrieval; small memory footprint |
| Planner | `qwen3:4b` | Strong enough for structured intent extraction without wasting RAM |
| Analyst | `qwen3:8b` | Main financial reasoning model; used selectively with thinking enabled |
| Verifier | `qwen3:4b` | Independent second pass for period, scope, arithmetic and source checks |

The models are invoked sequentially and `OLLAMA_KEEP_ALIVE=0` is the default, so the application does not intentionally keep all model weights resident. Ollama currently lists Qwen3 4B at about 2.5 GB, Qwen3 8B at about 5.2 GB, and Qwen3 Embedding 0.6B at about 639 MB. citeturn536755search0turn536755search2

## Reliability rules

1. Source facts outrank model prose.
2. Period and column identity stay attached to every numeric fact.
3. Reported values are never silently replaced by reconstructed values.
4. Arithmetic is performed in Python, not by the LLM.
5. The verifier can force one correction pass and then re-audit the corrected answer.
6. Unsupported answers are returned as provisional rather than dressed up as high-confidence facts.

Ollama's chat API supports JSON-schema-constrained output via the `format` field, which the planner, analyst and verifier use. citeturn432331view0turn432331view1

## Install

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b
ollama pull qwen3:8b
```

Recommended workstation settings:

```bash
export OLLAMA_KEEP_ALIVE=0
export OLLAMA_CONTEXT=12288
export OLLAMA_MAX_OUTPUT=768
```

## Run

```bash
python -m agent.qwen_cli path/to/report.pdf \
  --question "What was total debt and how did it change year over year?" \
  --out output
```
