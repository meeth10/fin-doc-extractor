# Qwen financial analysis stack

The local stack is intentionally small, sequential, and evidence-first for a 24 GB RAM workstation.

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
| Embedding | `qwen3-embedding:0.6b` | Cheap semantic retrieval with low memory use |
| Planner | `qwen3:4b` | Structured query interpretation without paying 8B/14B cost |
| Analyst | `qwen3:8b` | Main local reasoning model; thinking enabled for analytical questions |
| Verifier | `qwen3:4b` | Independent audit of period, scope, arithmetic and source support |

The application invokes these models sequentially. `OLLAMA_KEEP_ALIVE=0` prevents intentionally retaining every model in memory.

## Reliability rules

1. Source facts outrank model prose.
2. Period and column identity remain attached to every numeric fact.
3. Reported values are never silently replaced by reconstructed values.
4. Arithmetic is performed in Python.
5. Rejected answers receive one correction pass and then a second verification pass.
6. Unsupported answers are returned as provisional rather than disguised as high-confidence facts.

The Ollama client uses JSON-schema-constrained output for planner, analyst and verifier responses when supported by the installed Ollama version.

## Validation status

The final regression pass reached 69 passing tests with one remaining compatibility assertion; that assertion was corrected in the final debt-status fix on the rebuild branch.

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
