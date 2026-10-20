# evaldrift

Paraphrase-drift scoring for LLM eval sets. Given two texts, evaldrift
returns three plain similarity measures and one blended drift score:

- word-shingle Jaccard
- character-trigram Jaccard
- normalized Levenshtein similarity
- drift = 1 - weighted blend (weights 0.40 / 0.35 / 0.25)

No model calls, no network access, no dependencies: pure Python standard
library, single file, MIT licensed.

## Why

Paraphrase-heavy eval sets drift over time. Answers get rewritten,
questions get reworded, and nobody re-runs the comparison that justified
keeping both versions. evaldrift is a quick, deterministic way to score
those pairs before a human ever looks at them.

## Usage

Score one pair of text files:

    python3 evaldrift.py answer_v1.txt answer_v2.txt

Batch mode over a JSONL file where each line is `{"a": "...", "b": "..."}`:

    python3 evaldrift.py --pairs set.jsonl

Machine-readable output with `--json`. Word-shingle width with
`--shingle N` (default 2).

## Output

    word-shingle jaccard : 0.612
    char-trigram jaccard : 0.588
    normalized levenshtein: 0.704
    drift                 : 0.366

## License

MIT
