#!/usr/bin/env python3
"""evaldrift - paraphrase-drift scoring for LLM eval sets.

Scores how far two texts have drifted from each other using three
complementary measures: word-shingle Jaccard, character-trigram Jaccard,
and normalized Levenshtein distance. Pure stdlib.

Usage:
  python3 evaldrift.py TEXT_A.txt TEXT_B.txt [--shingle N] [--json]
  python3 evaldrift.py --pairs set.jsonl [--json]     # lines of {"a":..,"b":..}

Exit codes: 0 ok, 2 bad arguments, 3 unreadable input.
"""
import argparse
import json
import sys


def words(text):
    return text.lower().split()


def shingles(tokens, n):
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def levenshtein(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_lev_sim(a, b):
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / m


def score(text_a, text_b, shingle_n=2):
    wa, wb = words(text_a), words(text_b)
    ws = jaccard(shingles(wa, shingle_n), shingles(wb, shingle_n))
    cs = jaccard(shingles(list(text_a.lower()), 3), shingles(list(text_b.lower()), 3))
    ls = norm_lev_sim(text_a, text_b)
    similarity = 0.4 * ws + 0.35 * cs + 0.25 * ls
    return {
        "word_shingle_jaccard": round(ws, 4),
        "char_trigram_jaccard": round(cs, 4),
        "normalized_levenshtein_sim": round(ls, 4),
        "drift": round(1.0 - similarity, 4),
    }


def read(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError as exc:
        print("evaldrift: cannot read %s: %s" % (path, exc), file=sys.stderr)
        sys.exit(3)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evaldrift",
                                 description="paraphrase-drift scoring for LLM eval sets")
    ap.add_argument("file_a", nargs="?")
    ap.add_argument("file_b", nargs="?")
    ap.add_argument("--pairs", metavar="JSONL")
    ap.add_argument("--shingle", type=int, default=2, metavar="N")
    ap.add_argument("--json", action="store_true", dest="as_json")
    args = ap.parse_args(argv)

    if args.pairs:
        results = []
        try:
            with open(args.pairs, "r", encoding="utf-8", errors="replace") as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        print("evaldrift: %s line %d is not JSON" % (args.pairs, lineno),
                              file=sys.stderr)
                        sys.exit(3)
                    results.append({"line": lineno,
                                    **score(row.get("a", ""), row.get("b", ""), args.shingle)})
        except OSError as exc:
            print("evaldrift: cannot read %s: %s" % (args.pairs, exc), file=sys.stderr)
            sys.exit(3)
        if args.as_json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print("line %-4d drift %.3f  (word %.3f  char %.3f  lev %.3f)" % (
                    r["line"], r["drift"], r["word_shingle_jaccard"],
                    r["char_trigram_jaccard"], r["normalized_levenshtein_sim"]))
        return 0

    if not args.file_a or not args.file_b:
        ap.error("need two text files, or --pairs SET.jsonl")
    a, b = read(args.file_a), read(args.file_b)
    r = score(a, b, args.shingle)
    if args.as_json:
        print(json.dumps(r, indent=2))
    else:
        print("word-shingle jaccard : %.3f" % r["word_shingle_jaccard"])
        print("char-trigram jaccard : %.3f" % r["char_trigram_jaccard"])
        print("normalized levenshtein: %.3f" % r["normalized_levenshtein_sim"])
        print("drift                 : %.3f" % r["drift"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
