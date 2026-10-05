"""
Sweep: sentence segmentation of the chain of thought.

The full version needs H_BE recomputed at
the new step boundaries, and new boundaries mean new prefixes, which means resampling
continuations from the target model. That part needs generation and is out of scope here.

What can be done exactly, offline, is the structural half, and it is the half that
decides how much work the full version is. The complete un-segmented reasoning text is
persisted in `base_response.json:scratchpad`, so every trace can be re-segmented with
alternative splitters and compared against the shipped one.

The shipped segmenter is a hand-rolled character loop, not a library. It has two
behaviours worth isolating because an alternative splitter will not reproduce them:

  * the final chunk is discarded (the append is commented out), so trailing text after
    the last sentence terminator never becomes a step;
  * chunks under 10 characters are merged into a neighbour, joined with a space.

Each is given its own variant so its contribution can be read off separately.

Reported per variant: step counts, step length distributions, boundary agreement with
the shipped segmentation, how the first-50%% intervention window shifts, and what
fraction of shipped step boundaries survive, which is what fraction of the existing
H_BE values would remain valid.

Outputs: sweep_segmentation_results.json
"""

from __future__ import annotations

import re
import statistics

import common as C


def _pct(values, q):
    """Linear-interpolation percentile, matching numpy's default, on the stdlib."""
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = (len(xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))

OUTPUT = C.REPO / "sweep_segmentation_results.json"

SENTENCE_ENDERS = (".", "?", "!")
PARAGRAPH_PATTERNS = ("\n\n", "\r\n\r\n")
MERGE_THRESHOLD = 10

# Guard for the regex splitter so it does not break on common abbreviations.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
    "eg", "ie", "fig", "no", "inc", "ltd", "co", "corp", "approx", "cf", "al",
}


def preprocess(text):
    """Replica of the think-tag handling at the top of split_solution_into_chunks."""
    while "<think>" in text:
        text = text.split("<think>", 1)[1].strip()
    if "</think>" in text:
        text = text.split("</think>")[0].strip()
    return text


# ---------------------------------------------------------------------------
# segmenters: each returns a list of (start, end) spans into the preprocessed text
# ---------------------------------------------------------------------------

def seg_shipped(text, keep_tail=False, merge_short=True, return_strings=False):
    """Exact replica of src/generation/utils.py:split_solution_into_chunks."""
    spans, start, i, n = [], 0, 0, len(text)
    while i < n:
        is_para = text.startswith("\n\n", i) or text.startswith("\r\n\r\n", i)
        is_sent = (
            i < n - 1 and text[i] in SENTENCE_ENDERS and text[i + 1] in (" ", "\n")
        )
        if is_para or is_sent:
            if text[start:i + 1].strip():
                spans.append((start, i + 1))
                start = i + 1
        i += 1

    if keep_tail and text[start:].strip():
        spans.append((start, n))

    items = [[s, e, text[s:e].strip()] for s, e in spans]
    if merge_short:
        j = 0
        while j < len(items):
            if len(items[j][2]) < MERGE_THRESHOLD:
                if j == len(items) - 1:
                    if j > 0:
                        items[j - 1][1] = items[j][1]
                        items[j - 1][2] = items[j - 1][2] + " " + items[j][2]
                        items.pop(j)
                    else:
                        break
                else:
                    items[j + 1][0] = items[j][0]
                    items[j + 1][2] = items[j][2] + " " + items[j + 1][2]
                    items.pop(j)
                if j == 0 and len(items) == 1:
                    break
            else:
                j += 1
    # A merged chunk is joined with a single space by the pipeline, so its text is not
    # the raw slice of the source. Callers checking string equality need the joined form.
    if return_strings:
        return [txt for _, _, txt in items]
    return [(s, e) for s, e, _ in items]


def _spans_from_cuts(text, cuts):
    """Turn a sorted list of cut offsets into non-empty stripped spans."""
    spans, prev = [], 0
    for cut in list(cuts) + [len(text)]:
        if cut <= prev:
            continue
        if text[prev:cut].strip():
            spans.append((prev, cut))
        prev = cut
    return spans


def seg_regex_sentence(text):
    """Sentence split on terminal punctuation followed by whitespace, guarding abbreviations."""
    cuts = []
    for m in re.finditer(r"[.!?]+[\"')\]]*\s+", text):
        head = text[:m.start() + 1]
        token = re.split(r"[\s(\[\"']", head.rstrip("."))[-1].lower().rstrip(".")
        if token in ABBREVIATIONS:
            continue
        if re.fullmatch(r"[a-z]", token):  # single initial, e.g. "J."
            continue
        cuts.append(m.end())
    return _spans_from_cuts(text, cuts)


def seg_paragraph(text):
    """Coarser: split only on blank lines."""
    return _spans_from_cuts(text, [m.end() for m in re.finditer(r"\n\s*\n", text)])


def seg_newline(text):
    """Split on every line break."""
    return _spans_from_cuts(text, [m.end() for m in re.finditer(r"\n+", text)])


def seg_clause(text):
    """Finer: sentence terminators plus semicolons and colons."""
    return _spans_from_cuts(text, [m.end() for m in re.finditer(r"[.!?;:]+[\"')\]]*\s+", text)])


SEGMENTERS = {
    "shipped": lambda t: seg_shipped(t),
    "shipped_keep_tail": lambda t: seg_shipped(t, keep_tail=True),
    "shipped_no_merge": lambda t: seg_shipped(t, merge_short=False),
    "regex_sentence": seg_regex_sentence,
    "paragraph": seg_paragraph,
    "newline": seg_newline,
    "clause": seg_clause,
}


# ---------------------------------------------------------------------------

def boundary_offsets(text, spans):
    """
    Normalised boundary positions: the offset just past a chunk's last non-space
    character. Without this, two segmenters that agree on where a sentence ends still
    look disjoint because one cuts before the trailing space and the other after it.
    """
    return {s + len(text[s:e].rstrip()) for s, e in spans}


def boundary_agreement(ref_spans, var_spans, text):
    """Agreement between two segmentations, over normalised chunk end offsets."""
    a = boundary_offsets(text, ref_spans)
    b = boundary_offsets(text, var_spans)
    if not a and not b:
        return {"precision": None, "recall": None, "f1": None, "jaccard": None}
    inter = len(a & b)
    prec = inter / len(b) if b else 0.0
    rec = inter / len(a) if a else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": C._f(prec),
        "recall": C._f(rec),
        "f1": C._f(f1),
        "jaccard": C._f(inter / len(a | b)) if (a | b) else None,
    }


def main():
    all_models = {**C.MODELS, **C.EXTRA_MODELS}
    print("Loading raw traces from base_response.json:scratchpad ...")

    traces = {}
    for res_dir, proc_dir in all_models.items():
        loaded = C.load_base_traces(proc_dir)
        if loaded:
            traces[res_dir] = loaded
        print(f"  {res_dir:34s} {len(loaded):4d} traces")

    # --- gate: does the replica reproduce the shipped chunks exactly? --------
    print("\nValidating the shipped-segmenter replica against stored chunks...")
    checked = exact = 0
    mismatch_examples = []
    for res_dir in all_models:
        for scenario, blob in traces.get(res_dir, {}).items():
            stored = blob.get("chunks") or []
            if not stored:
                continue
            text = preprocess(blob["text"])
            mine = seg_shipped(text, return_strings=True)
            checked += 1
            if mine == stored:
                exact += 1
            elif len(mismatch_examples) < 3:
                mismatch_examples.append({
                    "model": res_dir, "scenario": scenario,
                    "n_replica": len(mine), "n_stored": len(stored),
                })
    gate = {
        "traces_checked": checked,
        "traces_exactly_reproduced": exact,
        "exact_fraction": C._f(exact / checked) if checked else None,
        "mismatch_examples": mismatch_examples,
    }
    print(f"  {exact}/{checked} traces reproduced exactly "
          f"({100*(exact/checked if checked else 0):.1f}%)")

    # --- structural comparison ---------------------------------------------
    print("\nSegmenting every trace with each variant...")

    # Stream: segment one trace under every variant, fold it into the accumulators, then
    # drop it. Materialising all 644 traces times 7 segmentations at once holds hundreds
    # of thousands of live span tuples for no benefit.
    n_traces = sum(1 for r in traces for s in traces[r] if preprocess(traces[r][s]["text"]))
    print(f"  {n_traces} traces to segment")

    stats = {
        name: {"counts": [], "char_lens": [], "word_lens": [], "agree": [],
               "cover": [], "halves": [], "preserved": [], "acc": {}}
        for name in SEGMENTERS
    }

    for res_dir in traces:
        for scenario, blob in traces[res_dir].items():
            text = preprocess(blob["text"])
            if not text:
                continue
            spans_by_variant = {n: fn(text) for n, fn in SEGMENTERS.items()}
            ref = spans_by_variant["shipped"]
            for name, spans in spans_by_variant.items():
                st = stats[name]
                st["counts"].append(len(spans))
                st["halves"].append(len(spans) // 2)
                for s, e in spans:
                    chunk = text[s:e].strip()
                    st["char_lens"].append(len(chunk))
                    st["word_lens"].append(len(chunk.split()))
                st["cover"].append(sum(e - s for s, e in spans) / len(text))
                ag = boundary_agreement(ref, spans, text)
                st["agree"].append(ag["f1"] if ag["f1"] is not None else 0.0)
                pres = ag["recall"] if ag["recall"] is not None else 0.0
                st["preserved"].append(pres)
                m = st["acc"].setdefault(res_dir, {"counts": [], "pres": []})
                m["counts"].append(len(spans))
                m["pres"].append(pres)
            del spans_by_variant

    per_variant = {}
    for name in SEGMENTERS:
        st = stats[name]
        counts, char_lens, word_lens = st["counts"], st["char_lens"], st["word_lens"]
        agree, cover, halves, preserved = st["agree"], st["cover"], st["halves"], st["preserved"]
        per_model = {}
        for res_dir, m in sorted(st["acc"].items()):
            per_model[res_dir] = {
                "mean_steps_per_trace": C._f(statistics.fmean(m['counts'])),
                "total_steps": int(sum(m['counts'])),
                "mean_shipped_boundaries_preserved": C._f(statistics.fmean(m['pres'])),
            }

        per_variant[name] = {
            "n_traces": len(counts),
            "total_steps": int(sum(counts)),
            "steps_per_trace": {
                "mean": C._f(statistics.fmean(counts)),
                "median": C._f(statistics.median(counts)),
                "min": int(min(counts)),
                "max": int(max(counts)),
            },
            "step_length_chars": {
                "mean": C._f(statistics.fmean(char_lens)),
                "median": C._f(statistics.median(char_lens)),
                "p10": C._f(_pct(char_lens, 10)),
                "p90": C._f(_pct(char_lens, 90)),
            },
            "step_length_words": {
                "mean": C._f(statistics.fmean(word_lens)),
                "median": C._f(statistics.median(word_lens)),
            },
            "text_coverage_fraction": C._f(statistics.fmean(cover)),
            "boundary_f1_vs_shipped": C._f(statistics.fmean(agree)),
            "shipped_boundaries_preserved": C._f(statistics.fmean(preserved)),
            "mean_steps_in_first_half_window": C._f(statistics.fmean(halves)),
            "per_model": per_model,
        }
        v = per_variant[name]
        print(f"  {name:19s} steps/trace={v['steps_per_trace']['mean']:7.1f} "
              f"chars={v['step_length_chars']['mean']:6.1f} "
              f"F1={v['boundary_f1_vs_shipped']:.3f} "
              f"preserved={v['shipped_boundaries_preserved']:.3f} "
              f"coverage={v['text_coverage_fraction']:.3f}")

    ship = per_variant["shipped"]
    payload = {
        "sweep": "segmentation",
        "question": (
            "How much does the step decomposition depend on the segmenter, and how much "
            "of the existing H_BE computation would survive a different one?"
        ),
        "requires_gpu": False,
        "requires_generation": False,
        "scope": (
            "Structural only. Recomputing H_BE at new boundaries requires resampling "
            "continuations from the target model, because a new boundary is a new prefix. "
            "The step counts here are exactly the number of resampling positions that "
            "full version would need."
        ),
        "source": "base_response.json:scratchpad, the complete un-segmented reasoning text",
        "coverage": {
            "models": sorted(traces.keys()),
            "n_traces": ship["n_traces"],
            "note": (
                "All seven model directories present in the artifacts are segmented, "
                "including Apriel-1.6-15b-Thinker and the _evil variant, which are not "
                "among the five reported models. Per-model figures are broken out."
            ),
        },
        "shipped_replica_gate": gate,
        "segmenter_definitions": {
            "shipped": "the pipeline's character loop, tail dropped, sub-10-char chunks merged",
            "shipped_keep_tail": "as shipped but retaining the final chunk",
            "shipped_no_merge": "as shipped but without the sub-10-character merge",
            "regex_sentence": "regex sentence split with an abbreviation guard",
            "paragraph": "blank-line splits only, coarser",
            "newline": "every line break",
            "clause": "sentence terminators plus semicolons and colons, finer",
        },
        "variants": per_variant,
    }
    C.write_results(OUTPUT, payload)


if __name__ == "__main__":
    main()
