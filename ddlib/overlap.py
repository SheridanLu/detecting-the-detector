"""Zero-overlap check between training/validation texts and test (exam) texts.
Rule copied from a40/build_splits.py and a40/followup_ml.py: two essays count as the same if their whitespace-
normalised full text is equal OR their first 200 normalised characters are equal."""
import re


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def keys_of(t):
    n = norm(t)
    return n, n[:200]


def build_index(texts):
    full, first200 = set(), set()
    for t in texts:
        a, b = keys_of(t); full.add(a); first200.add(b)
    return full, first200


def count_overlap(texts, index):
    """Return (n_full_text_matches, n_first200_matches, n_any) of `texts` against an index from build_index()."""
    full, first200 = index
    nf = n2 = na = 0
    for t in texts:
        a, b = keys_of(t)
        f, s = a in full, b in first200
        nf += f; n2 += s; na += (f or s)
    return nf, n2, na
