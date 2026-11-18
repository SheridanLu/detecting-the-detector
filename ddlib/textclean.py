"""Text normalisation used for every essay (training, validation and test).

`clean()` is copied character-for-character from the original A40 script `a40/common.py`.
Do not change it: every saved prediction in data/preds/ was produced from text passed through this function.

`format_gate()` re-implements the check reported as "FORMAT_GATE PASS" in the original run. The original check was run
interactively and its code was not saved as a script, so this is a re-implementation of the stated criterion:
after cleaning, no text in any pool may still contain a formatting artefact that a detector could use as a shortcut
(line breaks / tabs / repeated spaces, curly quotes or dashes, CP-1252 control characters, Cyrillic letters).
"""
import re as _re
import unicodedata as _ud

# ---- verbatim from a40/common.py (lines 7-29) ------------------------------------------------------------------
_CP1252 = {"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "-", "\x97": "-", "\x85": "...", "\x82": ","}
_PUNCT = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "´": "'", "`": "'",
          "“": '"', "”": '"', "„": '"', "–": "-", "—": "-", "―": "-", "−": "-"}
_HOMO = str.maketrans("аеорсхуіѕԁјԛԝАВЕКМНОРСТХ", "aeopcxyisdjqwABEKMHOPCTX")
_TL_LOWER = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i", "й": "i",
             "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f",
             "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
             "і": "i", "ї": "i", "є": "e", "ѕ": "s", "ј": "j"}
_TRANSLIT = dict(_TL_LOWER)
_TRANSLIT.update({k.upper(): v.capitalize() for k, v in _TL_LOWER.items()})
def clean(t):
    t = str(t)
    t = t.replace("Â ", " ").replace("Â", "")
    for a, b in _CP1252.items(): t = t.replace(a, b)
    t = _ud.normalize("NFKC", t)
    for a, b in _PUNCT.items(): t = t.replace(a, b)
    t = t.translate(_HOMO)
    t = "".join(_TRANSLIT.get(ch, ch) for ch in t)
    t = _re.sub(r"[\x80-\x9f]", "", t)
    t = _re.sub("[\u0400-\u052f]", "", t)   # any Cyrillic left after transliteration (9 chars in total)
    return _re.sub(r"\s+", " ", t).strip()
# ---- end of verbatim copy ----------------------------------------------------------------------------------------

# Artefacts that must be absent after clean(). Each entry: name -> regular expression.
FORMAT_ARTEFACTS = {
    "line break / tab": r"[\n\r\t]",
    "double space": r"  ",
    "leading/trailing space": r"^\s|\s$",
    "curly quote": "[\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u00b4`]",
    "long dash / minus": "[\u2013\u2014\u2015\u2212]",
    "CP-1252 control char": r"[\x80-\x9f]",
    "mojibake 'A-circumflex'": "\u00c2",
    "Cyrillic letter": "[\u0400-\u052f]",
}


def format_gate(pools):
    """pools: dict name -> iterable of already-cleaned texts.
    Returns (passed: bool, table: pandas.DataFrame with one row per pool and one column per artefact = number of texts
    that still contain it)."""
    import pandas as pd
    rows = []
    pats = {k: _re.compile(v) for k, v in FORMAT_ARTEFACTS.items()}
    for name, texts in pools.items():
        texts = list(texts)
        row = {"pool": name, "n_texts": len(texts)}
        for k, p in pats.items():
            row[k] = sum(1 for t in texts if p.search(t))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("pool")
    passed = int(table.drop(columns="n_texts").to_numpy().sum()) == 0
    return passed, table
