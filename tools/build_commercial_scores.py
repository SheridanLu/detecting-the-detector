"""Convert the archived commercial-detector results of Liang et al. (2023) into one long table.

Input : the `Data_and_Results/` folder released with Liang et al. 2023,
        "GPT detectors are biased against non-native English writers" (github.com/Weixin-Liang/ChatGPT-Detector-Bias).
        Each corpus folder holds one JSON file per detector; entry i is the score for essay i of that corpus.
Output: data/commercial/liang2023_detector_scores.parquet with columns
        exam, corpus, essay_id, detector, score   (score = probability-of-AI reported by the detector; NaN = no score)

Reading rule copied from a40/metrics.py and a40/prf1.py: essay_id = f"{corpus}#{i}", a missing/None score becomes NaN.
Usage: python tools/build_commercial_scores.py <path/to/Data_and_Results> data/commercial/liang2023_detector_scores.parquet
"""
import json, pathlib, sys
import numpy as np, pandas as pd

DETECTORS = ["Crossplag", "GPTZero", "HFOpenAI", "OriginalityAI", "Quil", "Sapling", "ZeroGPT"]


def main(src, dst):
    src = pathlib.Path(src); rows = []
    for sub in ("Human_Data", "GPT_Data"):
        for cdir in sorted((src / sub).iterdir()):
            if not cdir.is_dir():
                continue
            for det in DETECTORS:
                f = cdir / f"{det}.json"
                if not f.exists():
                    continue
                for i, d in enumerate(json.loads(f.read_text(encoding="utf-8"))):
                    s = d.get("score")
                    rows.append(dict(exam="E1", corpus=cdir.name, essay_id=f"{cdir.name}#{i}", detector=det,
                                     score=np.nan if s is None else float(s)))
    out = pd.DataFrame(rows)
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dst, index=False)
    print(out.groupby(["corpus", "detector"]).score.agg(["size", "count"]).unstack().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
