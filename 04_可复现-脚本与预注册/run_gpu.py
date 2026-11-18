# int4096 unified_v4 — GPU orchestrator (run only after production GLM is stopped).
# Lane plan (8x A40):
#   GPUs 0-3 : Qwen seed0 A (0,1) and seed0 B (2,3) start immediately
#   GPUs 4-5 : RoBERTa queue (30 runs), 2 concurrent per GPU
#   GPU 6    : Binoculars ; GPU 7 : Fast-DetectGPT
#   when GPUs 4-7 are all free: Qwen seed1 A (4,5) and seed1 B (6,7)
import subprocess, time, json, pathlib, os, sys
ROOT = pathlib.Path("/data/int4096_unified_v4")
PY = str(ROOT / "venv/bin/python"); SC = ROOT / "scripts"; LOG = ROOT / "logs/gpu"; LOG.mkdir(parents=True, exist_ok=True)
STATUS = ROOT / "logs/orchestrator_status.json"
manifests = json.load(open(ROOT / "splits/manifests.json"))
rob_queue = [r for r in manifests]                       # all 30 runs (sizes 1000/2000/4000)
qwen_first = [("seed0_A_n4000", "0,1"), ("seed0_B_n4000", "2,3")]
qwen_second = [("seed1_A_n4000", "4,5"), ("seed1_B_n4000", "6,7")]
running = {}   # name -> (Popen, gpus)
done, failed = [], []

def launch(name, cmd, gpus):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
    f = open(LOG / f"{name}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    running[name] = (p, gpus)
    (LOG / f"{name}.pid").write_text(str(p.pid))

def write_status():
    STATUS.write_text(json.dumps({"time": time.strftime("%F %T"), "running": {k: v[1] for k, v in running.items()},
                                  "done": done, "failed": failed, "rob_left": len(rob_queue)}, indent=1))

for name, g in qwen_first:
    launch(f"qwen_{name}", [PY, str(SC / "train_qwen.py"), name], g)
def falcon_state():
    log = ROOT / "logs/download_falcon.log"
    t = log.read_text() if log.exists() else ""
    if "FALCON_DONE" in t:
        return "failed" if "SHORT" in t else "ready"
    return "pending"
os_started = False
os_skipped = False
rob_done_at = None
rob_slots = {"4": 2, "5": 2}
second_started = False
while True:
    for name, (p, g) in list(running.items()):
        rc = p.poll()
        if rc is None:
            continue
        (done if rc == 0 else failed).append(f"{name}:{rc}")
        del running[name]
        if name.startswith("rob_"):
            rob_slots[g] += 1
    for g in rob_slots:
        while rob_slots[g] > 0 and rob_queue:
            r = rob_queue.pop(0)
            launch(f"rob_{r}", [PY, str(SC / "train_roberta.py"), r], g)
            rob_slots[g] -= 1
    if not rob_queue and not any(k.startswith("rob_") for k in running) and rob_done_at is None:
        rob_done_at = time.time()
    fs = falcon_state()
    if not os_started and not os_skipped and (fs == "failed" or (rob_done_at and time.time() - rob_done_at > 5400 and fs != "ready")):
        os_skipped = True; failed.append(f"opensource:skipped_falcon_{fs}")
    if not os_started and not os_skipped and fs == "ready":
        launch("binoculars", [PY, str(SC / "score_opensource.py"), "binoculars"], "6")
        launch("fastdetect", [PY, str(SC / "score_opensource.py"), "fastdetect"], "7")
        os_started = True
    lane_free = (os_started or os_skipped) and not rob_queue and not any(k.startswith("rob_") or k in ("binoculars", "fastdetect") for k in running)
    if lane_free and not second_started:
        for name, g in qwen_second:
            launch(f"qwen_{name}", [PY, str(SC / "train_qwen.py"), name], g)
        second_started = True
    write_status()
    if second_started and not running:
        break
    time.sleep(20)
write_status()
print("ORCHESTRATOR_DONE", json.dumps({"done": done, "failed": failed}), flush=True)
