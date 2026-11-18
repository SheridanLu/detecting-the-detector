# Follow-up GPU orchestrator (arms C and D for Qwen 27B and RoBERTa). Run only after production GLM is stopped.
# Qwen: 4 jobs x 2 GPUs fill all 8 cards. RoBERTa: 10 jobs, one at a time per GPU, co-located with Qwen in the free
# headroom (memory-capped so it can never starve Qwen), launched only once that GPU's Qwen job is in its training loop.
import subprocess, time, json, pathlib, os
ROOT = pathlib.Path("/data/int4096_unified_v4"); PY = str(ROOT / "venv/bin/python"); SC = ROOT / "scripts"
LOG = ROOT / "logs/gpu_followup"; LOG.mkdir(parents=True, exist_ok=True)
STATUS = ROOT / "logs/orchestrator_followup_status.json"
qwen = [] if (ROOT / "SKIP_QWEN").exists() else [("seed0_C", "0,1"), ("seed0_D", "2,3"), ("seed1_C", "4,5"), ("seed1_D", "6,7")]
print("Qwen jobs:", [r for r, _ in qwen] or "SKIPPED (SKIP_QWEN present)", flush=True)
rob_queue = [f"seed{s}_{a}" for s in range(5) for a in "CD"]
running, done, failed = {}, [], []
attempts, retry = {}, []

def launch(name, cmd, gpus, extra_env=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", **(extra_env or {}))
    f = open(LOG / f"{name}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    running[name] = (p, gpus); (LOG / f"{name}.pid").write_text(str(p.pid))

def free_mib(g):
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    for l in out.strip().split("\n"):
        i, u, t = [x.strip() for x in l.split(",")]
        if int(i) == g: return int(t) - int(u)
    return 0

def qwen_training(run):          # ETA file is written at step 5 of the training loop
    return (ROOT / "logs" / f"qwen_eta_{run}.json").exists()

for run, g in qwen:
    launch(f"qwen_{run}", [PY, str(SC / "train_qwen.py"), run], g)
gpu_of = {}                      # GPU index -> qwen run owning it
for run, g in qwen:
    for x in g.split(","): gpu_of[int(x)] = run
rob_gpu_busy = {g: False for g in range(8)}
while True:
    for name, (p, g) in list(running.items()):
        rc = p.poll()
        if rc is None: continue
        del running[name]
        if name.startswith("rob_"):
            rob_gpu_busy[int(g)] = False
            r = name[4:]; attempts[r] = attempts.get(r, 0) + 1
            if rc != 0 and attempts[r] < 2: retry.append(r); failed.append(f"{name}:{rc}:will_retry"); continue
        (done if rc == 0 else failed).append(f"{name}:{rc}")
    qwen_alive = any(k.startswith("qwen_") for k in running)
    if not qwen_alive and retry:
        rob_queue.extend(retry); retry.clear()
    for g in range(8):
        if not rob_queue or rob_gpu_busy[g]: continue
        ready = (not qwen_alive) or (qwen_training(gpu_of[g]) and free_mib(g) >= 12000)
        if ready:
            r = rob_queue.pop(0); launch(f"rob_{r}", [PY, str(SC / "train_roberta.py"), r], str(g), ({"ROB_MEMFRAC": "0.17"} if attempts.get(r, 0) == 0 else {}))
            rob_gpu_busy[g] = True
    STATUS.write_text(json.dumps({"time": time.strftime("%F %T"), "running": {k: v[1] for k, v in running.items()},
                                  "done": done, "failed": failed, "rob_left": len(rob_queue)}, indent=1))
    if not running and not rob_queue and not retry: break
    time.sleep(20)
print("FOLLOWUP_ORCHESTRATOR_DONE", json.dumps({"done": done, "failed": failed}), flush=True)
