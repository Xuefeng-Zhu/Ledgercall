# Single-file Kaggle T4x2 job used for the official LedgerCall run (model source: cohere-labs/tiny-aya/transformers/global/1).
import base64, io, os, subprocess, time, glob, zipfile
W = "/kaggle/working"
zipfile.ZipFile(io.BytesIO(base64.b64decode("<base64 zip of this repo: git archive --format=zip --prefix=ledgercall/ HEAD | base64 -w0>"))).extractall(W)
R = W + "/ledgercall"; os.chdir(R)
LOG = open(W + "/run_log.txt", "a")
def sh(cmd, check=False):
    t = time.time(); LOG.write(">>> " + cmd + "\n"); LOG.flush(); print(">>>", cmd, flush=True)
    p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = p.stdout[-15000:]; print(out, flush=True); LOG.write(out + f"\n[exit {p.returncode} in {time.time()-t:.0f}s]\n"); LOG.flush()
    if check and p.returncode: raise SystemExit(p.returncode)
    return p.returncode
cands = glob.glob("/kaggle/input/**/config.json", recursive=True)
MODEL = os.path.dirname(cands[0]); print("MODEL", MODEL, flush=True)
sh('pip install -q --ignore-requires-python "git+https://github.com/Lexsi-Labs/AgentTune" "auditkit[transformers]" peft datasets sentencepiece 2>&1 | tail -5')
sh("python -m ledgercall.data --out data/", check=True)
N = "100"
os.makedirs("results", exist_ok=True)
base = None and subprocess.Popen(f"CUDA_VISIBLE_DEVICES=1 python -m ledgercall.evaluate --model {MODEL} --data data/test.jsonl --tag base --limit {N} --out-dir results > {W}/base_eval.log 2>&1", shell=True)
sh(f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0 python -m ledgercall.sft --model {MODEL} --train_file data/train.jsonl --output_dir out/sft --epochs 1 --grad_accum 8 --lr 2e-4 --max_minutes 55 2>&1 | grep -v Warning")
sh(f"CUDA_VISIBLE_DEVICES=0 python -m ledgercall.evaluate --model out/sft --base-model {MODEL} --data data/test.jsonl --tag tuned --limit {N} --out-dir results")
sh("cp out/sft/train_summary.json results/ ")
sh(f"mkdir -p {W}/artifacts/adapter && cp -r results data {W}/artifacts/ && cp -r out/sft/. {W}/artifacts/adapter/ 2>/dev/null; ls -la {W}/artifacts/adapter")
