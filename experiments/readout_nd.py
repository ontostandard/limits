"""READOUT-ND (prereg: PREREG_readout_nd_2026-11-12.md; design: DIAGNOSIS3 section 3).
Teach ONLY the readout: LoRA on transformer blocks above the probe layer, all lower blocks frozen,
loss = 3-way cross-entropy on the first answer token (possible / impossible / need), both sides of every pair.
The probe layer is untouched by construction; we re-fit the probe after training as an assembly check.
Usage: python readout_nd.py --data data --out out --seed 0 [--model ID] [--probe_layer 24] [--smoke]
"""
import argparse, json, os, time, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import probe_nd as P

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out_readout")
ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct"); ap.add_argument("--probe_layer", type=int, default=24)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--eval_every", type=int, default=20); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--ablate", action="store_true")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
CLS = ("possible", "impossible", "need_data")

tr, va, te = P.load(a.data)
key = {}
for f in ("real_claims_v1.json", "real_claims_v2.json"):
    key.update({k: v["class"] for k, v in json.load(open(os.path.join(a.data, f), encoding="utf-8"))["chosen"].items()})
# 3-class labels for train/val from the jsonl
def lab3(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l); out[r["claim"]] = CLS.index(r["verdict"]["answer"])
    return out
L3 = lab3(os.path.join(a.data, "olmo_claims_live2_v1157.jsonl"))
Xtr = [t for t, _ in tr]; Ytr = np.array([L3[t] for t in Xtr])
Xva = [t for t, _ in va]; Yva = np.array([L3[t] for t in Xva])
Xte = [t for t, _, _ in te]; Yte = np.array([CLS.index(key[i]) for _, _, i in te]); ids = [i for _, _, i in te]
if a.smoke:
    Xtr, Ytr, Xva, Yva = Xtr[:64], Ytr[:64], Xva[:32], Yva[:32]
    a.epochs, a.eval_every = 1, 2

tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto")
nL = m.config.num_hidden_layers
first = [tok.encode(" " + w, add_special_tokens=False)[0] for w in ("possible", "impossible", "need")]
assert len(set(first)) == 3
fmt = lambda t: tok.apply_chat_template([{"role": "user", "content": P.PROMPT.format(t=t)}], tokenize=False, add_generation_prompt=True) + P.PREFIX

def batch(texts):
    return tok([fmt(t) for t in texts], return_tensors="pt", padding=True).to(m.device)

@torch.no_grad()
def readout(model, texts, layer=None):
    """returns 3-way probs [n,3] over first answer token, and hidden state at `layer` (last token) if asked."""
    model.eval(); P3, H = [], []
    for i in range(0, len(texts), a.bs):
        b = batch(texts[i:i + a.bs])
        o = model(**b, output_hidden_states=layer is not None, logits_to_keep=1)
        P3.append(o.logits[:, -1, first].float().softmax(-1).cpu().numpy())
        if layer is not None:
            H.append(o.hidden_states[layer][:, -1, :].float().cpu().numpy())
    return np.concatenate(P3), (np.concatenate(H) if H else None)

def metrics(p3, y):
    nd = (y == 2).astype(int)
    pred = p3.argmax(1)
    m_ = {"auroc_need": float(roc_auc_score(nd, p3[:, 2])),
          "caught_of_nd": int(((pred == 2) & (nd == 1)).sum()), "n_nd": int(nd.sum()),
          "false_need": int(((pred == 2) & (nd == 0)).sum()), "n_other": int((nd == 0).sum())}
    pi = y != 2
    s = p3[pi, 1] / (p3[pi, 0] + p3[pi, 1] + 1e-9)
    m_["auroc_pos_vs_imp"] = float(roc_auc_score((y[pi] == 1).astype(int), s))
    m_["pred_counts"] = {CLS[k]: int((pred == k).sum()) for k in range(3)}
    return m_

t0 = time.time()
# BEFORE: base model readout + probe features on train and test (probe layer is frozen by construction)
p_te0, h_te0 = readout(m, Xte, a.probe_layer); before = metrics(p_te0, Yte)
_, h_tr0 = readout(m, Xtr, a.probe_layer)
probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=3000)).fit(h_tr0, (Ytr == 2).astype(int))
probe_auc0 = float(roc_auc_score((Yte == 2).astype(int), probe.predict_proba(h_te0)[:, 1]))
print(f"BEFORE {json.dumps(before)} probe {probe_auc0:.3f}", flush=True)

# LoRA only on blocks ABOVE the probe layer (hidden_states[k] = output of block k-1, so blocks k..nL-1)
layers = list(range(a.probe_layer, nL))
cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, layers_to_transform=layers,
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
m = get_peft_model(m, cfg); m.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
ce = torch.nn.CrossEntropyLoss()
# sampler: need_data share fixed at 25% per batch (prereg); possible/impossible fill the rest in their natural ratio
nd_idx = np.where(Ytr == 2)[0]; ot_idx = np.where(Ytr != 2)[0]
steps_per_epoch = max(1, len(Xtr) // a.bs); total = steps_per_epoch * a.epochs
k_nd = max(1, round(0.25 * a.bs))
best = {"loss": 1e9, "step": 0}; log = []
def val_loss():
    p3, _ = readout(m, Xva)
    return float(-np.log(p3[np.arange(len(Yva)), Yva] + 1e-9).mean())
for step in range(1, total + 1):
    m.train()
    idx = np.concatenate([rng.choice(nd_idx, k_nd), rng.choice(ot_idx, a.bs - k_nd)])
    b = batch([Xtr[i] for i in idx])
    lg = m(**b, logits_to_keep=1).logits[:, -1, first].float()
    loss = ce(lg, torch.tensor(Ytr[idx], device=lg.device))
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if step % a.eval_every == 0 or step == total:
        vl = val_loss(); log.append({"step": step, "train": float(loss), "val": vl})
        print(f"step {step}/{total} train {loss.item():.3f} val {vl:.3f} {time.time() - t0:.0f}s", flush=True)
        if vl < best["loss"]:
            best = {"loss": vl, "step": step}
            m.save_pretrained(os.path.join(a.out, "best_adapter"))
# reload best adapter weights
from peft import set_peft_model_state_dict
from safetensors.torch import load_file
lr_ = set_peft_model_state_dict(m, load_file(os.path.join(a.out, "best_adapter", "adapter_model.safetensors")))
assert not getattr(lr_, "unexpected_keys", []), f"adapter reload: unexpected keys {lr_.unexpected_keys[:3]}"
print("adapter reloaded from step", best["step"], flush=True)
p_te1, h_te1 = readout(m, Xte, a.probe_layer); after = metrics(p_te1, Yte)
probe_auc1 = float(roc_auc_score((Yte == 2).astype(int), probe.predict_proba(h_te1)[:, 1]))
assembly_ok = abs(probe_auc1 - probe_auc0) < 0.005
drop = before["auroc_pos_vs_imp"] - after["auroc_pos_vs_imp"]
if not assembly_ok:
    verdict = "INVALID (probe layer changed: code error)"
elif after["auroc_need"] >= 0.75 and after["false_need"] <= 16 and after["caught_of_nd"] >= 13 and drop <= 0.05:
    verdict = "SUCCESS"
elif after["auroc_need"] < 0.65 or after["false_need"] > 31 or drop > 0.10:
    verdict = "FAIL"
else:
    verdict = "BETWEEN"
res = {"prereg": "PREREG_readout_nd_2026-11-12.md", "model": a.model, "seed": a.seed, "smoke": a.smoke,
       "probe_layer": a.probe_layer, "lora_layers": layers, "best": best, "log": log,
       "before": before, "after": after, "pos_vs_imp_drop": drop,
       "probe_auc_before": probe_auc0, "probe_auc_after": probe_auc1, "assembly_ok": assembly_ok, "verdict": verdict,
       "per_item": [{"id": i, "key": CLS[y], "p_before": [round(float(x), 4) for x in pb], "p_after": [round(float(x), 4) for x in pa]}
                    for i, y, pb, pa in zip(ids, Yte, p_te0, p_te1)]}
json.dump(res, open(os.path.join(a.out, f"readout_result_s{a.seed}.json"), "w"), indent=1)
print(f"VERDICT seed {a.seed}: {verdict} | need AUROC {before['auroc_need']:.3f} -> {after['auroc_need']:.3f} | caught {after['caught_of_nd']}/{after['n_nd']} "
      f"false {after['false_need']}/{after['n_other']} | pos/imp {before['auroc_pos_vs_imp']:.3f} -> {after['auroc_pos_vs_imp']:.3f} | "
      f"probe {probe_auc0:.3f} -> {probe_auc1:.3f} | best step {best['step']}", flush=True)

# ---------------- ABLATE-ND (PREREG_ablate_nd_2026-11-12.md) ----------------
if a.ablate:
    from sklearn.linear_model import LogisticRegression as LR
    yb = (Ytr == 2).astype(int); yv = (Yva == 2).astype(int)
    _, h_va = readout(m, Xva, a.probe_layer)            # layer-24 input is below LoRA: same with/without adapter
    Xt = h_tr0.astype(np.float64); mu = Xt.mean(0); sc = Xt.std() + 1e-8
    Z = (Xt - mu) / sc; W = []
    for it in range(8):
        w = LR(C=0.01, max_iter=3000).fit(Z, yb).coef_[0]
        for u in W: w = w - (w @ u) * u
        w = w / (np.linalg.norm(w) + 1e-12); W.append(w); Z = Z - np.outer(Z @ w, w)
    B = np.stack(W)                                      # 8 x d, orthonormal
    def erase(X, B): Xc = (X - mu); return X - (Xc @ B.T) @ B
    chk = LR(C=0.01, max_iter=3000).fit((erase(Xt, B) - mu) / sc, yb)
    erase_auc = float(roc_auc_score(yv, chk.predict_proba((erase(h_va.astype(np.float64), B) - mu) / sc)[:, 1]))
    print(f"erasure check AUROC {erase_auc:.3f} (must be <= 0.60)", flush=True)
    layer_mod = [mod for name, mod in m.named_modules() if name.endswith(f"layers.{a.probe_layer}")][0]
    dev = next(layer_mod.parameters()).device
    mu_t = torch.tensor(mu, dtype=torch.float32, device=dev)
    def run_with(Bm):
        Bt = torch.tensor(Bm, dtype=torch.float32, device=dev)
        def pre(module, args, kwargs):
            if args: h = args[0]
            else: h = kwargs["hidden_states"]
            hf = h.float(); hf = hf - ((hf - mu_t) @ Bt.T) @ Bt; h2 = hf.to(h.dtype)
            if args: return (h2,) + tuple(args[1:]), kwargs
            kwargs["hidden_states"] = h2; return args, kwargs
        hd = layer_mod.register_forward_pre_hook(pre, with_kwargs=True)
        try: p3, _ = readout(m, Xva)
        finally: hd.remove()
        return float(roc_auc_score(yv, p3[:, 2]))
    def auc_plain():
        p3, _ = readout(m, Xva); return float(roc_auc_score(yv, p3[:, 2]))
    rr = np.random.default_rng(100 + a.seed); d = B.shape[1]
    rand = [np.linalg.qr(rr.normal(size=(d, 8)))[0].T for _ in range(5)]
    out = {}
    for tag, ctx in (("adapter", None), ("base", "base")):
        if ctx == "base":
            with m.disable_adapter():
                a0 = auc_plain(); a1 = run_with(B); ar = [run_with(R) for R in rand]
        else:
            a0 = auc_plain(); a1 = run_with(B); ar = [run_with(R) for R in rand]
        out[tag] = {"auc": a0, "auc_ablated": a1, "auc_random": ar, "drop": a0 - a1, "drop_random": a0 - float(np.mean(ar))}
        print(f"ABLATE {tag}: auc {a0:.3f} ablated {a1:.3f} random {np.mean(ar):.3f} drop {a0-a1:+.3f} drop_random {a0-np.mean(ar):+.3f}", flush=True)
    o = out["adapter"]
    if erase_auc > 0.60: v = "INVALID (erasure incomplete)"
    elif o["drop"] >= 0.10 and o["drop_random"] <= 0.03: v = "SUCCESS"
    elif o["drop"] < 0.05 or (o["drop"] - o["drop_random"]) < 0.03: v = "FAIL"
    else: v = "BETWEEN"
    json.dump({"prereg": "PREREG_ablate_nd_2026-11-12.md", "seed": a.seed, "erase_check_auc": erase_auc, "results": out, "verdict_seed": v},
              open(os.path.join(a.out, f"ablate_result_s{a.seed}.json"), "w"), indent=1)
    print(f"ABLATE VERDICT seed {a.seed}: {v}", flush=True)
