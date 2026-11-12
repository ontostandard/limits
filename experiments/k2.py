"""K2 (PREREG_K1K2_2026-11-12.md): is the verbal-output readout of probe_nd.py sound?
(a) share of full next-token probability on the 3 answer tokens; (b) greedy answers on 201 vs key; (c) logreg on 3 logits trained on 1750.
Usage: python k2.py --model ID --out DIR
"""
import argparse, json, os, re, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import probe_nd as P

ap = argparse.ArgumentParser(); ap.add_argument("--model"); ap.add_argument("--out"); ap.add_argument("--data", default="data"); ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
tr, va, te = P.load(a.data)
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
W = ("possible", "impossible", "need")
ids_sp = [tok.encode(" " + w, add_special_tokens=False)[0] for w in W]
ids_ns = [tok.encode(w, add_special_tokens=False)[0] for w in W]
fmt = lambda t: tok.apply_chat_template([{"role": "user", "content": P.PROMPT.format(t=t)}], tokenize=False, add_generation_prompt=True) + P.PREFIX

def logits(texts):
    L, mass_sp, mass_any, top = [], [], [], []
    for i in range(0, len(texts), a.bs):
        b = tok([fmt(t) for t in texts[i:i + a.bs]], return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            lg = m(**b).logits[:, -1, :].float()
        pr = lg.softmax(-1)
        L.append(lg[:, ids_sp].cpu().numpy())
        mass_sp.append(pr[:, ids_sp].sum(1).cpu().numpy())
        mass_any.append(pr[:, list(set(ids_sp + ids_ns))].sum(1).cpu().numpy())
        top += [tok.decode([j]) for j in lg.argmax(-1).tolist()]
    return np.concatenate(L), np.concatenate(mass_sp), np.concatenate(mass_any), top

Ltr, *_ = logits([t for t, _ in tr]); ytr = np.array([y for _, y in tr])
Lte, msp, many, top = logits([t for t, _, _ in te]); yte = np.array([y for _, y, _ in te])
clf = LogisticRegression(max_iter=3000).fit(Ltr, ytr); auc_c = roc_auc_score(yte, clf.predict_proba(Lte)[:, 1])
# (b) greedy generation
gen = []
for i in range(0, len(te), a.bs):
    b = tok([fmt(t) for t, _, _ in te[i:i + a.bs]], return_tensors="pt", padding=True).to(m.device)
    with torch.no_grad():
        o = m.generate(**b, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    gen += [tok.decode(x[b["input_ids"].shape[1]:], skip_special_tokens=True) for x in o]
def cls(s):
    s = s.lower()
    if "need" in s or "not enough" in s: return "need_data"
    if "impossible" in s: return "impossible"
    if "possible" in s: return "possible"
    return "other"
key = {}
for f in ("real_claims_v1.json", "real_claims_v2.json"):
    key.update({k: v["class"] for k, v in json.load(open(os.path.join(a.data, f)))["chosen"].items()})
tab = {}
for (_, _, i), g in zip(te, gen):
    k = (key[i], cls(g)); tab[f"{k[0]}->{k[1]}"] = tab.get(f"{k[0]}->{k[1]}", 0) + 1
res = {"model": a.model, "a_mass_space_tokens_median": float(np.median(msp)), "a_mass_any_variant_median": float(np.median(many)),
       "a_top_token_counts": {t: top.count(t) for t in sorted(set(top), key=top.count, reverse=True)[:8]},
       "b_greedy_table": tab, "c_logreg_3logits_auc": float(auc_c),
       "verdict_a": "ARTIFACT (median<0.5)" if np.median(msp) < 0.5 else "ok", "verdict_c": "NOT chance (>=0.70)" if auc_c >= 0.70 else "ok",
       "greedy": [{"id": i, "key": key[i], "gen": g} for (_, _, i), g in zip(te, gen)]}
json.dump(res, open(os.path.join(a.out, "k2_result.json"), "w"), indent=1)
print(f"K2 {a.model}: (a) median mass {np.median(msp):.3f} (any variant {np.median(many):.3f}) top {res['a_top_token_counts']} · (c) logreg-3-logits AUROC {auc_c:.3f} · (b) {tab} · {res['verdict_a']} / {res['verdict_c']}")
