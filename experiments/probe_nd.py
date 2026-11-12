"""PROBE-ND (prereg: PREREG_probe_nd_2026-11-12.md, written before data).
Is there an internal "the text lacks a number" signal in Qwen2.5-Coder-7B-Instruct, beyond surface and model output?
Weights untouched. Usage: python probe_nd.py --data data --out out [--model ID] [--dry]
"""
import argparse, json, os, re, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

PROMPT = ("Claim: {t}\n\nIs what this claim states physically possible? Answer with exactly one word: "
          "possible, impossible, or need_data (the text does not give enough numbers to decide).")
PREFIX = "Answer:"
UNIT = re.compile(r"(?<![A-Za-z])(W|kW|MW|GW|TW|Wh|kWh|MWh|GWh|V|kV|A|mA|Ah|mAh|J|kJ|MJ|N|mN|kN|km|cm|mm|nm|m|s|ms|ns|h|hr|kg|g|mg|K|°C|°F|lm|lux|Hz|kHz|MHz|GHz|KB|MB|GB|TB|PB|%|m²|m2|m/s|km/h|mph|rpm|COP|SEER|lbs?|ft|in|Da|eV)(?![A-Za-z])"
                  r"|\b(?:(?:kilo|mega|giga|milli|micro|nano)?(?:watts?|volts?|amps?|amperes?|joules?|newtons?|meters?|metres?|seconds?|grams?|hertz)"
                  r"|hours?|minutes?|lumens?|degrees?|celsius|fahrenheit|kelvin|percent|bits?|bytes?|miles?|feet|inches?|liters?|litres?|knots?|horsepower|hp|btu)\b", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)*")


def strip_q(q):
    q = re.sub(r"^Is this possible\?\s*", "", q).strip()
    return q[1:-1] if q.startswith('"') and q.endswith('"') else q


def load(d):
    tr, va = [], []
    for l in open(os.path.join(d, "olmo_claims_live2_v1157.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        (va if r.get("heldout") else tr).append((r["claim"], int(r["verdict"]["answer"] == "need_data")))
    te = []
    for f in ("real_claims_v1.json", "real_claims_v2.json"):
        j = json.load(open(os.path.join(d, f), encoding="utf-8"))
        for q in j["questions"]:
            te.append((strip_q(q["question"]), int(j["chosen"][q["id"]]["class"] == "need_data"), q["id"]))
    return tr, va, te


def surface(texts):
    return np.array([[len(NUM.findall(t)), len(UNIT.findall(t)), len(t), len(t.split())] for t in texts], float)


def extract(texts, a):
    """returns {layer: [n, d]} last-token hidden states and p_need (output baseline)."""
    if a.dry:
        rng = np.random.default_rng(0)
        return {L: rng.normal(size=(len(texts), 64)) for L in (0, 4, 8)}, rng.random(len(texts))
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    nL = m.config.num_hidden_layers
    layers = sorted(set(list(range(0, nL + 1, 4)) + [nL]))
    first = {w: tok.encode(" " + w, add_special_tokens=False)[0] for w in ("possible", "impossible", "need")}
    print("first-token ids", first, "layers", layers, flush=True)
    assert len(set(first.values())) == 3, "answer words share a first token"
    def fmt(t):
        msgs = [{"role": "user", "content": PROMPT.format(t=t)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + PREFIX
    H = {L: [] for L in layers}; P = []
    for i in range(0, len(texts), a.bs):
        b = tok([fmt(t) for t in texts[i:i + a.bs]], return_tensors="pt", padding=True).to(m.device)
        with torch.no_grad():
            o = m(**b, output_hidden_states=True)
        for L in layers:
            H[L].append(o.hidden_states[L][:, -1, :].float().cpu().numpy())
        lg = o.logits[:, -1, :].float()
        sel = torch.stack([lg[:, first[w]] for w in ("possible", "impossible", "need")], 1).softmax(1)
        P.append(sel[:, 2].cpu().numpy())
        if i % (a.bs * 20) == 0:
            print(f"{i}/{len(texts)} {time.strftime('%H:%M:%S')}", flush=True)
    return {L: np.concatenate(v) for L, v in H.items()}, np.concatenate(P)


def boot(y, s1, s2, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tr, va, te = load(a.data)
    overlap = len({t for t, _ in tr + va} & {t for t, _, _ in te})
    print(f"train {len(tr)} (nd {sum(y for _, y in tr)}) · val {len(va)} (nd {sum(y for _, y in va)}) · "
          f"test {len(te)} (nd {sum(y for _, y, _ in te)}) · exact-text overlap train/test {overlap}", flush=True)
    assert len(te) == 201 and sum(y for _, y, _ in te) == 45, "test set must be 201 / 45 need_data"
    texts = [t for t, _ in tr] + [t for t, _ in va] + [t for t, _, _ in te]
    y = np.array([y for _, y in tr] + [y for _, y in va] + [y for _, y, _ in te])
    ntr, nva = len(tr), len(va)
    S = slice(0, ntr); V = slice(ntr, ntr + nva); T = slice(ntr + nva, None)
    feats = os.path.join(a.out, "feats.npz")
    if os.path.exists(feats) and not a.dry:
        z = np.load(feats); H = {int(k[1:]): z[k] for k in z.files if k.startswith("L")}; p_need = z["p_need"]
    else:
        H, p_need = extract(texts, a)
        if not a.dry:
            np.savez_compressed(feats, p_need=p_need, **{f"L{L}": v for L, v in H.items()})
    # probe: layer and C chosen by 5-fold CV on train only
    best = None; cv = {}
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for L, X in H.items():
        for C in (0.01, 0.1, 1.0):
            sc = []
            for a_, b_ in skf.split(X[S], y[S]):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
                clf.fit(X[S][a_], y[S][a_]); sc.append(roc_auc_score(y[S][b_], clf.predict_proba(X[S][b_])[:, 1]))
            cv[f"L{L}_C{C}"] = float(np.mean(sc))
            if best is None or np.mean(sc) > best[0]:
                best = (float(np.mean(sc)), L, C)
    _, L, C = best
    probe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(H[L][S], y[S])
    s_probe = probe.predict_proba(H[L])[:, 1]
    surf = surface(texts)
    s_surf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(surf[S], y[S]).predict_proba(surf)[:, 1]
    yt = y[T]
    auc = {"probe": roc_auc_score(yt, s_probe[T]), "output": roc_auc_score(yt, p_need[T]), "surface": roc_auc_score(yt, s_surf[T])}
    base = max(("output", "surface"), key=lambda k: auc[k])
    lo, hi = boot(yt, s_probe[T], p_need[T] if base == "output" else s_surf[T])
    diff = auc["probe"] - auc[base]
    if auc["probe"] >= 0.80 and diff >= 0.10 and lo > 0:
        verdict = "SUCCESS"
    elif auc["probe"] < 0.70 or diff <= 0:
        verdict = "FAIL"
    else:
        verdict = "BETWEEN"
    # record (not a gate): threshold from val at <=5% false positives
    vneg = np.sort(s_probe[V][y[V] == 0])
    thr = float(vneg[int(np.ceil(0.95 * len(vneg))) - 1]) if len(vneg) else 1.0
    caught = int(((s_probe[T] > thr) & (yt == 1)).sum()); false = int(((s_probe[T] > thr) & (yt == 0)).sum())
    res = {"prereg": "PREREG_probe_nd_2026-11-12.md", "model": a.model, "dry": a.dry,
           "n": {"train": ntr, "val": nva, "test": int(len(yt)), "test_nd": int(yt.sum())},
           "overlap_train_test": overlap, "chosen": {"layer": int(L), "C": C, "cv_auc": best[0]},
           "auc_test": {k: round(float(v), 4) for k, v in auc.items()}, "best_baseline": base,
           "diff": round(float(diff), 4), "diff_ci95": [round(lo, 4), round(hi, 4)], "verdict": verdict,
           "record_val_thr_fpr5": {"thr": thr, "caught_of_45": caught, "false_of_156": false},
           "per_item_test": [{"id": te[i][2], "y": int(yt[i]), "probe": round(float(s_probe[T][i]), 4),
                              "p_need": round(float(p_need[T][i]), 4)} for i in range(len(yt))],
           "cv": cv}
    json.dump(res, open(os.path.join(a.out, "probe_nd_result.json"), "w"), indent=1)
    print(f"VERDICT {verdict} · AUROC probe {auc['probe']:.3f} · output {auc['output']:.3f} · surface {auc['surface']:.3f} · "
          f"diff vs {base} {diff:+.3f} CI [{lo:+.3f},{hi:+.3f}] · layer {L} C {C} · record: {caught}/45 caught, {false}/156 false")


if __name__ == "__main__":
    main()
