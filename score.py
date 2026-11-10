#!/usr/bin/env python3
"""ONTO Limits: judge-free scorer.

Scores a model's answers on the 25 demo claims against the answer key. No AI judge is used.

Two input modes, detected per answer:
  verdict   {"id": ..., "answer": "impossible"|"possible"|"need_data", "limit": "betz_power"|null}
            "impossible" counts only with the right limit group.
  grounded  {"id": ..., "verdict": {"answer": ...}, "calls": [{"result": {"within_limits": false, "limit": ...}}]}
            "impossible" counts only if one of the model's own calls returned within_limits == false
            with the right limit group. A bare "impossible" without such a call is scored as "unclear".

Usage:
  python score.py results/olmo2-1b-onto-rl.json
  python score.py my_answers.json --json        # machine-readable summary only
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEY = json.load(open(os.path.join(HERE, "data", "answer_key.json"), encoding="utf-8"))

# Limits that belong to the same physical law count as the same answer.
GROUP_OF = {
    "betz": "betz", "betz_power": "betz",
    "first_law_any": "first_law", "first_law_component": "first_law", "first_law_solar": "first_law",
    "first_law_engine": "first_law", "first_law_fuel_cell": "first_law",
    "shockley_queisser": "solar", "devos_inf_1sun": "solar", "devos_inf_max": "solar",
    "carnot": "carnot", "carnot_cop": "carnot", "carnot_cop_cooling": "carnot",
    "light_time_one_way": "light_speed",
    "white_light": "light", "kcd_683": "light",
    "incompressible": "compress", "object_K": "compress",
}
ANSWERS = ("impossible", "possible", "need_data")


def group(limit):
    return GROUP_OF.get(limit, limit)


def read_answer(r):
    """Return (answer, limit) for one model answer, in either mode."""
    if isinstance(r.get("verdict"), dict):  # grounded mode
        a = r["verdict"].get("answer")
        if a == "impossible":
            hits = [c["result"] for c in r.get("calls") or []
                    if isinstance(c.get("result"), dict) and c["result"].get("within_limits") is False]
            return ("impossible", hits[0].get("limit")) if hits else ("unclear", None)
        return (a, None) if a in ANSWERS else ("unclear", None)
    a = r.get("answer")  # verdict mode
    if a == "impossible":
        return ("impossible", r.get("limit")) if r.get("limit") else ("unclear", None)
    return (a, None) if a in ANSWERS else ("unclear", None)


def score(answers):
    by_id = {r["id"]: r for r in answers}
    s = dict(n=len(KEY), answered=0, impossible_total=0, impossible_right=0, possible_total=0, possible_right=0,
             need_data_total=0, need_data_right=0, false_impossible=0, unclear=0)
    rows = []
    for cid, k in sorted(KEY.items()):
        a, lim = read_answer(by_id[cid]) if cid in by_id else ("missing", None)
        s["answered"] += cid in by_id
        s[k["answer"] + "_total"] += 1
        ok = a == k["answer"] and (a != "impossible" or group(lim) == group(k["limit"]))
        if ok:
            s[a + "_right"] += 1
        if a == "impossible" and k["answer"] != "impossible":
            s["false_impossible"] += 1
        if a in ("unclear", "missing"):
            s["unclear"] += 1
        rows.append("%-12s key %-10s model %-10s %-18s %s" % (cid, k["answer"], a, lim or "", "OK" if ok else "--"))
    return s, rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    s, rows = score(data["answers"] if isinstance(data, dict) else data)
    if "--json" in sys.argv:
        print(json.dumps(s))
        return
    print("\n".join(rows))
    print()
    print("false \"impossible\": %d of %d possible claims" % (s["false_impossible"], s["possible_total"]))
    print("impossible found:   %d of %d (right limit)" % (s["impossible_right"], s["impossible_total"]))
    print("possible kept:      %d of %d" % (s["possible_right"], s["possible_total"]))
    print("need data noticed:  %d of %d" % (s["need_data_right"], s["need_data_total"]))
    print("unclear or missing: %d" % s["unclear"])


if __name__ == "__main__":
    main()
