#!/usr/bin/env python3
"""Run an open Hugging Face instruct model on the ONTO Limits claims and save answers for score.py.

Usage: python examples/run_hf.py MODEL_ID_OR_PATH out.json [--4bit] [--name PUBLIC_MODEL_NAME]
"""
import json, os, re, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLAIMS = json.load(open(os.path.join(HERE, "data", "claims.json"), encoding="utf-8"))
CLAIMS = CLAIMS["claims"] if isinstance(CLAIMS, dict) else CLAIMS

# Names shown to the model -> names the scorer understands (score.py GROUP_OF).
LIMITS = {
    "betz": "betz", "carnot": "carnot", "first_law": "first_law_any",
    "luminous_efficacy": "white_light", "shockley_queisser": "shockley_queisser",
    "speed_of_light": "light_time_one_way", "data_compression": "incompressible",
    "information_density": "dna_per_gram", "other": "other",
}
PROMPT = (
    "Read this claim from a real product page or news story.\n\n{claim}\n\n"
    "Is what it claims physically possible? Answer with one JSON object and nothing else:\n"
    '{{"answer": "impossible" | "possible" | "need_data", "limit": one of [' + ", ".join(LIMITS) + '] or null}}\n'
    "Use \"impossible\" only if the numbers break a physical limit, and name that limit. "
    "Use \"need_data\" if the text does not give enough to decide."
)


def parse(text):
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        o = json.loads(m.group(0)) if m else {}
    except Exception:
        o = {}
    a = str(o.get("answer", "")).strip().lower()
    lim = o.get("limit")
    lim = LIMITS.get(str(lim).strip().lower()) if lim else None
    return {"answer": a, "limit": lim if a == "impossible" else None}


def main():
    model_id, out = sys.argv[1], sys.argv[2]
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16) if "--4bit" in sys.argv else None
    tok = AutoTokenizer.from_pretrained(model_id)
    kw = {"quantization_config": q} if q else {}  # a pre-quantized checkpoint carries its own config
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw, device_map="cuda:0" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16)
    rows, t0 = [], time.time()
    for c in CLAIMS:
        msgs = [{"role": "user", "content": PROMPT.format(claim=c["text"])}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(model.device)
        with torch.no_grad():
            g = model.generate(ids, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        raw = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        r = {"id": c["id"], **parse(raw), "raw": raw}
        rows.append(r)
        print(c["id"], r["answer"], r["limit"], flush=True)
    json.dump({"model": sys.argv[sys.argv.index("--name") + 1] if "--name" in sys.argv else model_id, "settings": {"quant": "nf4" if (q or getattr(model.config, "quantization_config", None)) else "bf16", "decoding": "greedy", "max_new_tokens": 256},
               "answers": rows}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("done in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
