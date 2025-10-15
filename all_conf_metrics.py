import os
import argparse
import numpy as np
import json
import random
from typing import List, Any

import torch
from datasets import load_dataset
from tqdm import tqdm

from prm import load_prm
from utils.llm import load_llm_with_retries, get_prompt_format, get_sampling_params

def parse_args():
    parser = argparse.ArgumentParser(description="PRM confidence experiment with vLLM (no generation).")

    parser.add_argument(
        "--model_id", 
        type=str, 
        default="meta-llama/Llama-3.2-1B-Instruct"
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--prm_model_name",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        choices=[
            "peiyi9979/math-shepherd-mistral-7b-prm",
            "Qwen/Qwen2.5-Math-PRM-7B",
            "GAIR/ReasonEval-7B",
        ],
        help="Name or path of the reward (PRM) model."
    )
    parser.add_argument("
        --prm_peft_dir", 
        type=str, 
        default=None,
        help="Optional PEFT adapter dir for PRM (if you use calibrated heads)."
    )
    parser.add_argument("
        --lb_idx", 
        type=int, 
        default=None,
        help="Optional: if your PRM outputs quantiles, choose lower-bound index (e.g., 0,1,2)."
    )
    parser.add_argument(
        "--prm_device", 
        type=str, 
        default=None,
        help="Device for PRM, e.g. cuda:1 or cpu. Default: cuda:1 if available else cuda:0/cpu."
    )
    parser.add_argument(
        "--prm_batch_size", 
        type=int, 
        default=16,
        help="Batch size for PRM scoring."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="young-j-park/prm_calibration",
        help="HuggingFace dataset name to use."
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=1
    )
    parser.add_argument("
        --subset_size", 
        type=int, 
        default=1000,
        help="Random subset size for faster runs."
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.2 # because PRM
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true"
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true"
    )
    parser.add_argument("
        --topk_logprobs", 
        type=int, 
        default=20,
        help="Top-k to collect for logprobs at each position."
    )
    parser.add_argument("
        --dtype", 
        type=str, 
        default="auto"
    )
    parser.add_argument("
        --seed", 
        type=int, 
        default=42
    )
    parser.add_argument("
        --output_dir", 
        type=str, 
        default="./mihika_experiment_results"
    )
    parser.add_argument("
        --output_name", 
        type=str, 
        default="experiment_prm_conf_nogen.json"
    )
    parser.add_argument("
        --debug", 
        action="store_true"
    )
    return parser.parse_args()

def get_entropy_from_logprobs(logprobs: List[float]) -> float:
    if not logprobs:
        return float("nan")
    a = np.array(logprobs, dtype=np.float64)
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    Z = np.sum(exp_a)
    if Z == 0.0 or not np.isfinite(Z):
        return float("nan")
    p = exp_a / Z
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -np.sum(p * np.log(p))
    return float(H)

def extract_topk_logprobs_list(pos_entry: Any, k: int = 20) -> List[float]:
    """
    pos_entry: vLLM returns per-position dict of token->logprob (top candidates),
               or objects with .logprob fields.
    Return sorted (desc) list of top-k logprobs.
    """
    if pos_entry is None:
        return []
    vals = []
    if isinstance(pos_entry, dict):
        for _, v in pos_entry.items():
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                try:
                    vals.append(float(getattr(v, "logprob")))
                except Exception:
                    continue
    else:
        try:
            for cand in pos_entry:
                if hasattr(cand, "logprob"):
                    vals.append(float(cand.logprob))
        except Exception:
            pass
    vals.sort(reverse=True)
    return vals[:k]


def per_position_token_confidence(topk_lists: List[List[float]]) -> List[float]:
    """token_confidence[i] = - average(top-k logprob) at position i."""
    out = []
    for lp_list in topk_lists:
        if not lp_list:
            out.append(float("nan"))
        else:
            out.append(-float(np.mean(lp_list)))
    return out


def build_sampling_params(args):
    """
    Try to use repo's get_sampling_params; if its signature doesn't support
    what we need, fall back to vllm.SamplingParams to guarantee prompt_logprobs.
    """
    want = dict(
        model_id=args.model_id,
        n_generations=args.n_generations,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        prompt_logprobs=args.topk_logprobs
    )
    try:
        return get_sampling_params(**want)
    except TypeError:
        try:
            from vllm import SamplingParams
        except Exception as e:
            raise RuntimeError("Need prompt_logprobs but utils.llm.get_sampling_params does not support it, "
                               "and vllm.SamplingParams is unavailable.") from e
        return SamplingParams(
            n=args.n_generations,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt_logprobs=args.topk_logprobs
        )

def score_with_prm_batched(prm, questions: List[str], traces: List[str], batch_size: int = 16):
    """
    questions: List[str] length N
    traces:    List[str] length N  (the reasoning text to score)
    Returns:   List[List[float]] length N (each inner list = per-trace PRM scores)
    """
    assert len(questions) == len(traces)
    out = []
    for i in tqdm(range(0, len(questions), batch_size), desc="PRM scoring", leave=False):
        q_batch = questions[i:i+batch_size]
        t_batch = traces[i:i+batch_size]
        scores = prm.score(q_batch, t_batch)   # List[float] or List[List[float]]
        # Normalize to List[List[float]]
        if len(scores) and not isinstance(scores[0], (list, tuple, np.ndarray)):
            scores = [[float(s)] for s in scores]
        out.extend(scores)
    return out

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = "math500/Qwen2.5-Math-7B-Instruct/data.json"
    ds = load_dataset(args.dataset_name, data_files=dataset_path, split="train")

    if args.subset_size and args.subset_size < len(ds):
        indices = random.sample(range(len(ds)), args.subset_size)
        ds = ds.select(indices)

    questions = [row["question"] for row in ds]
    reasoning_prefixes = [row.get("reasoning_prefix", "") for row in ds]

    success_probs = None
    if "success_prob" in ds.column_names:
        success_probs = [row["success_prob"] for row in ds]

    prompt_template = get_prompt_format(args.model_id)
    prompts = [
        prompt_template.replace("{input}", q) + (rp or "")
        for q, rp in zip(questions, reasoning_prefixes)
    ]

    sampling_params = build_sampling_params(args)

    llm = load_llm_with_retries(args)
    results = llm.generate(prompts, sampling_params)

    def resolve_prm_device(cli_value: str | None) -> str:
        """Return a safe device string for PRM ('cuda:0', 'cuda:1', or 'cpu')."""
        if not torch.cuda.is_available():
            return "cpu"
    
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible.strip():
            num_visible = len([x for x in visible.split(",") if x.strip() != ""])
        else:
            num_visible = torch.cuda.device_count()
    
        def clamp_to_valid(s: str) -> str:
            if not s.startswith("cuda:"):
                return "cpu"
            try:
                idx = int(s.split(":")[1])
            except Exception:
                return "cpu"
            if idx < 0 or idx >= num_visible:
                return "cuda:0" if num_visible >= 1 else "cpu"
            return s
     
        if cli_value is not None:
            return clamp_to_valid(cli_value)
    
        if num_visible >= 2:
            return "cuda:1"
        if num_visible >= 1:
            return "cuda:0"
        return "cpu"
        
    prm_device = resolve_prm_device(args.prm_device)
    print(f"[PRM] Target device resolved to: {prm_device}")
        
    prm = load_prm(args.prm_model_name, device=prm_device)
    print(f"[PRM] Loaded on {prm_device}")

    if args.prm_peft_dir:
        _N_BINS = 9
        prm.model.resize_token_embeddings(len(prm.tokenizer) + _N_BINS)
        from peft import PeftModelForCausalLM
        peft_model = PeftModelForCausalLM.from_pretrained(prm.model, args.prm_peft_dir)
        peft_model.eval()

    traces = [rp or "" for rp in reasoning_prefixes]
    prm_scores_per_prompt = score_with_prm_batched(
        prm, questions, traces, batch_size=args.prm_batch_size
    )

    output_data = []
    for i, (q, rp, res) in enumerate(zip(questions, reasoning_prefixes, results)):

        prompt_topk_logprobs = []
        prompt_lp_list = getattr(res, "prompt_logprobs", None)
        if prompt_lp_list:
            for pos_entry in prompt_lp_list:
                prompt_topk_logprobs.append(extract_topk_logprobs_list(pos_entry, k=args.topk_logprobs))

        token_entropy_per_pos = [get_entropy_from_logprobs(lps) for lps in prompt_topk_logprobs]
        token_confidence_per_pos = per_position_token_confidence(prompt_topk_logprobs)
        if len(token_confidence_per_pos) == 0:
            avg_trace_confidence = float("nan")
        else:
            avg_trace_confidence = float(np.nanmean(np.array(token_confidence_per_pos, dtype=np.float64)))

        rec = {
            "question": q,
            "reasoning_prefix": rp,
            "prm_scores": prm_scores_per_prompt[i],
            "confidence": {
                "method_token_entropy_per_pos": token_entropy_per_pos,              
                "method_token_confidence_per_pos": token_confidence_per_pos,        
                "method_avg_trace_confidence": avg_trace_confidence                 
            },
            "prompt_topk_logprobs": prompt_topk_logprobs                           
        }

        if success_probs is not None:
            rec["success_prob"] = success_probs[i]

        output_data.append(rec)

        if args.debug and i >= 2:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved {len(output_data)} prompts with PRM scores and confidence metrics to {out_path}")


if __name__ == "__main__":
    main()
    