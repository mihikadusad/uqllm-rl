import os
import argparse
import numpy as np
import json
import random
import math
from typing import List, Any, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from prm import load_prm
from utils.llm import load_llm_with_retries, get_prompt_format, get_sampling_params

def parse_args():
    parser = argparse.ArgumentParser(description="PRM confidence experiment with vLLM (no generation).")

    parser.add_argument(
        "--chunk",
        type=int,
        default=0
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="math500train",
        choices=["math500", "math500train", "aime2024", "aime2025", "aime2025-2"],
        help="Which dataset to run on."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="young-j-park/prm_calibration",
        help="HuggingFace dataset name to use."
    )
    parser.add_argument(
        "--debug", 
        action="store_true"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="auto"
    )
    parser.add_argument(
        "--enable_chunked_prefill",
        action="store_true"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.2  # because PRM
    )
    parser.add_argument(
        "--hf_token", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--lb_idx", 
        type=int, 
        default=None,
        help="Optional: if your PRM outputs quantiles, choose lower-bound index (e.g., 0,1,2)."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1
    )
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        choices=["Qwen/Qwen2.5-Math-7B-Instruct","meta-llama/Llama-3.2-1B-Instruct"]
    )
    parser.add_argument(
        "--n_generations",
        type=int,
        default=1
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./mihika_experiment_results"
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="experiment_conf_all_types.json"
    )
    parser.add_argument(
        "--prm_batch_size", 
        type=int, 
        default=1,
        help="Batch size for PRM scoring."
    )
    parser.add_argument(
        "--prm_device", 
        type=str, 
        default=None,
        help="Device for PRM, e.g. cuda:1 or cpu. Default: cuda:1 if available else cuda:0/cpu."
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
    parser.add_argument(
        "--prm_peft_dir", 
        type=str, 
        default=None,
        help="Optional PEFT adapter dir for PRM (if you use calibrated heads)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    parser.add_argument(
        "--subset_size", 
        type=int, 
        default=None,
        help="Random subset size for faster runs."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )
    parser.add_argument(
        "--topk_logprobs", 
        type=int, 
        default=20,
        help="Top-k to collect for logprobs at each position."
    )
    parser.add_argument(
        "--total_chunks",
        type=int,
        default=1
    )
    parser.add_argument(
        "--use_cuda_graph",
        action="store_true"
    )

    return parser.parse_args()


def get_entropy_from_logprobs(logprobs: List[float]) -> float:
    if not logprobs:
        return None
    a = np.array(logprobs, dtype=np.float64)
    a_max = np.max(a)
    exp_a = np.exp(a - a_max)
    Z = np.sum(exp_a)
    if Z == 0.0 or not np.isfinite(Z):
        return None
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
            out.append(None)
        else:
            out.append(-float(np.mean(lp_list)))
    return out


def get_mean_ignoring_none(my_lis: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in my_lis if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def allocate_tokens_by_chars(segment_lengths: List[int], total_tokens: int) -> List[int]:
    if total_tokens <= 0 or sum(segment_lengths) <= 0:
        return [0] * len(segment_lengths)
    shares = [total_tokens * (L / sum(segment_lengths)) for L in segment_lengths]
    counts = [int(math.floor(s)) for s in shares]
    r = total_tokens - sum(counts)
    frac = sorted(enumerate([s - c for s, c in zip(shares, counts)]), key=lambda x: x[1], reverse=True)
    for i in range(r):
        counts[frac[i % len(frac)][0]] += 1
    return counts


def reasoning_only_confidences(token_conf_all: List[Optional[float]],
                               pre_reasoning_text: str,
                               reasoning_steps: List[str]) -> List[Optional[float]]:
    per_pos = token_conf_all[1:]
    seg_lengths = [len(pre_reasoning_text)] + [len(s) for s in reasoning_steps]
    seg_tokens = allocate_tokens_by_chars(seg_lengths, len(per_pos))
    start = seg_tokens[0]
    length = sum(seg_tokens[1:])
    return per_pos[start:start+length]


def group_confidence_sliding(reasoning_conf: List[Optional[float]], window: int) -> List[Optional[float]]:
    out = []
    for i in range(len(reasoning_conf)):
        L = max(0, i - window + 1)
        out.append(get_mean_ignoring_none(reasoning_conf[L:i+1]))
    return out


def bottom10_group_confidence(group_confs: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in group_confs if v is not None]
    if not vals:
        return None
    k = max(1, math.ceil(0.10 * len(vals)))
    return sum(sorted(vals)[:k]) / k


def lowest_group_confidence(group_confs: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in group_confs if v is not None]
    return min(vals) if vals else None


def tail_confidence(reasoning_conf: List[Optional[float]], tail_tokens: int) -> Optional[float]:
    vals = [v for v in reasoning_conf if v is not None]
    if not vals:
        return None
    m = max(1, min(tail_tokens, len(vals)))
    return get_mean_ignoring_none(vals[-m:])


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

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    #dataset_path = f"{args.dataset}/{args.model_id}/data.json"
    dataset_path = "aime2024/Llama-3.2-1B-Instruct/data.json"
    ds = load_dataset(args.dataset_name, data_files=dataset_path, split="train")

    if args.subset_size and args.subset_size < len(ds):
        indices = random.sample(range(len(ds)), args.subset_size)
        ds = ds.select(indices)

    questions = [row["question"] for row in ds]
    nonsplit_reasoning_prefixes = [row.get("reasoning_prefix", "") for row in ds]

    print('Getting questions and reasoning prefixes:')
    reasoning_prefixes = []
    for prefix in nonsplit_reasoning_prefixes:
        reasoning_prefixes += [prefix.split('\n\n')] # List[List[str]]    
    
    success_probs = None
    try:
        success_probs = [row["success_prob"] for row in ds]
    except:
        success_probs = [0]*len(ds)

    # len(questions) = len(reasoning_prefixes) = len(success_probs)

    prompt_template = get_prompt_format(args.model_id)
    prompts = [
        prompt_template.replace("{input}", q) + ('\n\n'.join(reasoning_steps) or "")
        for q, reasoning_steps in zip(questions, reasoning_prefixes)
    ]

    sampling_params = build_sampling_params(args)

    llm = load_llm_with_retries(args)
    results = llm.generate(prompts, sampling_params)

    output_data = []
    for i, (q, prefixes, res) in enumerate(zip(questions, reasoning_prefixes, results)):

        prompt_topk_logprobs = []
        prompt_lp_list = getattr(res, "prompt_logprobs", None)
        if prompt_lp_list:
            for pos_entry in prompt_lp_list:
                prompt_topk_logprobs.append(extract_topk_logprobs_list(pos_entry, k=args.topk_logprobs))

        token_entropy_per_pos = [get_entropy_from_logprobs(lps) for lps in prompt_topk_logprobs]
        token_confidence_per_pos = per_position_token_confidence(prompt_topk_logprobs)
        if len(token_confidence_per_pos) == 0:
            avg_trace_confidence = None
        else:
            avg_trace_confidence = float(np.nanmean(np.array(token_confidence_per_pos, dtype=np.float64)))

        pre_reasoning = get_prompt_format(args.model_id).replace("{input}", q)

        reasoning_conf = reasoning_only_confidences(
            token_confidence_per_pos,
            pre_reasoning_text=pre_reasoning,
            reasoning_steps=prefixes
        )

        group_window = 128
        tail_len = 128
        
        group_confs = group_confidence_sliding(reasoning_conf, window=group_window)
        conf_bottom10 = bottom10_group_confidence(group_confs)
        conf_lowest = lowest_group_confidence(group_confs)
        conf_tail = tail_confidence(reasoning_conf, tail_tokens=tail_len) 
        
        rec = {
            "question": q,
            "reasoning_prefix": prefixes,
            "confidence": {
                "token_entropy_per_pos": token_entropy_per_pos,              
                "token_confidence_per_pos": token_confidence_per_pos,        
                "avg_trace_confidence": avg_trace_confidence,
                "reasoning_group_conf_per_pos": group_confs,
                "reasoning_bottom10_group_confidence": conf_bottom10,
                "reasoning_lowest_group_confidence": conf_lowest,
                "reasoning_tail_confidence": conf_tail,
            },
            "prompt_topk_logprobs": prompt_topk_logprobs,
        }
        
        if success_probs is not None:
            rec["success_prob"] = success_probs[i]

        output_data.append(rec)

        if args.debug and i >= 2:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"[OK] Saved {len(output_data)} prompts with confidence metrics to {out_path}")


if __name__ == "__main__":
    main()