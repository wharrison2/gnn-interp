"""Gemma 2 2B + GemmaScope transcoders: generation, step localisation and error norms.

Convention shared with graph construction: for a step whose final token sits at sequence
position ``p``, the attribution graph is built on ``tokens[:p]`` and its rank-1 logit target is
``tokens[p]`` (the step's final token, which greedy decoding chose).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

MODEL_NAME = "google/gemma-2-2b"
TRANSCODER_SET = "mntss/gemma-scope-transcoders"
OPERATORS = set("+-*/=()")
DIGIT, OPERATOR, OTHER = 0, 1, 2  # coarse token classes (never token identity)


@dataclass
class Step:
    """One arithmetic step: ``pos`` is the index of its final token in the full sequence."""

    pos: int
    line: str


def load_model(device: str | None = None, dtype: torch.dtype = torch.float32):
    """Load the circuit-tracer ``ReplacementModel`` (Gemma 2 2B with GemmaScope transcoders)."""
    from circuit_tracer import ReplacementModel

    return ReplacementModel.from_pretrained(MODEL_NAME, TRANSCODER_SET, device=device, dtype=dtype)


def load_transcoders(device: str = "cpu"):
    """Load only the transcoder set (lazy decoders), e.g. to fetch decoder vectors on CPU."""
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    return load_transcoder_from_hub(TRANSCODER_SET, device=torch.device(device))[0]


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 128) -> torch.Tensor:
    """Greedy-decode ``prompt``; return the 1-D token sequence (BOS + prompt + completion).

    Generation is truncated after the first blank line (``\\n\\n``), which ends the worked answer.
    """
    tokens = model.ensure_tokenized(prompt)  # prepends BOS
    out = model.generate(
        tokens[None], max_new_tokens=max_new_tokens, do_sample=False,
        return_type="tokens", verbose=False,
    )[0].cpu()
    text = ""
    for i in range(len(tokens), len(out)):
        text += model.tokenizer.decode(out[i])
        if "\n\n" in text:
            return out[: i + 1]
    return out


def step_final_positions(tokens: torch.Tensor, tokenizer, n_prompt: int, prefix: str = "=") -> list[Step]:
    """Locate steps (completion lines starting with ``prefix``) and their final token positions.

    Lines are split by decoding token-by-token; a line's final token is the last token before the
    one containing its terminating newline (or the last generated token).
    """
    steps, line, last = [], "", None
    for i in range(n_prompt, len(tokens)):
        piece = tokenizer.decode(tokens[i])
        if "\n" in piece:
            head = piece.split("\n", 1)[0]
            if head.strip():  # newline glued onto content: this token ends the line
                line, last = line + head, i
            if line.strip().startswith(prefix) and last is not None:
                steps.append(Step(last, line.strip()))
            line, last = piece.rsplit("\n", 1)[1], (i if piece.rsplit("\n", 1)[1].strip() else None)
        else:
            line += piece
            last = i if piece.strip() else last
    if line.strip().startswith(prefix) and last is not None:
        steps.append(Step(last, line.strip()))
    return steps


@torch.no_grad()
def error_norms(model, tokens: torch.Tensor) -> torch.Tensor:
    """Per-(layer, position) transcoder error norms ``||mlp_out - reconstruction||``, (n_layers, n_pos).

    The model is causal, so norms for a prefix ``tokens[:p]`` are ``error_norms(model, tokens)[:, :p]``.
    """
    ctx = model.setup_attribution(tokens)
    return ctx.error_vectors.norm(dim=-1).float().cpu()


def token_class_table(tokenizer) -> torch.Tensor:
    """Map every vocab id to a coarse class: DIGIT, OPERATOR (``+-*/=()``) or OTHER."""
    table = torch.full((len(tokenizer),), OTHER, dtype=torch.long)
    for tid in range(len(tokenizer)):
        s = tokenizer.decode([tid]).strip()
        if s and s.isdigit():
            table[tid] = DIGIT
        elif s and set(s) <= OPERATORS:
            table[tid] = OPERATOR
    return table


def run_problems(model, prompts: dict[str, str], out_dir: str | Path, max_new_tokens: int = 128) -> list[dict]:
    """Generate on each prompt and save one record per problem to ``out_dir/{problem_id}.pt``.

    Record: ``{problem_id, prompt, tokens, n_prompt, steps: [{pos, line, error_norms}]}`` where
    ``error_norms`` is (n_layers, pos), matching the graph input ``tokens[:pos]``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pid, prompt in prompts.items():
        tokens = generate(model, prompt, max_new_tokens)
        n_prompt = len(model.ensure_tokenized(prompt))
        norms = error_norms(model, tokens)
        steps = [
            {"pos": s.pos, "line": s.line, "error_norms": norms[:, : s.pos].clone()}
            for s in step_final_positions(tokens, model.tokenizer, n_prompt)
        ]
        rec = {"problem_id": pid, "prompt": prompt, "tokens": tokens, "n_prompt": n_prompt, "steps": steps}
        torch.save(rec, out_dir / f"{pid}.pt")
        records.append(rec)
    return records
