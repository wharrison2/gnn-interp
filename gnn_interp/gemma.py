"""Gemma 2 2B + GemmaScope transcoders: generation, step localisation and error norms.

Convention shared with graph construction: for a step whose final token sits at sequence
position ``p``, the attribution graph is built on ``tokens[:p]`` and its rank-1 logit target is
``tokens[p]`` (the step's final token, which greedy decoding chose).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

MODELS = {"base": "google/gemma-2-2b", "instruct": "google/gemma-2-2b-it"}  # both use the base-model transcoders
MODEL_NAME = MODELS["base"]
# Gemma 2 chat turn (BOS is prepended by ``ensure_tokenized``); the reply ends with <end_of_turn>.
CHAT_TEMPLATE = "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n"
TRANSCODER_SET = "mntss/gemma-scope-transcoders"
OPERATORS = set("+-*/=()")
DIGIT, OPERATOR, OTHER = 0, 1, 2  # coarse token classes (never token identity)


@dataclass
class Step:
    """One arithmetic step: ``pos`` is the index of its final token; ``prev_line`` the line before."""

    pos: int
    line: str
    prev_line: str = ""


def load_model(device: str | None = None, dtype: torch.dtype = torch.float32, model_name: str = MODEL_NAME,
               **kwargs):
    """Load the circuit-tracer ``ReplacementModel`` (Gemma 2 2B, base or ``-it``, with GemmaScope transcoders).

    The transcoders were trained on the base model; on ``-it`` expect larger errors (measured by ``experiments``).
    ``kwargs`` go to ``ReplacementModel.from_pretrained`` (e.g. ``lazy_encoder``, ``lazy_decoder``).
    """
    from circuit_tracer import ReplacementModel

    return ReplacementModel.from_pretrained(model_name, TRANSCODER_SET, device=device, dtype=dtype, **kwargs)


def is_chat(model) -> bool:
    """Instruction-tuned model: prompts go in a chat turn and the reply ends at <end_of_turn>."""
    return model.cfg.model_name.endswith("-it")


def load_transcoders(device: str = "cpu"):
    """Load only the transcoder set (lazy decoders), e.g. to fetch decoder vectors on CPU."""
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    return load_transcoder_from_hub(TRANSCODER_SET, device=torch.device(device))[0]


@torch.no_grad()
def generate(model, prompt: str, max_new_tokens: int = 128) -> tuple[torch.Tensor, bool]:
    """Greedy-decode ``prompt`` (already chat-formatted for ``-it``); return (BOS + prompt + completion, complete).

    Base: truncated after the first blank line (``\\n\\n``), which ends the worked answer. Chat: truncated
    before <end_of_turn> / <eos> (not included). ``complete`` is False if ``max_new_tokens`` ran out first.
    """
    tokens = model.ensure_tokenized(prompt)  # prepends BOS
    chat = is_chat(model)
    eot = model.tokenizer.convert_tokens_to_ids("<end_of_turn>") if chat else None
    stop_ids = {eot, model.tokenizer.eos_token_id} if chat else set()
    # Early stop (TransformerLens takes one id): chat at <end_of_turn>; base at the single "\n\n" token if the
    # vocab has one (else it runs to max_new_tokens). Truncation below is the same either way.
    blank = model.tokenizer.encode("\n\n", add_special_tokens=False)
    stop = eot if chat else (blank[0] if len(blank) == 1 else None)
    out = model.generate(
        tokens[None], max_new_tokens=max_new_tokens, do_sample=False, return_type="tokens", verbose=False,
        **({"eos_token_id": stop} if stop is not None else {}),
    )[0].cpu()
    text = ""
    for i in range(len(tokens), len(out)):
        if int(out[i]) in stop_ids:
            return out[:i], True
        text += model.tokenizer.decode(out[i])
        if not chat and "\n\n" in text:
            return out[: i + 1], True
    return out, False


def step_final_positions(tokens: torch.Tensor, tokenizer, n_prompt: int, prefix: str = "=") -> list[Step]:
    """Locate steps (completion lines starting with ``prefix``) and their final token positions.

    Text is split into lines by decoding token-by-token (index 0 is BOS and skipped). A step's
    final token is the last digit-containing token on its line; ``prev_line`` is the previous
    non-empty line (prompt lines included, chat markers like ``<start_of_turn>model`` skipped), so the first
    step sees the problem statement.
    """
    lines, cur, last = [], "", None
    for i in range(1, len(tokens)):
        for j, part in enumerate(tokenizer.decode(tokens[i]).split("\n")):
            if j:  # a newline closed the current line
                lines.append((cur.strip(), last))
                cur, last = "", None
            cur += part
            if any(c.isdigit() for c in part):
                last = i
    lines.append((cur.strip(), last))

    steps, prev = [], ""
    for text, pos in lines:
        if text.startswith(prefix) and pos is not None and pos >= n_prompt:
            steps.append(Step(pos, text, prev))
        if text and not text.startswith("<"):
            prev = text
    return steps


@torch.no_grad()
def error_norms(model, tokens: torch.Tensor) -> torch.Tensor:
    """Per-(layer, position) transcoder error norms ``||mlp_out - reconstruction||``, (n_layers, n_pos).

    The model is causal, so norms for a prefix ``tokens[:p]`` are ``error_norms(model, tokens)[:, :p]``.
    """
    ctx = model.setup_attribution(tokens)
    return ctx.error_vectors.norm(dim=-1).float().cpu()


@torch.no_grad()
def residual_stream(model, tokens: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Residual stream at ``positions``: embeddings + every layer's output, (len(positions), n_layers + 1, d), fp16.

    For a step whose final token is at ``p``, use position ``p - 1``: the last input of its graph (``tokens[:p]``),
    whose output predicts ``tokens[p]``.
    """
    names = ["blocks.0.hook_resid_pre"] + [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    _, cache = model.run_with_cache(tokens[None].to(model.cfg.device), names_filter=lambda n: n in names)
    idx = torch.as_tensor(positions, dtype=torch.long)
    return torch.stack([cache[n][0, idx.to(cache[n].device)] for n in names], 1).half().cpu()


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

    Record: ``{problem_id, prompt, tokens, n_prompt, complete, steps: [{pos, line, prev_line, error_norms}]}``
    where ``error_norms`` is (n_layers, pos), matching the graph input ``tokens[:pos]``. For a chat model the
    prompt is wrapped in ``CHAT_TEMPLATE`` first and ``prompt`` stores the wrapped text.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for pid, prompt in prompts.items():
        if is_chat(model):
            prompt = CHAT_TEMPLATE.format(prompt)
        tokens, complete = generate(model, prompt, max_new_tokens)
        n_prompt = len(model.ensure_tokenized(prompt))
        norms = error_norms(model, tokens)
        steps = [
            {"pos": s.pos, "line": s.line, "prev_line": s.prev_line, "error_norms": norms[:, : s.pos].clone()}
            for s in step_final_positions(tokens, model.tokenizer, n_prompt)
        ]
        rec = {"problem_id": pid, "prompt": prompt, "tokens": tokens, "n_prompt": n_prompt, "complete": complete,
               "steps": steps}
        torch.save(rec, out_dir / f"{pid}.pt")
        records.append(rec)
    return records
