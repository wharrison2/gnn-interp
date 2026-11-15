"""Pure-python helpers of gemma.py (no model download)."""
import pytest
import torch

from gnn_interp.gemma import DIGIT, OPERATOR, OTHER, step_final_positions, token_class_table


class FakeTok:
    """One vocab entry per string; decode returns it verbatim."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        ids = torch.as_tensor(ids).reshape(-1)
        return "".join(self.vocab[int(i)] for i in ids)


def test_token_class_table():
    t = token_class_table(FakeTok(["<bos>", " 12", "+", " =", "x", " "]))
    assert t.tolist() == [OTHER, DIGIT, OPERATOR, OPERATOR, OTHER, OTHER]


def test_step_final_positions_and_prev_line():
    vocab = ["<bos>", "2", "+", "3", "\n", "=", "5", "\n", "=", "7", "\n\n"]
    tokens = torch.arange(len(vocab))
    steps = step_final_positions(tokens, FakeTok(vocab), n_prompt=5)
    assert [(s.pos, s.line) for s in steps] == [(6, "=5"), (9, "=7")]
    assert [s.prev_line for s in steps] == ["2+3", "=5"]


def test_residual_stream_matches_cache():
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    from gnn_interp.gemma import residual_stream

    torch.manual_seed(0)
    model = HookedTransformer(HookedTransformerConfig(n_layers=2, d_model=8, n_ctx=16, d_head=4, d_vocab=10,
                                                      act_fn="relu", device="cpu"))
    tokens = torch.randint(0, 10, (7,))
    r = residual_stream(model, tokens, [2, 5])
    _, cache = model.run_with_cache(tokens[None])
    assert r.shape == (2, 3, 8) and r.dtype == torch.float16
    assert torch.allclose(r[1, 0].float(), cache["blocks.0.hook_resid_pre"][0, 5], atol=1e-2)
    assert torch.allclose(r[0, 2].float(), cache["blocks.1.hook_resid_post"][0, 2], atol=1e-2)
