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


def test_step_prev_line_skips_chat_markers():
    vocab = ["<bos>", "2+3", "\n", "<end_of_turn>", "\n", "<start_of_turn>model", "\n", "=", "5", "\n"]
    steps = step_final_positions(torch.arange(len(vocab)), FakeTok(vocab), n_prompt=7)
    assert [(s.pos, s.line, s.prev_line) for s in steps] == [(8, "=5", "2+3")]


class FakeGenModel:
    """``generate`` appends ``completion`` ids; vocab ids: 0 <bos>, 1 '=5', 2 '\\n', 3 '\\n\\n', 4 <end_of_turn>."""

    def __init__(self, name, completion):
        from types import SimpleNamespace

        self.cfg = SimpleNamespace(model_name=name)
        self.completion = torch.tensor(completion)
        self.tokenizer = FakeTok(["<bos>", "=5", "\n", "\n\n", "<end_of_turn>"])
        self.tokenizer.convert_tokens_to_ids = lambda t: 4
        self.tokenizer.eos_token_id = 0
        self.tokenizer.encode = lambda text, add_special_tokens: [3] if text == "\n\n" else [9, 9]
        self.kwargs = None

    def ensure_tokenized(self, prompt):
        return torch.tensor([0, 1])

    def generate(self, tokens, max_new_tokens, **kw):
        self.kwargs = kw
        return torch.cat([tokens[0], self.completion])[None]


def test_generate_stops_base_and_chat():
    from gnn_interp.gemma import generate

    base = FakeGenModel("gemma-2-2b", [1, 2, 3, 1])
    tokens, complete = generate(base, "p")
    assert tokens.tolist() == [0, 1, 1, 2, 3] and complete and base.kwargs["eos_token_id"] == 3
    chat = FakeGenModel("gemma-2-2b-it", [1, 3, 1, 4, 1])  # a blank line does not end a chat reply
    tokens, complete = generate(chat, "p")
    assert tokens.tolist() == [0, 1, 1, 3, 1] and complete and chat.kwargs["eos_token_id"] == 4
    tokens, complete = generate(FakeGenModel("gemma-2-2b-it", [1, 2]), "p")
    assert tokens.tolist() == [0, 1, 1, 2] and not complete
