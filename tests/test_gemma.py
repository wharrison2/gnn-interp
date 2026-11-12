"""Pure-python helpers of gemma.py (no model download)."""
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
