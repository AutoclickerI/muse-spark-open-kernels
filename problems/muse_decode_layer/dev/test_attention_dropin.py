"""Structural test for the v9-derived attention drop-in."""

from pathlib import Path

from attention_dropin import SHAPES, build


def test_shape_contract_and_generated_copy(tmp_path: Path):
    assert len(SHAPES) == 12
    assert len(set(SHAPES)) == 12
    out = build(tmp_path / "cand_r3_v9_attention_dropin.py")
    text = out.read_text()
    assert "ATTENTION_DROPIN_SHAPES" in text
    assert "def attention_dropin_choice" in text
    assert "o_and_pool" not in text
    for name in ("ATTN_X_CUDA", "ATTN_X_CPP", "ATTN_G_CUDA", "ATTN_P8_CUDA"):
        assert f"{name} =" in text
