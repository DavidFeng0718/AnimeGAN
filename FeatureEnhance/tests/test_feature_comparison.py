import pytest

from generate_feature_comparison import parse_args, percent_change


def test_percent_change_uses_base_as_denominator():
    assert percent_change(90.0, 100.0) == pytest.approx(-10.0)
    assert percent_change(120.0, 100.0) == pytest.approx(20.0)


def test_comparison_requires_explicit_attention_result():
    with pytest.raises(SystemExit):
        parse_args([])
    parsed = parse_args(["--attention-run", "run", "--attention-epoch", "200"])
    assert parsed.attention_run == "run"
    assert parsed.attention_epoch == 200
    assert parsed.se_multiseed_summary is None
