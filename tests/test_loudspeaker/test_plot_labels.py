from __future__ import annotations

from loudspeaker.plot_labels import LabelSpec, normalized_labels, raw_labels


def test_label_spec_raw_includes_units():
    spec = LabelSpec("Velocity", "m/s", "v")
    assert spec.raw() == "Velocity [m/s]"
    assert spec.normalized() == "Velocity [m/s] (v/\\Sigma_v)"


def test_label_spec_derives_symbol_when_missing():
    spec = LabelSpec("Cone Current", "A")
    assert spec.normalized().startswith("Cone Current [A] (c/")


def test_label_spec_without_unit_inherits_plain_name():
    spec = LabelSpec("duty cycle")
    assert spec.raw() == "duty cycle"
    assert "(d/\\Sigma_d)" in spec.normalized()


def test_label_spec_falls_back_to_x_symbol():
    spec = LabelSpec("!!!")
    assert "(x/\\Sigma_x)" in spec.normalized()


def test_label_helpers_return_tuples():
    spec_a = LabelSpec("Position", "m", "x")
    spec_b = LabelSpec("Velocity", "m/s", "v")
    assert raw_labels(spec_a, spec_b) == ("Position [m]", "Velocity [m/s]")
    assert normalized_labels(spec_a, spec_b)[0].endswith("(x/\\Sigma_x)")
