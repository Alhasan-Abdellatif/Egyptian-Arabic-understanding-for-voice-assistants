import argparse
import json

from lahja.data.schema import Example
from lahja.eval.run_eval import evaluate_set, fingerprint

GOLDS = [
    Example(id="a", utt="صحيني بكرة", intent="alarm_set", slots=[]),
    Example(id="b", utt="شغل اغاني", intent="play_music", slots=[]),
]


class FakeBackend:
    """Counts how many predictions were actually generated."""

    def __init__(self, intent="alarm_set"):
        self.calls = 0
        self.intent = intent

    def generate(self, batch):
        self.calls += len(batch)
        return [json.dumps({"intent": self.intent, "slots": {}})] * len(batch)


def _args(adapter=None, model="m", prompt="sft", shots=0):
    return argparse.Namespace(
        backend="mlx", model=model, adapter=adapter, prompt=prompt, shots=shots
    )


def test_fingerprint_tracks_adapter_weights(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"x" * 10)
    (adapter / "0000800_adapters.safetensors").write_bytes(b"y" * 99)  # checkpoint: ignored
    before = fingerprint(_args(adapter=str(adapter)))
    assert [w[0] for w in before["weights"]] == ["adapters.safetensors"]

    (adapter / "adapters.safetensors").write_bytes(b"x" * 20)  # "retrained"
    assert fingerprint(_args(adapter=str(adapter))) != before
    assert fingerprint(_args(adapter=str(adapter), prompt="full")) != fingerprint(
        _args(adapter=str(adapter))
    )


def test_cached_predictions_are_reused_then_invalidated(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapters.safetensors").write_bytes(b"x" * 10)
    out = tmp_path / "run" / "egy_test"
    args = _args(adapter=str(adapter))

    first = FakeBackend()
    evaluate_set(first, GOLDS, "SYS", [], out, 8, fingerprint(args))
    assert first.calls == 2
    assert json.loads((out / "scores.json").read_text())["exact_match"] == 0.5

    second = FakeBackend()  # same weights: nothing regenerated
    evaluate_set(second, GOLDS, "SYS", [], out, 8, fingerprint(args))
    assert second.calls == 0

    (adapter / "adapters.safetensors").write_bytes(b"x" * 20)  # retrained
    third = FakeBackend(intent="play_music")
    scores = evaluate_set(third, GOLDS, "SYS", [], out, 8, fingerprint(args))
    assert third.calls == 2
    assert scores["exact_match"] == 0.5  # now the other example is the correct one
    assert json.loads((out / "predictions.jsonl").read_text().splitlines()[0])["raw"].count(
        "play_music"
    )


def test_fresh_flag_forces_regeneration(tmp_path):
    out = tmp_path / "run" / "egy_test"
    args = _args()
    evaluate_set(FakeBackend(), GOLDS, "SYS", [], out, 8, fingerprint(args))
    again = FakeBackend()
    evaluate_set(again, GOLDS, "SYS", [], out, 8, fingerprint(args), fresh=True)
    assert again.calls == 2
