"""Speech-to-text engine choice: the Qwen3-ASR worker, its fallback, and run_poc.sh's --stt flags.

Runs on the PC (no MLX): the real qwen3_asr_worker.py is started with stand-in `mlx` and
`mlx_audio` modules on PYTHONPATH.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import muse_stt  # noqa: E402

WORKER = HERE.parent / "qwen3_asr_worker.py"
RUN_POC = HERE.parents[1] / "run_poc.sh"


@pytest.fixture
def fake_mlx(tmp_path, monkeypatch):
    """Stand-in mlx + mlx_audio: the 'model' answers with the clip length and the model name."""
    (tmp_path / "mlx").mkdir()
    (tmp_path / "mlx" / "__init__.py").write_text("")
    (tmp_path / "mlx" / "core.py").write_text("import numpy as np\narray = np.asarray\n")
    (tmp_path / "mlx_audio" / "stt").mkdir(parents=True)
    (tmp_path / "mlx_audio" / "__init__.py").write_text("")
    (tmp_path / "mlx_audio" / "stt" / "__init__.py").write_text(textwrap.dedent("""
        import types
        def load(name):
            if "missing" in name:
                raise FileNotFoundError("not cached")
            def generate(audio, language):
                assert language == "English"
                return types.SimpleNamespace(text=f" {len(audio)} samples {name.rsplit('/', 1)[-1]} ")
            return types.SimpleNamespace(generate=generate)
    """))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("MUSE_KOKORO_PYTHON", sys.executable)
    return tmp_path


def test_worker_transcribes_clips_over_the_pipe(fake_mlx):
    engine = muse_stt.Qwen3AsrEngine(muse_stt.qwen3_asr_command(muse_stt.QWEN3_ASR_MODEL))
    try:
        assert engine(np.zeros(8000, dtype=np.float32)) == "8000 samples Qwen3-ASR-0.6B-bf16"
        assert engine(np.ones(123, dtype=np.float64)) == "123 samples Qwen3-ASR-0.6B-bf16"  # any dtype in
    finally:
        engine.close()
    assert engine.proc.poll() is not None, "close() stops the worker"


def test_qwen3_asr_is_chosen_by_env_with_model_aliases(fake_mlx, monkeypatch):
    monkeypatch.setenv("MUSE_STT", "qwen3-asr")
    monkeypatch.setenv("MUSE_STT_MODEL", "1.7B")
    name, transcribe = muse_stt.make_transcriber()
    try:
        assert name == "qwen3-asr (Qwen3-ASR-1.7B-8bit)"
        assert transcribe(np.zeros(16, np.float32)) == "16 samples Qwen3-ASR-1.7B-8bit"
    finally:
        transcribe.close()
    assert muse_stt.qwen3_asr_model(None) == "mlx-community/Qwen3-ASR-0.6B-bf16"
    assert muse_stt.qwen3_asr_model("0.6b") == "mlx-community/Qwen3-ASR-0.6B-bf16"
    assert muse_stt.qwen3_asr_model("org/other") == "org/other"


def test_qwen3_asr_failure_falls_back_to_parakeet(fake_mlx, monkeypatch, caplog):
    monkeypatch.setenv("MUSE_STT", "qwen3-asr")
    monkeypatch.setenv("MUSE_STT_MODEL", "org/missing-model")
    loaded: list[str] = []

    def fake_parakeet(model_id):
        loaded.append(model_id)
        return lambda audio: "parakeet"

    monkeypatch.setattr(muse_stt, "parakeet_transcriber", fake_parakeet)
    caplog.set_level(logging.WARNING)
    name, transcribe = muse_stt.make_transcriber()
    assert name == "parakeet-mlx"
    assert loaded == [muse_stt.PARAKEET_MODEL], "the Qwen3 model id isn't passed to parakeet"
    assert any("Qwen3-ASR unavailable" in r.getMessage() for r in caplog.records)


def test_default_is_qwen3_asr_then_parakeet(fake_mlx, monkeypatch):
    monkeypatch.delenv("MUSE_STT", raising=False)
    monkeypatch.delenv("MUSE_STT_MODEL", raising=False)
    name, transcribe = muse_stt.make_transcriber()
    try:
        assert name == "qwen3-asr (Qwen3-ASR-0.6B-bf16)"
    finally:
        transcribe.close()
    # Qwen3-ASR can't load (no mlx_audio) -> parakeet; parakeet fails too -> whisper.
    monkeypatch.setenv("MUSE_KOKORO_PYTHON", "/nonexistent/python")
    monkeypatch.setattr(muse_stt, "parakeet_transcriber", lambda model_id: lambda audio: model_id)
    name, transcribe = muse_stt.make_transcriber()
    assert name == "parakeet-mlx" and transcribe(None) == muse_stt.PARAKEET_MODEL
    monkeypatch.setattr(muse_stt, "parakeet_transcriber", lambda model_id: (_ for _ in ()).throw(ImportError("x")))
    monkeypatch.setattr(muse_stt, "whisper_transcriber", lambda model_id: lambda audio: model_id)
    name, transcribe = muse_stt.make_transcriber()
    assert name == "mlx-whisper" and transcribe(None) == muse_stt.WHISPER_MODEL


def test_explicit_parakeet_skips_qwen3_asr(monkeypatch):
    monkeypatch.setenv("MUSE_STT", "parakeet")
    monkeypatch.setattr(muse_stt, "parakeet_transcriber", lambda model_id: lambda audio: model_id)
    monkeypatch.setattr(muse_stt, "qwen3_asr_transcriber", lambda *a: pytest.fail("Qwen3-ASR loaded"))
    name, _ = muse_stt.make_transcriber()
    assert name == "parakeet-mlx"


@pytest.mark.skipif(not RUN_POC.exists(), reason="run_poc.sh not here")
@pytest.mark.parametrize("args,err", [(["--stt", "qwen3"], "--stt must be"),
                                      (["--stt-model", "x;id"], "unsupported --stt-model")])
def test_run_poc_refuses_bad_stt_options(args, err):
    done = subprocess.run(["bash", str(RUN_POC), *args], capture_output=True, text=True, timeout=10)
    assert done.returncode == 2 and err in done.stderr


@pytest.mark.skipif(not RUN_POC.exists(), reason="run_poc.sh not here")
def test_run_poc_passes_stt_to_the_app_with_qwen3_asr_default():
    text = RUN_POC.read_text()
    assert "stt=qwen3-asr; stt_model=;" in text
    assert "-e MUSE_STT=$stt -e MUSE_STT_MODEL=$stt_model" in text
    start_app = next(l for l in text.splitlines() if l.startswith("start_job app "))
    assert "$tts_env" in start_app or "$tts_env" in text.split("start_job app ", 1)[1].split("\n", 2)[1]
    help_text = subprocess.run(["bash", str(RUN_POC), "--help"], capture_output=True, text=True, timeout=10).stdout
    assert "--stt qwen3-asr|parakeet|whisper" in help_text and "default qwen3-asr" in help_text and "--stt-model" in help_text
