"""Tests for voice_pd mobile deployment module (converter, protocol, benchmark)."""

import numpy as np
import pytest


@pytest.fixture
def dummy_model():
    """Small SpectrogramCNN (non-pretrained for fast tests)."""
    from bioagentics.voice_pd.deep.cnn_model import build_model

    return build_model(pretrained=False)


@pytest.fixture
def dummy_spectrogram():
    """Random spectrogram matching CNN input shape."""
    from bioagentics.voice_pd.mobile.converter import INPUT_N_MELS, INPUT_TIME_FRAMES

    rng = np.random.default_rng(42)
    return rng.random((3, INPUT_N_MELS, INPUT_TIME_FRAMES)).astype(np.float32)


@pytest.fixture
def protocol_audio():
    """Synthetic 30-second audio at 16kHz."""
    rng = np.random.default_rng(42)
    sr = 16_000
    t = np.linspace(0, 30.0, sr * 30, endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 200 * t) + rng.normal(0, 0.05, size=t.shape)
    return y.astype(np.float32)


# ── Converter tests ──


class TestExportOnnx:
    def test_creates_onnx_file(self, tmp_path, dummy_model):
        from bioagentics.voice_pd.mobile.converter import export_onnx

        onnx_path = export_onnx(model=dummy_model, output_path=tmp_path / "test.onnx")
        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"
        assert onnx_path.stat().st_size > 0

    def test_onnx_loadable(self, tmp_path, dummy_model):
        import onnx

        from bioagentics.voice_pd.mobile.converter import export_onnx

        onnx_path = export_onnx(model=dummy_model, output_path=tmp_path / "test.onnx")
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)


class TestRunPytorchInference:
    def test_returns_probability(self, dummy_model, dummy_spectrogram):
        from bioagentics.voice_pd.mobile.converter import run_pytorch_inference

        prob = run_pytorch_inference(dummy_model, dummy_spectrogram)
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_deterministic(self, dummy_model, dummy_spectrogram):
        from bioagentics.voice_pd.mobile.converter import run_pytorch_inference

        p1 = run_pytorch_inference(dummy_model, dummy_spectrogram)
        p2 = run_pytorch_inference(dummy_model, dummy_spectrogram)
        assert p1 == pytest.approx(p2)


class TestBenchmarkInference:
    def test_returns_timing_dict(self, dummy_model):
        from bioagentics.voice_pd.mobile.converter import benchmark_inference

        results = benchmark_inference(model=dummy_model, n_runs=3, warmup=1)
        assert "mean_sec" in results
        assert "median_sec" in results
        assert "meets_target" in results
        assert results["n_runs"] == 3
        assert results["mean_sec"] > 0

    def test_target_is_two_seconds(self, dummy_model):
        from bioagentics.voice_pd.mobile.converter import benchmark_inference

        results = benchmark_inference(model=dummy_model, n_runs=2, warmup=1)
        assert results["target_sec"] == 2.0


# ── Protocol tests ──


class TestValidateRecording:
    def test_valid_recording(self, protocol_audio):
        from bioagentics.voice_pd.mobile.protocol import validate_recording

        result = validate_recording(protocol_audio)
        assert result["is_valid"] is True
        assert len(result["issues"]) == 0
        assert result["duration_sec"] == pytest.approx(30.0, abs=0.1)

    def test_too_short(self):
        from bioagentics.voice_pd.mobile.protocol import validate_recording

        short = np.zeros(16_000 * 5, dtype=np.float32)  # 5 seconds
        short[0] = 0.1  # avoid silent rejection
        result = validate_recording(short)
        assert result["is_valid"] is False
        assert any("short" in i for i in result["issues"])

    def test_too_quiet(self):
        from bioagentics.voice_pd.mobile.protocol import validate_recording

        silent = np.zeros(16_000 * 30, dtype=np.float32)
        result = validate_recording(silent)
        assert result["is_valid"] is False
        assert any("quiet" in i for i in result["issues"])

    def test_clipping_detected(self):
        from bioagentics.voice_pd.mobile.protocol import validate_recording

        clipped = np.ones(16_000 * 30, dtype=np.float32)  # all clipped
        result = validate_recording(clipped)
        assert result["is_valid"] is False
        assert any("clipping" in i.lower() for i in result["issues"])


class TestSegmentRecording:
    def test_returns_vowel_and_sentence(self, protocol_audio):
        from bioagentics.voice_pd.mobile.protocol import segment_recording

        segments = segment_recording(protocol_audio)
        assert "vowel" in segments
        assert "sentence" in segments
        assert len(segments["vowel"]) > 0
        assert len(segments["sentence"]) > 0

    def test_vowel_duration(self, protocol_audio):
        from bioagentics.voice_pd.mobile.protocol import VOWEL_DURATION_SEC, segment_recording

        segments = segment_recording(protocol_audio)
        expected_samples = int(VOWEL_DURATION_SEC * 16_000)
        assert len(segments["vowel"]) == expected_samples

    def test_short_recording_no_crash(self):
        from bioagentics.voice_pd.mobile.protocol import segment_recording

        short = np.zeros(16_000 * 5, dtype=np.float32)
        segments = segment_recording(short)
        assert len(segments["vowel"]) > 0


class TestExtractVowelClip:
    def test_output_length(self):
        from bioagentics.voice_pd.mobile.protocol import extract_vowel_clip

        vowel = np.random.default_rng(0).random(16_000 * 10).astype(np.float32)
        clip = extract_vowel_clip(vowel)
        assert len(clip) == int(5.0 * 16_000)

    def test_short_vowel_pads(self):
        from bioagentics.voice_pd.mobile.protocol import extract_vowel_clip

        short = np.ones(8000, dtype=np.float32)  # 0.5 seconds
        clip = extract_vowel_clip(short)
        assert len(clip) == int(5.0 * 16_000)
        # Padded region should be zeros
        assert clip[0] == 0.0

    def test_exact_length_no_pad(self):
        from bioagentics.voice_pd.mobile.protocol import extract_vowel_clip

        exact = np.ones(int(5.0 * 16_000), dtype=np.float32)
        clip = extract_vowel_clip(exact)
        assert len(clip) == len(exact)
        np.testing.assert_array_equal(clip, exact)


class TestGetProtocolDescription:
    def test_has_required_fields(self):
        from bioagentics.voice_pd.mobile.protocol import get_protocol_description

        desc = get_protocol_description()
        assert desc["total_duration_sec"] == 30.0
        assert desc["sample_rate"] == 16_000
        assert len(desc["steps"]) == 3
        assert desc["steps"][0]["task"] == "sustained_vowel"
        assert desc["steps"][2]["task"] == "sentence_reading"

    def test_serializable(self):
        import json

        from bioagentics.voice_pd.mobile.protocol import get_protocol_description

        desc = get_protocol_description()
        serialized = json.dumps(desc)
        assert len(serialized) > 0


# ── Benchmark tests ──


class TestGenerateSpectrogram:
    def test_output_shape(self):
        from bioagentics.voice_pd.mobile.benchmark import _generate_spectrogram

        audio = np.random.default_rng(0).random(16_000 * 5).astype(np.float32)
        spec = _generate_spectrogram(audio)
        assert spec.ndim == 3
        assert spec.shape[0] == 3  # 3 channels
        assert spec.shape[1] == 128  # n_mels

    def test_normalized_range(self):
        from bioagentics.voice_pd.mobile.benchmark import _generate_spectrogram

        audio = np.random.default_rng(0).random(16_000 * 5).astype(np.float32) * 0.5
        spec = _generate_spectrogram(audio)
        assert spec.min() >= 0.0
        assert spec.max() <= 1.0 + 1e-6


class TestBenchmarkEndToEnd:
    def test_returns_results(self, tmp_path):
        from bioagentics.voice_pd.mobile.benchmark import benchmark_end_to_end

        results = benchmark_end_to_end(n_runs=2, warmup=1, output_dir=tmp_path)
        assert "total" in results
        assert "pipeline" in results
        assert results["total"]["median_sec"] > 0
        assert results["meets_target"] in (True, False)

    def test_saves_json(self, tmp_path):
        from bioagentics.voice_pd.mobile.benchmark import benchmark_end_to_end

        benchmark_end_to_end(n_runs=2, warmup=1, output_dir=tmp_path)
        assert (tmp_path / "mobile_benchmark_results.json").exists()

    def test_pipeline_breakdown(self, tmp_path):
        from bioagentics.voice_pd.mobile.benchmark import benchmark_end_to_end

        results = benchmark_end_to_end(n_runs=2, warmup=1, output_dir=tmp_path)
        pipeline = results["pipeline"]
        assert "clip_sec" in pipeline
        assert "spectrogram_sec" in pipeline
        assert "inference_sec" in pipeline
