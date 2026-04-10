import numpy as np

from aji_engine import RegistrationBuffer, Trace


def test_overflow_compresses_oldest_traces() -> None:
    buffer = RegistrationBuffer(
        capacity=2,
        resonance_high=0.8,
        resonance_low=0.2,
        histogram_bins=4,
    )

    traces = [
        Trace(np.array([1.0, 0.0], dtype=np.float32), 0.3, np.array([1.0, 0.0, 0.0], dtype=np.float32), 0),
        Trace(np.array([0.0, 1.0], dtype=np.float32), 0.5, np.array([0.0, 1.0, 0.0], dtype=np.float32), 1),
        Trace(np.array([1.0, 1.0], dtype=np.float32), 0.7, np.array([0.0, 0.0, 1.0], dtype=np.float32), 2),
        Trace(np.array([-1.0, 0.0], dtype=np.float32), 0.1, np.array([1.0, 1.0, 0.0], dtype=np.float32), 3),
    ]
    buffer.extend(traces)

    assert len(buffer) == 2
    assert buffer.summary is not None
    assert buffer.summary.count == 2
    assert int(buffer.summary.resonance_histogram.sum()) == 2
    assert buffer.aji_density() > 0.0


def test_drain_includes_summary_as_synthetic_trace_and_clears_it() -> None:
    buffer = RegistrationBuffer(
        capacity=1,
        resonance_high=0.8,
        resonance_low=0.2,
        histogram_bins=4,
    )

    first = Trace(np.array([1.0, 0.0], dtype=np.float32), 0.3, np.array([0.0, 1.0], dtype=np.float32), 0)
    second = Trace(np.array([0.0, 1.0], dtype=np.float32), 0.7, np.array([1.0, 0.0], dtype=np.float32), 1)
    buffer.extend([first, second])

    drained = buffer.drain()

    assert len(drained) == 2
    np.testing.assert_allclose(drained[0].input_embed, first.input_embed)
    np.testing.assert_allclose(drained[0].output_embed, first.output_embed)
    assert drained[0].timestamp == -1
    assert buffer.summary is None
    assert len(buffer) == 0
