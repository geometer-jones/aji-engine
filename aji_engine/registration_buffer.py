from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Trace:
    input_embed: np.ndarray
    resonance: float
    output_embed: np.ndarray
    timestamp: int


@dataclass
class CompressedTraceSummary:
    mean_input_embed: np.ndarray
    mean_output_embed: np.ndarray
    resonance_histogram: np.ndarray
    count: int

    def mean_resonance(self) -> float:
        total = float(self.resonance_histogram.sum())
        if total <= 0.0:
            return 0.0
        bin_count = int(self.resonance_histogram.shape[0])
        edges = np.linspace(-1.0, 1.0, bin_count + 1, dtype=np.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])
        weighted = float(np.dot(self.resonance_histogram.astype(np.float32), centers))
        return weighted / total


class RegistrationBuffer:
    def __init__(
        self,
        *,
        capacity: int,
        resonance_high: float,
        resonance_low: float,
        histogram_bins: int = 8,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        if resonance_low >= resonance_high:
            raise ValueError("resonance_low must be smaller than resonance_high")

        self.capacity = capacity
        self.resonance_high = resonance_high
        self.resonance_low = resonance_low
        self.histogram_bins = histogram_bins
        self.traces: list[Trace] = []
        self.summary: Optional[CompressedTraceSummary] = None

    def __len__(self) -> int:
        return len(self.traces)

    def is_full(self) -> bool:
        return len(self.traces) >= self.capacity

    def add_trace(self, trace: Trace) -> None:
        frozen = Trace(
            input_embed=np.asarray(trace.input_embed, dtype=np.float32).copy(),
            resonance=float(trace.resonance),
            output_embed=np.asarray(trace.output_embed, dtype=np.float32).copy(),
            timestamp=int(trace.timestamp),
        )
        self.traces.append(frozen)
        overflow = max(0, len(self.traces) - self.capacity)
        if overflow:
            self._compress_oldest(overflow)

    def extend(self, traces: list[Trace]) -> None:
        for trace in traces:
            self.add_trace(trace)

    def drain(self) -> list[Trace]:
        traces: list[Trace] = []
        if self.summary is not None:
            traces.append(self._summary_to_trace(self.summary))
            self.summary = None
        traces.extend(self.traces)
        self.traces.clear()
        return traces

    def friction_zone(self) -> list[Trace]:
        return [
            trace
            for trace in self.traces
            if self.resonance_low < trace.resonance < self.resonance_high
        ]

    def aji_density(self) -> float:
        if not self.traces:
            return 0.0
        midpoint = 0.5 * (self.resonance_low + self.resonance_high)
        friction = self.friction_zone()
        if not friction:
            return 0.0
        total = sum(abs(trace.resonance - midpoint) for trace in friction)
        return total / len(self.traces)

    def _compress_oldest(self, count: int) -> None:
        oldest = [self.traces.pop(0) for _ in range(count)]
        input_stack = np.stack([trace.input_embed for trace in oldest], axis=0)
        output_stack = np.stack([trace.output_embed for trace in oldest], axis=0)
        resonances = np.asarray([trace.resonance for trace in oldest], dtype=np.float32)
        histogram, _bins = np.histogram(
            resonances,
            bins=self.histogram_bins,
            range=(-1.0, 1.0),
        )
        histogram = histogram.astype(np.float32)

        mean_input = input_stack.mean(axis=0)
        mean_output = output_stack.mean(axis=0)
        if self.summary is None:
            self.summary = CompressedTraceSummary(
                mean_input_embed=mean_input,
                mean_output_embed=mean_output,
                resonance_histogram=histogram,
                count=len(oldest),
            )
            return

        total = self.summary.count + len(oldest)
        self.summary.mean_input_embed = (
            self.summary.mean_input_embed * self.summary.count + mean_input * len(oldest)
        ) / total
        self.summary.mean_output_embed = (
            self.summary.mean_output_embed * self.summary.count + mean_output * len(oldest)
        ) / total
        self.summary.resonance_histogram = self.summary.resonance_histogram + histogram
        self.summary.count = total

    @staticmethod
    def _summary_to_trace(summary: CompressedTraceSummary) -> Trace:
        return Trace(
            input_embed=np.asarray(summary.mean_input_embed, dtype=np.float32).copy(),
            resonance=float(summary.mean_resonance()),
            output_embed=np.asarray(summary.mean_output_embed, dtype=np.float32).copy(),
            timestamp=-int(summary.count),
        )
