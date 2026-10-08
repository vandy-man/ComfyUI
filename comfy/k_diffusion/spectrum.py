"""Spectrum feature forecasting: training-free step skipping with replay.

Adapted from Wan2GP (GPL-3.0) shared/steps_skipping/spectrum.py by DeepBeepMeep,
full-anchor capture mode only, for use by sample_euler_spectrum.
"""

import bisect
import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SpectrumConfig:
    degree: int = 1
    ridge_lambda: float = 0.1
    default_blend_weight: float = 0.5
    max_history: int = 8
    window_size: float = 2.0
    flex_window: float = 0.75
    max_consecutive_forecasts: int = 1
    chunk_bytes: int = 32 * 1024 * 1024
    validation_samples: int = 16 * 1024
    tail_actual_steps: int = 1


@dataclass
class _HistoryEntry:
    step: int
    coordinate: float
    feature_flat: torch.Tensor


class SpectrumFeatureForecaster:
    """Archive per-step features on CPU, forecast skipped steps, then replay a
    transformer-free smoothing pass over the fully archived trajectory."""

    def __init__(self, sigmas, start_step=0, config=None):
        self.config = SpectrumConfig() if config is None else config
        self.degree = self.config.degree
        self.min_fit_points = self.degree + 1
        evaluated = sigmas.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        sigma_min, sigma_max = float(evaluated.min()), float(evaluated.max())
        if sigma_max > sigma_min:
            evaluated = 2.0 * (evaluated - sigma_min) / (sigma_max - sigma_min) - 1.0
        else:
            evaluated.zero_()
        self.coordinates = tuple(float(value) for value in evaluated)
        self.total_steps = len(self.coordinates)
        self.warmup_steps = max(1, int(start_step))
        self._history = []
        self._anchors = []
        self._decisions = []
        self._replay_weights = {}
        self._feature_shape = None
        self._feature_dtype = None
        self._next_step = 0
        self._current_step = -1
        self._current_coordinate = 0.0
        self._actual = None
        self._logical_actual = None
        self._bootstrap = False
        self._observed = False
        self._adaptive_recompute = False
        self._current_window = self.config.window_size
        self._consecutive_forecasts = 0
        self._phase = "capture"

    @property
    def forecasting(self):
        if self._actual is None:
            raise RuntimeError("Spectrum step has not started")
        return not self._actual

    @property
    def replaying(self):
        return self._phase == "replay"

    def begin_step(self, step):
        if self._actual is not None or step != self._next_step:
            raise RuntimeError("Spectrum step order is inconsistent with the denoising schedule")
        self._current_step = step
        self._current_coordinate = self.coordinates[step]
        self._bootstrap = False
        self._adaptive_recompute = False
        if self.replaying:
            self._logical_actual = self._decisions[step]
            self._actual = False
        else:
            tail_start = max(0, self.total_steps - self.config.tail_actual_steps)
            self._bootstrap = (step == 1 and self.warmup_steps <= 1 and
                               len(self._history) == 1 and self.degree == 1)
            if step < self.warmup_steps or step >= tail_start or (len(self._history) < self.min_fit_points and not self._bootstrap):
                self._actual = True
            elif self._bootstrap:
                self._actual = False
            else:
                interval = max(1, math.floor(self._current_window))
                self._actual = (self._consecutive_forecasts + 1) % interval == 0
                self._adaptive_recompute = self._actual
                if not self._actual and self._consecutive_forecasts >= self.config.max_consecutive_forecasts:
                    self._actual = True
                    self._adaptive_recompute = False
            self._logical_actual = self._actual
        self._observed = False

    def observe(self, feature):
        if not self._actual or self._observed or self.replaying:
            raise RuntimeError("Spectrum received an unexpected actual feature")
        feature_shape = tuple(feature.shape)
        if self._feature_shape is None:
            self._feature_shape, self._feature_dtype = feature_shape, feature.dtype
        elif feature_shape != self._feature_shape or feature.dtype != self._feature_dtype:
            raise RuntimeError("Spectrum feature topology changed during forecasting")
        archived = feature.detach().to(device="cpu", dtype=feature.dtype, copy=True).contiguous().reshape(-1)
        entry = _HistoryEntry(self._current_step, self._current_coordinate, archived)
        self._history.append(entry)
        self._anchors.append(entry)
        if len(self._history) > self.config.max_history:
            self._history.pop(0)
        self._observed = True

    def _chebyshev_design(self, coordinates):
        x = coordinates.reshape(-1, 1).to(dtype=torch.float32)
        columns = [torch.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            columns.append(2.0 * x * columns[-1] - columns[-2])
        return torch.cat(columns[:self.degree + 1], dim=1)

    def _spectral_weights(self, entries, coordinate, affine=False):
        coordinates = torch.tensor([entry.coordinate for entry in entries], device="cpu", dtype=torch.float32)
        design = self._chebyshev_design(coordinates)
        gram = design.T @ design + self.config.ridge_lambda * torch.eye(self.degree + 1, device="cpu", dtype=torch.float32)
        phi = self._chebyshev_design(torch.tensor([coordinate], device="cpu", dtype=torch.float32))
        weights = (phi @ torch.linalg.solve(gram, design.T)).reshape(-1)
        if affine:
            weights.add_((1.0 - float(weights.sum())) / weights.numel())
        return weights

    @staticmethod
    def _causal_linear_weights(entries, coordinate):
        weights = torch.zeros(len(entries), device="cpu", dtype=torch.float32)
        previous, latest = entries[-2].coordinate, entries[-1].coordinate
        ratio = (coordinate - latest) / (latest - previous)
        weights[-2], weights[-1] = -ratio, 1.0 + ratio
        return weights

    def _predict_segment(self, result_flat, start, end, entries, weights, device):
        chunk_elements = max(1024, self.config.chunk_bytes // 4)
        for offset in range(start, end, chunk_elements):
            length = min(chunk_elements, end - offset)
            accumulator = torch.zeros(length, device=device, dtype=torch.float32)
            for weight, entry in zip(weights, entries):
                if weight:
                    source = entry.feature_flat.narrow(0, offset, length).to(device=device, dtype=torch.float32)
                    accumulator.add_(source, alpha=float(weight))
                    del source
            result_flat.narrow(0, offset, length).copy_(accumulator.to(result_flat.dtype))
            del accumulator

    def predict(self, device, dtype):
        if self._actual or self._feature_shape is None:
            raise RuntimeError("Spectrum forecast requested without sufficient feature history")
        result = torch.empty(self._feature_shape, device=device, dtype=dtype)
        result_flat = result.reshape(-1)
        total = result_flat.numel()
        if self.replaying:
            entries = self._anchors
            if self._logical_actual:
                weights = torch.zeros(len(entries), device="cpu", dtype=torch.float32)
                anchor_index = next(index for index, entry in enumerate(entries) if entry.step == self._current_step)
                weights[anchor_index] = 1.0
                self._predict_segment(result_flat, 0, total, entries, weights, device)
            else:
                for weights in self._replay_weights[self._current_step]:
                    self._predict_segment(result_flat, 0, total, entries, weights, device)
        else:
            entries = self._history
            if self._bootstrap:
                weights = torch.ones(1, device="cpu", dtype=torch.float32)
            else:
                weights = self._causal_linear_weights(entries, self._current_coordinate)
            self._predict_segment(result_flat, 0, total, entries, weights, device)
        return result

    def _validation_scores(self):
        total = self._anchors[0].feature_flat.numel()
        sample_count = min(self.config.validation_samples, total)
        indices = torch.div(torch.arange(sample_count, device="cpu", dtype=torch.int64) * total,
                            sample_count, rounding_mode="floor")
        samples = torch.stack([entry.feature_flat[indices].to(torch.float32) for entry in self._anchors])
        scores = [None] * len(self._anchors)
        if len(self._anchors) < max(3, self.degree + 2):
            return scores
        for target_index in range(1, len(self._anchors) - 1):
            retained = [index for index in range(len(self._anchors)) if index != target_index]
            spectral = self._spectral_weights([self._anchors[index] for index in retained],
                                              self._anchors[target_index].coordinate, affine=True)
            spectral_prediction = spectral @ samples[retained]
            left, target, right = self._anchors[target_index - 1:target_index + 2]
            ratio = (target.coordinate - left.coordinate) / (right.coordinate - left.coordinate)
            local_prediction = torch.lerp(samples[target_index - 1], samples[target_index + 1], ratio)
            actual = samples[target_index]
            spectral_rms = float(torch.sqrt(torch.mean((spectral_prediction - actual) ** 2)))
            local_rms = float(torch.sqrt(torch.mean((local_prediction - actual) ** 2)))
            actual_rms = float(torch.sqrt(torch.mean(actual * actual)))
            epsilon = max(actual_rms * 1e-6, torch.finfo(torch.float32).eps)
            scores[target_index] = 0.0 if spectral_rms <= epsilon and local_rms <= epsilon else spectral_rms / max(local_rms, epsilon)
        return scores

    @staticmethod
    def _interval_score(scores, position):
        nearby = [scores[index] for index in (position - 1, position) if scores[index] is not None]
        return max(nearby, default=1.0)

    def _build_replay_weights(self):
        anchor_steps = [entry.step for entry in self._anchors]
        scores = self._validation_scores()
        for step, actual in enumerate(self._decisions):
            if actual:
                continue
            coordinate = self.coordinates[step]
            spectral = self._spectral_weights(self._anchors, coordinate, affine=True)
            position = bisect.bisect_left(anchor_steps, step)
            if position == 0 or position == len(self._anchors):
                raise RuntimeError("Spectrum replay forecast is missing a bracketing actual anchor")
            left, right = self._anchors[position - 1], self._anchors[position]
            ratio = (coordinate - left.coordinate) / (right.coordinate - left.coordinate)
            local = torch.zeros(len(self._anchors), device="cpu", dtype=torch.float32)
            local[position - 1], local[position] = 1.0 - ratio, ratio
            blend = self.config.default_blend_weight / max(1.0, self._interval_score(scores, position))
            self._replay_weights[step] = (blend * spectral + (1.0 - blend) * local,)

    def finish_step(self):
        if self._actual is None or (self._actual and not self._observed):
            raise RuntimeError("Spectrum step finished without its required feature")
        if not self.replaying:
            self._decisions.append(bool(self._actual))
            if self._actual:
                self._consecutive_forecasts = 0
                if self._adaptive_recompute:
                    self._current_window = min(round(self._current_window + self.config.flex_window, 6),
                                               float(self.config.max_history))
            else:
                self._consecutive_forecasts += 1
        self._next_step += 1
        self._actual = None
        self._logical_actual = None
        self._bootstrap = False
        self._observed = False
        self._adaptive_recompute = False

    def complete_capture(self):
        if self._phase != "capture" or self._next_step != self.total_steps:
            raise RuntimeError("Spectrum full-anchor capture is incomplete")
        actual_steps = [step for step, actual in enumerate(self._decisions) if actual]
        if actual_steps != [entry.step for entry in self._anchors]:
            raise RuntimeError("Spectrum capture decisions do not match the full anchor archive")
        self._build_replay_weights()
        self._phase = "captured"

    def start_replay(self):
        if self._phase != "captured":
            raise RuntimeError("Spectrum replay requested before full-anchor capture completed")
        self._phase = "replay"
        self._next_step = 0
        self._current_step = -1
        self._actual = None
        self._logical_actual = None
        self._observed = False
