# Generalized CLRS probe extraction utilities
# Supports ANY probe shape: node, edge, scalar, pointer, etc.
# Includes robust handling for probes whose ground-truth is stored as indices
# while predictions are stored as distributions (common for POINTER/CATEGORICAL/MASK_ONE).

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence
import json
import numpy as np

try:
  import jax
  _HAS_JAX = True
except Exception:
  _HAS_JAX = False

from clrs._src import probing
from clrs._src import samplers
from clrs._src import specs
from absl import logging

# Use CLRS' official correctness evaluation for hints.
from clrs._src import evaluation


def _to_numpy(x: Any) -> np.ndarray:
  """Converts JAX/NumPy/tensor arrays to a NumPy array."""
  if isinstance(x, np.ndarray):
    return x
  if _HAS_JAX:
    try:
      if isinstance(x, jax.Array):
        return np.asarray(x)
    except Exception:
      try:
        from jax.interpreters.xla import DeviceArray
        if isinstance(x, DeviceArray):
          return np.asarray(x)
      except Exception:
        pass
  return np.asarray(x)


def _truth_to_batch_first(dp: probing.DataPoint, lengths_list: List[int]) -> np.ndarray:
  """Converts ground-truth hint to batch-first array [B, T, ...]."""
  arr = _to_numpy(dp.data)
  lengths_len = len(lengths_list)

  # If looks like [T,B,...] with B matching lengths, move axis to [B,T,...]
  if arr.ndim >= 2 and lengths_len > 0 and arr.shape[1] == lengths_len:
    arr = np.moveaxis(arr, 1, 0)
  else:
    # No batch axis -> inject B=1 at front, keep time as axis=1
    arr = np.expand_dims(arr, axis=0)

  # Remove trailing singleton dims after T
  while arr.ndim > 2 and arr.shape[-1] == 1:
    arr = arr.squeeze(axis=-1)

  return arr


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
  x = x - np.max(x, axis=axis, keepdims=True)
  e = np.exp(x)
  s = np.sum(e, axis=axis, keepdims=True)
  s = np.where(s == 0, 1.0, s)
  return e / s


def _sigmoid(x: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-x))


def _looks_like_probabilities(x: np.ndarray) -> bool:
  """Heuristic: does x already look like probabilities in [0,1]?

  Used to avoid applying sigmoid to already-decoded MASK outputs.
  """
  try:
    if x.size == 0:
      return False
    a = np.asarray(x)
    if not np.issubdtype(a.dtype, np.number):
      return False
    amin = float(np.nanmin(a))
    amax = float(np.nanmax(a))
    if not np.isfinite(amin) or not np.isfinite(amax):
      return False
    return (amin >= 0.0) and (amax <= 1.0)
  except Exception:
    return False


def _looks_like_indices(x: np.ndarray) -> bool:
  """Heuristic: does x look like integer indices stored as float?

  We use this to avoid applying softmax on already-decoded outputs (e.g., argmax
  indices for POINTER/CATEGORICAL), which would incorrectly squash values.
  """
  try:
    if x.size == 0:
      return False
    if np.issubdtype(x.dtype, np.integer):
      return True
    if np.issubdtype(x.dtype, np.floating):
      r = np.rint(x)
      return bool(np.allclose(x, r, atol=0, rtol=0))
  except Exception:
    return False
  return False


def _dbg_stats(name: str, arr: Any) -> None:
  """Small numeric summary to catch 'all zeros' / constant tensors."""
  try:
    a = _to_numpy(arr)
    if a.size == 0:
      logging.info("[viz] %s stats: <empty>", name)
      return
    a = a.astype(np.float64, copy=False)
    logging.info(
        "[viz] %s stats: min=%g max=%g mean=%g std=%g",
        name,
        float(np.min(a)),
        float(np.max(a)),
        float(np.mean(a)),
        float(np.std(a)),
    )
  except Exception as e:
    logging.info("[viz] %s stats: <unavailable> (%s)", name, e)


def _maybe_one_hot_truth_to_match_pred(
    truth_bt: np.ndarray,
    pred_bt: np.ndarray,
    ttype: Any,
) -> np.ndarray:
  """If predictions are distributions but truth is stored as indices, one-hot truth."""
  if ttype not in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL, specs.Type.POINTER):
    return truth_bt

  # If truth is indices and pred has one extra class dimension, expand truth.
  if truth_bt.ndim == pred_bt.ndim - 1:
    K = pred_bt.shape[-1]
    idx = truth_bt

    if np.issubdtype(idx.dtype, np.floating):
      idx_round = np.rint(idx)
      if not np.allclose(idx, idx_round, atol=0, rtol=0):
        return truth_bt
      idx = idx_round.astype(np.int64)
    else:
      idx = idx.astype(np.int64)

    invalid = (idx < 0) | (idx >= K)
    idx_safe = np.clip(idx, 0, K - 1)

    oh = np.eye(K, dtype=np.float32)[idx_safe]  # [..., K]
    if np.any(invalid):
      oh[invalid] = 0.0
    return oh

  return truth_bt


def extract_probe(
    preds: Sequence[Mapping[str, probing.DataPoint]],
    feedback: samplers.Feedback,
    probe_name: str,
    *,
    align_mode: str = "pad_from_truth_first",
    postprocess: str = "auto",
    batch_index: int = 0  # kept for backward-compat; not used
) -> Dict[str, List[Any]]:
  """General-purpose probe extractor for ANY CLRS probe shape."""
  truth_dp = None
  for dp in feedback.features.hints:
    if dp.name == probe_name:
      truth_dp = dp
      break
  if truth_dp is None:
    raise KeyError(f"Probe '{probe_name}' not found in ground truth.")

  lengths_list: List[int] = []
  try:
    lengths_arr = _to_numpy(feedback.features.lengths)
    if lengths_arr.ndim == 0:
      lengths_list = [int(lengths_arr.item())]
    else:
      lengths_list = [int(x) for x in list(lengths_arr.flatten())]
  except Exception:
    lengths_list = []
  B_from_lengths = len(lengths_list) if lengths_list else None

  truth_bt = _truth_to_batch_first(truth_dp, lengths_list)
  _dbg_shape(f"truth.{probe_name} (bt)", truth_bt)

  pred_frames: List[np.ndarray] = []
  for t_idx, pred_map in enumerate(preds):
    if probe_name not in pred_map:
      raise KeyError(f"Probe '{probe_name}' missing at timestep {t_idx}")
    entry = pred_map[probe_name]
    x = entry.data if isinstance(entry, probing.DataPoint) else entry
    x = _to_numpy(x)

    if t_idx == 0:
      _dbg_shape(f"pred0.{probe_name} (raw)", x)

    if B_from_lengths is not None and x.ndim >= 1:
      b_axis = None
      try:
        b_axis = next((ax for ax in range(x.ndim) if x.shape[ax] == B_from_lengths), None)
      except Exception:
        b_axis = None

      if b_axis is not None and b_axis != 0:
        x = np.moveaxis(x, b_axis, 0)
      elif b_axis is None:
        x = np.expand_dims(x, axis=0)
    else:
      x = np.expand_dims(x, axis=0)

    if x.ndim > 1:
      axes = tuple(i for i in range(1, x.ndim) if x.shape[i] == 1)
      if axes:
        x = np.squeeze(x, axis=axes)

    pred_frames.append(x)

  if not pred_frames:
    raise ValueError(f"No prediction frames for probe '{probe_name}'")

  non_batch_ranks = [a.ndim - 1 for a in pred_frames]
  target_rank = min(non_batch_ranks)

  def _to_target_rank(a: np.ndarray) -> np.ndarray:
    while (a.ndim - 1) > target_rank:
      squeezed = False
      for ax in range(a.ndim - 1, 0, -1):
        if a.shape[ax] == 1:
          a = np.squeeze(a, axis=ax)
          squeezed = True
          break
      if not squeezed:
        break
    return a

  pred_frames = [_to_target_rank(a) for a in pred_frames]

  base_shape = pred_frames[0].shape
  for idx, a in enumerate(pred_frames):
    if a.shape != base_shape:
      raise ValueError(
          f"Inconsistent pred frame shapes over time for probe '{probe_name}': "
          f"t={idx} has shape {a.shape}, expected {base_shape}."
      )

  pred_bt = np.stack(pred_frames, axis=1)  # [B,T,...]
  _dbg_shape(f"pred.{probe_name} (bt)", pred_bt)

  ttype = truth_dp.type_

  if postprocess != "none":
    if ttype == specs.Type.MASK:
     
      # Sigmoid should only be applied to logits (unbounded real values).
      if _looks_like_probabilities(pred_bt):
        logging.info("[viz] %s: skipping sigmoid (pred looks like probabilities)", probe_name)
      else:
        pred_bt = _sigmoid(pred_bt)
    elif ttype in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL, specs.Type.POINTER):
      # Important: OUTPUT preds are often already decoded (indices) or probabilities.
      # Applying softmax blindly can collapse index-like tensors to ~0 everywhere.
      if _looks_like_indices(pred_bt):
        logging.info("[viz] %s: skipping softmax (pred looks index-like)", probe_name)
      else:
        _dbg_stats(f"pre_softmax.{probe_name}", pred_bt)
        pred_bt = _softmax(pred_bt, axis=-1)
        _dbg_stats(f"post_softmax.{probe_name}", pred_bt)

  truth_bt = _maybe_one_hot_truth_to_match_pred(truth_bt, pred_bt, ttype)
  _dbg_shape(f"truth.{probe_name} (bt,after_align)", truth_bt)

  T_true = truth_bt.shape[1]
  T_pred = pred_bt.shape[1]
  if T_true == T_pred + 1:
    if align_mode == "pad_from_truth_first":
      pad = np.take(truth_bt, indices=[0], axis=1)
      pred_bt = np.concatenate([pad, pred_bt], axis=1)
    elif align_mode == "pad_pred_first":
      pad = np.take(pred_bt, indices=[0], axis=1)
      pred_bt = np.concatenate([pad, pred_bt], axis=1)
    elif align_mode == "drop_truth_first":
      truth_bt = truth_bt[:, 1:]

  T = min(truth_bt.shape[1], pred_bt.shape[1])
  truth_bt = truth_bt[:, :T]
  pred_bt = pred_bt[:, :T]

  while pred_bt.ndim > 2 and pred_bt.shape[-1] == 1:
    pred_bt = pred_bt.squeeze(axis=-1)
  while truth_bt.ndim > 2 and truth_bt.shape[-1] == 1:
    truth_bt = truth_bt.squeeze(axis=-1)

  axes: List[str] = ["B", "T"]
  rem_rank = max(0, truth_bt.ndim - 2)

  if rem_rank == 0:
    pass
  elif ttype == specs.Type.POINTER:
    if rem_rank >= 2:
      axes += ["N", "N"]
    else:
      axes += ["N"]
  elif ttype in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL):
    if rem_rank == 1:
      axes += ["K"]
    elif rem_rank == 2:
      axes += ["N", "K"]
    elif rem_rank >= 3:
      axes += ["N", "N", "K"]
      for i in range(rem_rank - 3):
        axes.append(f"D{i}")
  else:
    if rem_rank == 1:
      axes += ["N"]
    elif rem_rank == 2:
      axes += ["H", "W"]
    else:
      axes += [f"D{i}" for i in range(rem_rank)]

  out: Dict[str, Any] = {
      "true": truth_bt.tolist(),
      "pred": pred_bt.tolist(),
      "axes": axes,
      "shape": list(truth_bt.shape),
  }
  if lengths_list:
    out["lengths"] = lengths_list
  return out


def _pred_array_to_bt(arr: np.ndarray, lengths_list: List[int], T_hint: int | None) -> np.ndarray:
  """Convert a model output prediction array to [B,T,...].

  Model output_preds for return_all_outputs=True are typically stacked as [T,B,...].
  Some edge cases can be [B,T,...] or [B,...] (snapshot). We normalize here.
  """
  B = len(lengths_list) if lengths_list else None

  # If already [T,B,...]
  if arr.ndim >= 2 and B is not None and arr.shape[1] == B:
    bt = np.moveaxis(arr, 1, 0)
  # If already [B,T,...]
  elif arr.ndim >= 2 and B is not None and arr.shape[0] == B:
    bt = arr
  else:
    # Snapshot case
    if B is not None and arr.ndim >= 1 and arr.shape[0] == B:
      bt0 = np.expand_dims(arr, 1)  # [B,1,...]
    else:
      bt0 = np.expand_dims(np.expand_dims(arr, 0), 0)  # [1,1,...]
    T = int(T_hint) if T_hint is not None else int(bt0.shape[1])
    bt = np.repeat(bt0, T, axis=1)

  while bt.ndim > 2 and bt.shape[-1] == 1:
    bt = bt.squeeze(axis=-1)
  return bt


def _infer_B_T(feedback: samplers.Feedback, hint_preds: Sequence[Mapping[str, Any]] | None = None) -> tuple[int | None, int | None]:
  """Infer batch size B and time steps T for a sample."""
  B = None
  T = None

  try:
    lengths_arr = _to_numpy(feedback.features.lengths)
    if lengths_arr.ndim == 0:
      B = 1
    else:
      B = int(lengths_arr.size)
  except Exception:
    B = None

  # Prefer hints truth for T.
  try:
    if getattr(feedback.features, "hints", None):
      h0 = feedback.features.hints[0]
      h0_arr = _to_numpy(h0.data)
      if h0_arr.ndim >= 2:
        T = int(h0_arr.shape[0])
  except Exception:
    pass

  # Fallback to number of pred frames.
  if T is None and hint_preds is not None:
    try:
      T = int(len(hint_preds))
    except Exception:
      T = None

  return B, T


def _ensure_bt_from_snapshot(arr: np.ndarray, *, B: int | None, T: int | None) -> np.ndarray:
  """Force arr into [B,T,...] by treating it as a snapshot and broadcasting."""
  a = _to_numpy(arr)

  # Ensure batch-first snapshot [B,...]
  if B is None:
    # unknown B => assume batch=1
    a_b = np.expand_dims(a, 0) if (a.ndim == 0 or a.shape[0] != 1) else a
    B_eff = 1
  else:
    B_eff = int(B)
    if a.ndim >= 1 and a.shape[0] == B_eff:
      a_b = a
    else:
      # If missing batch axis, broadcast to B.
      a_b = np.broadcast_to(a, (B_eff,) + a.shape)

  # Add time axis and broadcast.
  a_bt = np.expand_dims(a_b, 1)  # [B,1,...]
  T_eff = int(T) if T is not None else 1
  a_bt = np.repeat(a_bt, T_eff, axis=1)

  while a_bt.ndim > 2 and a_bt.shape[-1] == 1:
    a_bt = a_bt.squeeze(axis=-1)
  return a_bt


def _output_truth_to_bt(output_dp: probing.DataPoint, *, B: int | None, T: int | None) -> np.ndarray:
  """OUTPUT truth normalization.

  We intentionally *do not* try to interpret OUTPUT tensors as time-series.
  In CLRS, outputs are evaluation targets and are usually snapshots [B,...].
  To make visualizer consistent, we broadcast them to [B,T,...].
  """
  return _ensure_bt_from_snapshot(_to_numpy(output_dp.data), B=B, T=T)


def _truth_output_sequence(dp: probing.DataPoint, *, T: int) -> List[probing.DataPoint]:
  """Repeat an OUTPUT truth DataPoint across time as [T,B,...] for alignment."""
  arr = _to_numpy(dp.data)
  # Ensure [B,...]
  if arr.ndim == 0:
    arr_b = np.expand_dims(arr, 0)
  else:
    arr_b = arr
  # Repeat to [T,B,...]
  tiled = np.repeat(np.expand_dims(arr_b, 0), int(T), axis=0)
  return [probing.DataPoint(name=dp.name, location=dp.location, type_=dp.type_, data=tiled[t]) for t in range(int(T))]


def _unwrap_pred_value(v: Any) -> Any:
  """Unwrap a prediction value to a numeric array.

  `extract_probe` expects each pred_map[name] entry to be either:
  - a probing.DataPoint (then we use .data), or
  - a numeric ndarray-like.

  When some upstream code provides object arrays of DataPoints, numpy ops
  (softmax/max) explode. This function normalizes those cases.
  """
  if isinstance(v, probing.DataPoint):
    return v.data

  a = v
  try:
    a_np = _to_numpy(a)
    # Object arrays sometimes pack DataPoint entries.
    if isinstance(a_np, np.ndarray) and a_np.dtype == object:
      # If it's a scalar object: unwrap directly.
      if a_np.ndim == 0:
        obj = a_np.item()
        return obj.data if isinstance(obj, probing.DataPoint) else obj

      # If it's an array of DataPoints, unwrap elementwise.
      try:
        flat = a_np.ravel().tolist()
        if flat and all(isinstance(x, probing.DataPoint) for x in flat):
          flat_data = [_to_numpy(x.data) for x in flat]
          out = np.stack(flat_data, axis=0)
          return out.reshape(a_np.shape + flat_data[0].shape)
      except Exception:
        pass
  except Exception:
    pass

  return v


def _pad_pred_seq_to_T(pred_seq: List[Any], *, T: int) -> List[Any]:
  """Pad/crop a per-timestep prediction sequence to length T."""
  if not pred_seq:
    return pred_seq
  if len(pred_seq) == T:
    return pred_seq
  if len(pred_seq) == T - 1:
    # Typical CLRS case: predictions for steps 1..T-1; pad step 0.
    return [pred_seq[0]] + pred_seq
  if len(pred_seq) > T:
    return pred_seq[:T]
  # len < T-1: extend by repeating last
  last = pred_seq[-1]
  return pred_seq + [last] * (T - len(pred_seq))


def _output_pred_sequence(output_preds: Any, name: str, *, T: int) -> List[Any]:
  """Get per-timestep predicted datapoints/arrays for an output probe."""
  if not isinstance(output_preds, dict) or name not in output_preds:
    return []

  raw = output_preds[name]
  raw = _unwrap_pred_value(raw)
  arr = _to_numpy(raw)

  # Common: [T_pred,B,...] from return_all_outputs=True
  if arr.ndim >= 2 and arr.shape[1] > 0:
    # If axis1 looks like batch (matches typical B=32 etc.) assume axis0 is time.
    # We don't know B here, but we DO know T (truth time). If arr.shape[0] is close
    # to T (T or T-1), treat it as time.
    if arr.shape[0] in (int(T), int(T) - 1):
      seq = [_unwrap_pred_value(arr[t]) for t in range(int(arr.shape[0]))]
      return _pad_pred_seq_to_T(seq, T=int(T))

  # Snapshot: [B,...] or anything else.
  snap = _unwrap_pred_value(arr)
  return [snap for _ in range(int(T))]


def _maybe_cast_float_indices_to_int(bt: np.ndarray) -> np.ndarray:
  """If bt is float but represents integer indices, cast to int64."""
  if not np.issubdtype(bt.dtype, np.floating):
    return bt
  r = np.rint(bt)
  if np.allclose(bt, r, atol=0, rtol=0):
    return r.astype(np.int64)
  return bt


def extract_all_probes(
    hint_preds: Sequence[Mapping[str, probing.DataPoint]],
    feedback: samplers.Feedback,
    *,
    include_gnn_hidden_states: bool = False,
    output_preds: Any | None = None,
    gnn_hidden_states: Any | None = None,
) -> Dict[str, Any]:
  """Extracts ALL probes present in `feedback` into a single structure."""
  lengths_list: List[int] = []
  try:
    lengths_arr = _to_numpy(feedback.features.lengths)
    if lengths_arr.ndim == 0:
      lengths_list = [int(lengths_arr.item())]
    else:
      lengths_list = [int(x) for x in list(lengths_arr.flatten())]
  except Exception:
    lengths_list = []

  B, T = _infer_B_T(feedback, hint_preds)
  if T is None:
    T = 1

  algo_name = getattr(getattr(feedback, 'features', None), 'algo', None)
  if algo_name is None:
    algo_name = getattr(getattr(feedback, 'features', None), 'name', None)
  if algo_name is None:
    algo_name = "unknown"

  probes: Dict[str, Any] = {}

  def _attach_meta(entry: Dict[str, Any], dp: probing.DataPoint, stage_name: str) -> None:
    entry["type"] = str(getattr(dp.type_, "name", dp.type_)).lower()
    entry["location"] = str(getattr(dp.location, "name", dp.location)).lower()
    entry["stage"] = stage_name

  # HINT probes (time series)
  for dp in feedback.features.hints:
    try:
      entry = extract_probe(
          preds=hint_preds,
          feedback=feedback,
          probe_name=dp.name,
          align_mode="pad_from_truth_first",
          postprocess="auto",
      )
      probes[dp.name] = entry
      _attach_meta(entry, dp, "HINT")
    except Exception as e:
      probes[dp.name] = {"error": str(e), "stage": "HINT"}

  # Attach official CLRS hint correctness curves (per timestep), if possible.
  # This uses evaluation.evaluate_hints() (same semantics as training/eval).
  try:
    hints_tuple = tuple(getattr(feedback.features, 'hints', ()) or ())
    lengths_arr = _to_numpy(getattr(feedback.features, 'lengths', np.array([])))
    if hints_tuple and hint_preds:
      by_name_truth = {h.name: h for h in hints_tuple}

      # Infer target T from truth (preferred) or prediction frames.
      T_eval: int | None = None
      try:
        if hints_tuple:
          h0_arr = _to_numpy(hints_tuple[0].data)
          if h0_arr.ndim >= 1:
            T_eval = int(h0_arr.shape[0])
      except Exception:
        T_eval = None
      if T_eval is None:
        try:
          T_eval = int(len(hint_preds))
        except Exception:
          T_eval = None
      if T_eval is None:
        T_eval = 1

      def _scalar_mse_to_correctness(mse: float, truth_step: np.ndarray) -> float:
        """Map scalar MSE (0=perfect) to a [0,1] correctness score (1=perfect)."""
        try:
          t = np.asarray(truth_step, dtype=np.float32)
          if t.size == 0:
            return 0.0
          # Robust scale proxy: RMS of truth (avoid division by near-zero).
          rms = float(np.sqrt(np.mean(t * t)))
          scale = max(rms, 1e-6)
          return float(1.0 / (1.0 + (float(mse) / (scale * scale))))
        except Exception:
          # Fallback: still monotonic but scale-free.
          return float(1.0 / (1.0 + float(mse)))

      def _scalar_mse_to_correctness_global(mse: float, *, scale_sq: float) -> float:
        """Map scalar MSE to [0,1] using a fixed global scale (1=perfect)."""
        try:
          denom = float(max(scale_sq, 1e-12))
          return float(1.0 / (1.0 + (float(mse) / denom)))
        except Exception:
          return float(1.0 / (1.0 + float(mse)))

      def _scalar_global_scale_sq(truth_full: np.ndarray) -> float:
        """Compute a per-hint global scale^2 from truth over time.

        Uses RMS over all non-time dims and across timesteps 1..T-1 (since idx=0
        isn't evaluated by CLRS for hints).
        """
        try:
          tf = np.asarray(truth_full, dtype=np.float32)
          if tf.size == 0:
            return 1.0
          if tf.ndim >= 1 and tf.shape[0] > 1:
            tf = tf[1:]
          # RMS^2 == mean(x^2)
          scale_sq = float(np.mean(tf * tf))
          return max(scale_sq, 1e-12)
        except Exception:
          return 1.0

      # Align prediction representation to match truth representation for evaluation.
      def _align_pred_to_truth(truth: probing.DataPoint, pred: probing.DataPoint) -> probing.DataPoint:
        t = truth.type_

        # IMPORTANT: hint truth usually includes a time axis [T,B,...]. For deciding
        # whether a prediction is a distribution that needs argmax-decoding, compare
        # to the per-timestep truth shape (time axis removed).
        t_arr_full = _to_numpy(truth.data)
        t_arr = t_arr_full
        try:
          if t_arr_full.ndim >= 1 and int(t_arr_full.shape[0]) == int(T_eval):
            # Use step-0 as shape reference.
            t_arr = _to_numpy(t_arr_full[0])
        except Exception:
          t_arr = t_arr_full

        p_arr = _to_numpy(pred.data)

        # POINTER: if pred is distribution [...,N] but truth is indices [...], decode.
        if t == specs.Type.POINTER and p_arr.ndim == t_arr.ndim + 1:
          p_arr = np.argmax(p_arr, axis=-1)

        # CATEGORICAL/MASK_ONE: allow distribution -> decode to label indices.
        if t in (specs.Type.CATEGORICAL, specs.Type.MASK_ONE) and p_arr.ndim == t_arr.ndim + 1:
          p_arr = np.argmax(p_arr, axis=-1)

        # Ensure dtypes are reasonable for discrete types.
        if t in (specs.Type.POINTER, specs.Type.CATEGORICAL, specs.Type.MASK_ONE):
          if np.issubdtype(p_arr.dtype, np.floating):
            p_arr = np.rint(p_arr).astype(np.int64)

        return probing.DataPoint(name=truth.name, location=truth.location, type_=truth.type_, data=p_arr)

      # Collect per-hint prediction sequences and evaluate each hint independently so
      # one bad probe can't disable all corr toggles.
      for name, truth in by_name_truth.items():
        try:
          # Build a raw sequence (may have missing steps) from hint_preds.
          raw_seq: List[probing.DataPoint] = []
          for pm in hint_preds:
            if not isinstance(pm, Mapping):
              continue
            # Keys may not always be strings; compare via str(k).
            found = None
            for k, v in pm.items():
              if str(k) == name:
                found = v
                break
            if found is None:
              continue
            if isinstance(found, probing.DataPoint):
              raw_seq.append(found)
            else:
              raw_seq.append(
                  probing.DataPoint(
                      name=name,
                      location=truth.location,
                      type_=truth.type_,
                      data=_to_numpy(_unwrap_pred_value(found)),
                  )
              )

          if not raw_seq:
            continue

          # Align seq length to truth timeline (typical: preds are for steps 1..T-1).
          raw_seq_any: List[Any] = list(raw_seq)
          raw_seq_any = _pad_pred_seq_to_T(raw_seq_any, T=int(T_eval))

          # IMPORTANT: evaluation.evaluate_hints compares pred at list index i
          # against truth[idx=i+1]. That means hint_preds must correspond to
          # algorithm steps 1..T-1 (no explicit timestep-0 prediction).
          #
          # Our visualizer alignment pads a timestep-0 prediction to make the
          # exported arrays [B,T,...] line up for rendering. For correctness,
          # we must drop that padded step to avoid an off-by-one shift.
          if len(raw_seq_any) == int(T_eval):
            eval_raw_seq_any = raw_seq_any[1:]
          else:
            eval_raw_seq_any = raw_seq_any

          one_hint_preds: List[Dict[str, probing.DataPoint]] = []
          for t in range(len(eval_raw_seq_any)):
            dp_pred = eval_raw_seq_any[t]
            if not isinstance(dp_pred, probing.DataPoint):
              dp_pred = probing.DataPoint(
                  name=name,
                  location=truth.location,
                  type_=truth.type_,
                  data=_to_numpy(_unwrap_pred_value(dp_pred)),
              )
            one_hint_preds.append({name: _align_pred_to_truth(truth, dp_pred)})

          evals = evaluation.evaluate_hints((truth,), lengths_arr, one_hint_preds)
          key = name + '_along_time'
          if key in evals and name in probes and isinstance(probes[name], dict) and 'error' not in probes[name]:
            series = _to_numpy(evals[key]).astype(np.float32)

            if truth.type_ == specs.Type.SCALAR:
              # Convert per-step MSE to a [0,1] correctness curve.
              truth_full = _to_numpy(truth.data)
              global_scale_sq = _scalar_global_scale_sq(truth_full)
              corr: List[float] = []
              for i in range(int(series.shape[0])):
                corr.append(_scalar_mse_to_correctness_global(float(series[i]), scale_sq=global_scale_sq))
              # Pad to length T_eval for the viewer (t=0 has no eval by design).
              if int(T_eval) == int(series.shape[0]) + 1:
                corr = [corr[0] if corr else 0.0] + corr
              probes[name]['correctness_along_time'] = corr
            else:
              # Pad to length T_eval for the viewer (t=0 has no eval by design).
              out_series = series.tolist()
              if int(T_eval) == int(series.shape[0]) + 1:
                out_series = [out_series[0] if out_series else 0.0] + out_series
              probes[name]['correctness_along_time'] = out_series
        except Exception as e_one:
          logging.info('[viz] Hint correctness skipped for %s due to evaluation error: %s', name, e_one)
  except Exception as e:
    logging.info('[viz] Failed to compute hint correctness via evaluation.evaluate_hints: %s', e)

  # --- OUTPUT probes
  outputs_container = getattr(getattr(feedback, "features", None), "outputs", None)
  if outputs_container is None:
    outputs_container = getattr(feedback, "outputs", None)
  outputs_dps = list(outputs_container) if outputs_container is not None else []

  for dp in outputs_dps:
    _dbg_shape(f"output_truth_raw.{dp.name}", dp.data)

    truth_seq = _truth_output_sequence(dp, T=int(T))
    truth_tb = np.stack([_to_numpy(_unwrap_pred_value(x.data)) for x in truth_seq], axis=0)  # [T,B,...]
    _dbg_shape(f"output_truth_tb.{dp.name}", truth_tb)

    # Raw model output prediction tensor for this output (for debugging)
    if isinstance(output_preds, dict) and dp.name in output_preds:
      _dbg_shape(f"output_pred_raw.{dp.name}", _unwrap_pred_value(output_preds[dp.name]))

    pred_seq = _output_pred_sequence(output_preds, dp.name, T=int(T))
    logging.info("[viz] output.%s T_truth=%d T_pred=%s", dp.name, int(T), (len(pred_seq) if pred_seq else None))

    if pred_seq:
      _dbg_shape(f"output_pred0.{dp.name}", pred_seq[0])
    else:
      logging.info("[viz] output_pred.%s missing -> using truth as pred", dp.name)

    # Build pred maps for extract_probe.
    pred_maps: List[Dict[str, Any]] = []
    for t in range(int(T)):
      pm: Dict[str, Any] = {}
      pm[dp.name] = _unwrap_pred_value(pred_seq[t]) if pred_seq else truth_tb[t]
      pred_maps.append(pm)

    fake_truth_dp = probing.DataPoint(name=dp.name, location=dp.location, type_=dp.type_, data=truth_tb)
    fake_features = feedback.features._replace(hints=tuple([fake_truth_dp]))
    fake_feedback = feedback._replace(features=fake_features)

    try:
      entry = extract_probe(
          preds=pred_maps,
          feedback=fake_feedback,
          probe_name=dp.name,
          align_mode="pad_from_truth_first",
          postprocess="auto",
      )
      _attach_meta(entry, dp, "OUTPUT")

      # Enforce invariant: OUTPUT pred shape MUST match OUTPUT true shape.
      # Representation is driven solely by what the model predicts.
      pred_bt = _to_numpy(entry.get("pred"))
      true_bt = _to_numpy(entry.get("true"))

      # If model predicts distributions/logits with an extra class dimension, one-hot truth.
      true_bt = _maybe_cast_float_indices_to_int(true_bt)
      if true_bt.ndim == pred_bt.ndim - 1 and dp.type_ in (specs.Type.POINTER, specs.Type.CATEGORICAL, specs.Type.MASK_ONE):
        true_bt = _maybe_one_hot_truth_to_match_pred(true_bt, pred_bt, dp.type_)

      # After potential truth expansion, pred and true must match. If not, do NOT
      # invent dimensions; instead, fall back to keeping both in their native forms.
      if true_bt.shape != pred_bt.shape:
        logging.warning(
            "[viz] OUTPUT shape mismatch for %s: true=%s pred=%s (keeping native)",
            dp.name, list(true_bt.shape), list(pred_bt.shape)
        )
        # Keep truth as-is (already constant over time); keep pred as-is.
      else:
        entry["true"] = true_bt.tolist()
        entry["shape"] = list(true_bt.shape)

      _dbg_shape(f"output_final_true.{dp.name}", _to_numpy(entry.get("true")))
      _dbg_shape(f"output_final_pred.{dp.name}", _to_numpy(entry.get("pred")))

      probes[dp.name] = entry
    except Exception as e:
      logging.exception("[viz] Failed extracting OUTPUT %s", dp.name)
      probes[dp.name] = {"error": str(e), "stage": "OUTPUT"}

  out: Dict[str, Any] = {"algorithm": str(algo_name), "probes": probes}

  if include_gnn_hidden_states:
    try:
      # Prefer explicit hidden states passed from the model (debug=True path).
      hs_src = gnn_hidden_states
      # Back-compat fallback: some pipelines may attach hidden_states onto features.
      if hs_src is None:
        hs_src = getattr(feedback.features, "hidden_states", None)

      if hs_src is None:
        raise AttributeError("No hidden states available (gnn_hidden_states is None and feedback.features has no 'hidden_states')")

      hs = _to_numpy(hs_src)
      if hs.ndim >= 2 and hs.shape[0] != 0:
        inferred_B = len(lengths_list) if lengths_list else None
        if inferred_B is not None:
          # Expected viewer format is [B,T,N,D].
          # Common model format is [T,B,N,D].
          if hs.ndim >= 2 and hs.shape[1] == inferred_B and hs.shape[0] != inferred_B:
            hs_btnd = np.moveaxis(hs, 0, 1)
          elif hs.shape[0] == inferred_B:
            hs_btnd = hs
          else:
            hs_btnd = np.moveaxis(hs, 0, 1)
        else:
          hs_btnd = np.moveaxis(hs, 0, 1)
      else:
        hs_btnd = hs

      while hs_btnd.ndim < 4:
        hs_btnd = np.expand_dims(hs_btnd, axis=-1)

      out["gnn_hidden_states"] = {
          "data": hs_btnd.tolist(),
          "axes": ["B", "T", "N", "D"],
          "shape": list(hs_btnd.shape),
      }
    except Exception as e:
      out["gnn_hidden_states"] = {"error": f"Failed to serialize hidden states: {e}"}

  if lengths_list:
    out["lengths"] = lengths_list
    is_last: List[List[bool]] = []
    for L in lengths_list:
      try:
        L_int = int(L)
      except Exception:
        L_int = None
      if (L_int is None) or (L_int <= 0):
        is_last.append([])
      else:
        row = [False] * L_int
        row[-1] = True
        is_last.append(row)
    out["is_last"] = is_last

  return out


def _json_sanitize(obj):
  """Convert objects to JSON-serializable equivalents."""
  if obj is None or isinstance(obj, (bool, int, float, str)):
    return obj
  if isinstance(obj, (list, tuple)):
    return [_json_sanitize(x) for x in obj]
  if isinstance(obj, dict):
    return {str(k): _json_sanitize(v) for k, v in obj.items()}

  # numpy scalars/arrays
  try:
    import numpy as _np

    if isinstance(obj, _np.ndarray):
      return obj.tolist()
    if isinstance(obj, _np.generic):
      return obj.item()
  except Exception:
    pass

  # CLRS DataPoint (or similar objects)
  if hasattr(obj, "data") and hasattr(obj, "name"):
    try:
      return {
          "name": getattr(obj, "name", None),
          "type": str(getattr(obj, "type_", None)),
          "location": str(getattr(obj, "location", None)),
          "data": _json_sanitize(getattr(obj, "data", None)),
      }
    except Exception:
      return str(obj)

  return str(obj)


def save_algo_to_json(path: str, algo_data: Dict[str, Any]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump(_json_sanitize(algo_data), f, ensure_ascii=False, separators=(",", ":"))


def quantize_floats_in_obj(obj, decimals: int = 9):
  """Recursively rounds floats to a fixed number of decimals in a JSON-ready tree."""
  try:
    import numpy as _np  # local import to avoid hard dep
    np_floating = _np.floating
  except Exception:
    _np = None

    class _Dummy:
      pass

    np_floating = _Dummy

  def _q(x):
    try:
      y = round(float(x), decimals)
      if y == 0.0:
        return 0.0
      return y
    except Exception:
      return x

  if isinstance(obj, float) or (_np is not None and isinstance(obj, np_floating)):
    return _q(obj)
  if isinstance(obj, list):
    return [quantize_floats_in_obj(v, decimals) for v in obj]
  if isinstance(obj, tuple):
    return tuple(quantize_floats_in_obj(v, decimals) for v in obj)
  if isinstance(obj, dict):
    return {k: quantize_floats_in_obj(v, decimals) for k, v in obj.items()}
  return obj


def extract_output_probe(
    output_dp: probing.DataPoint,
    feedback: samplers.Feedback,
) -> Dict[str, Any]:
  """Extract a CLRS OUTPUT DataPoint as a time series [B,T,...] suitable for viewer.

  Strategy:
  - If output already contains a time axis (e.g., [T,B,...] or [B,T,...]), preserve it.
  - Otherwise, treat it as a final snapshot and broadcast to match the hint T.
  """
  # Determine B and T from lengths and hints.
  lengths_list: List[int] = []
  try:
    lengths_arr = _to_numpy(feedback.features.lengths)
    if lengths_arr.ndim == 0:
      lengths_list = [int(lengths_arr.item())]
    else:
      lengths_list = [int(x) for x in list(lengths_arr.flatten())]
  except Exception:
    lengths_list = []

  # Infer T from a hint if possible
  T_hint = None
  try:
    if getattr(feedback.features, "hints", None):
      h0 = feedback.features.hints[0]
      h0_arr = _to_numpy(h0.data)
      # Most hints are [T,B] (or scalar edge cases)
      if h0_arr.ndim >= 2:
        T_hint = int(h0_arr.shape[0])
  except Exception:
    T_hint = None

  arr = _to_numpy(output_dp.data)
  B = len(lengths_list) if lengths_list else None

  # Case 1: already has a time axis.
  # Prefer [T,B,...] -> convert to [B,T,...]
  if arr.ndim >= 2 and B is not None:
    if arr.shape[1] == B:
      bt = np.moveaxis(arr, 1, 0)  # [B,T,...]
    elif arr.shape[0] == B:
      bt = arr  # likely already [B,T,...]
    else:
      bt = None
  else:
    bt = None

  if bt is None:
    # Case 2: snapshot. Make it [B,1,...] then tile to T.
    if B is None:
      bt0 = np.expand_dims(np.expand_dims(arr, 0), 0)  # [1,1,...]
    else:
      # Ensure batch is first.
      if arr.ndim >= 1 and arr.shape[0] == B:
        bt0 = np.expand_dims(arr, 1)  # [B,1,...]
      else:
        bt0 = np.expand_dims(np.expand_dims(arr, 0), 0)  # [1,1,...]

    T = int(T_hint) if T_hint is not None else int(bt0.shape[1])
    bt = np.repeat(bt0, T, axis=1)

  # Squeeze trailing singleton dims
  while bt.ndim > 2 and bt.shape[-1] == 1:
    bt = bt.squeeze(axis=-1)

  axes = ["B", "T"]
  rem_rank = max(0, bt.ndim - 2)
  ttype = output_dp.type_
  if rem_rank == 0:
    pass
  elif ttype == specs.Type.POINTER:
    # If has [B,T,N,N] -> show matrix; if [B,T,N] -> show vector
    if rem_rank >= 2:
      axes += ["N", "N"]
    else:
      axes += ["N"]
  elif ttype in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL):
    if rem_rank == 1:
      axes += ["K"]
    elif rem_rank == 2:
      axes += ["N", "K"]
    elif rem_rank >= 3:
      axes += ["N", "N", "K"]
      for i in range(rem_rank - 3):
        axes.append(f"D{i}")
  else:
    if rem_rank == 1:
      axes += ["N"]
    elif rem_rank == 2:
      axes += ["H", "W"]
    else:
      axes += [f"D{i}" for i in range(rem_rank)]

  out: Dict[str, Any] = {
      "true": bt.tolist(),
      # NOTE: pred is filled in extract_all_probes when model output preds are available.
      "pred": bt.tolist(),
      "axes": axes,
      "shape": list(bt.shape),
  }
  if lengths_list:
    out["lengths"] = lengths_list
  return out


def _dbg_shape(name: str, arr: Any) -> None:
  try:
    a = _to_numpy(arr)
    logging.info("[viz] %s shape=%s dtype=%s", name, list(a.shape), getattr(a, "dtype", None))
  except Exception as e:
    logging.info("[viz] %s shape=<unavailable> (%s)", name, e)


__all__ = [
    "extract_probe",
    "extract_all_probes",
    "save_algo_to_json",
    "quantize_floats_in_obj",
]
