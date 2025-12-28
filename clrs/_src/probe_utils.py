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


def _strip_batch_dim(x: np.ndarray) -> np.ndarray:
  """If batch dimension B=1 exists as leading dim, remove it."""
  if x.ndim >= 2 and x.shape[0] == 1:
    return x[0]
  return x


def _strip_trailing_singletons(x: np.ndarray) -> np.ndarray:
  """Remove trailing singleton dims (e.g., (n,1) → (n,))."""
  while x.ndim > 1 and x.shape[-1] == 1:
    x = x.squeeze(axis=-1)
  return x


def _truth_to_batch_first(dp: probing.DataPoint, lengths_list: List[int]) -> np.ndarray:
  """Converts ground-truth hint to batch-first array [B, T, ...].

  CLRS feedback hints are often time-major ([T,B,...]) for batched probes,
  or [T,...] when unbatched. This detects a batch dim via lengths_list if present.
  """
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


def _maybe_one_hot_truth_to_match_pred(
    truth_bt: np.ndarray,
    pred_bt: np.ndarray,
    ttype: Any,
) -> np.ndarray:
  """If predictions are distributions but truth is stored as indices, one-hot truth.

  This situation is common for POINTER / CATEGORICAL / MASK_ONE probes:
    - truth: [B,T,...] integer indices
    - pred : [B,T,...,K] logits/probs distribution over K classes

  We detect it generically by checking rank mismatch of exactly 1.
  """
  if ttype not in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL, specs.Type.POINTER):
    return truth_bt

  # If truth is indices and pred has one extra class dimension, expand truth.
  if truth_bt.ndim == pred_bt.ndim - 1:
    K = pred_bt.shape[-1]
    idx = truth_bt

    # Convert floats safely if they are actually integral.
    if np.issubdtype(idx.dtype, np.floating):
      idx_round = np.rint(idx)
      if not np.allclose(idx, idx_round, atol=0, rtol=0):
        # Not integral; don't one-hot.
        return truth_bt
      idx = idx_round.astype(np.int64)
    else:
      idx = idx.astype(np.int64)

    # Guard against invalid indices (e.g., -1 padding). If present, clamp and mask.
    invalid = (idx < 0) | (idx >= K)
    idx_safe = np.clip(idx, 0, K - 1)

    oh = np.eye(K, dtype=np.float32)[idx_safe]  # [..., K]
    if np.any(invalid):
      # Set invalid positions to all-zeros distribution
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
    batch_index: int = 0,  # kept for backward-compat; not used
) -> Dict[str, List[Any]]:
  """General-purpose probe extractor for ANY CLRS probe shape.

  Returns:
    Dict with keys: true, pred, axes, shape (+ optional lengths).
    Shapes are batch-first [B,T,...].
  """
  # Find truth datapoint
  truth_dp = None
  for dp in feedback.features.hints:
    if dp.name == probe_name:
      truth_dp = dp
      break
  if truth_dp is None:
    raise KeyError(f"Probe '{probe_name}' not found in ground truth.")

  # Collect lengths for batch detection
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

  # Build truth [B,T,...]
  truth_bt = _truth_to_batch_first(truth_dp, lengths_list)

  # Build preds frames normalized to [B,...] per timestep, then stack -> [B,T,...]
  pred_frames: List[np.ndarray] = []
  for t_idx, pred_map in enumerate(preds):
    if probe_name not in pred_map:
      raise KeyError(f"Probe '{probe_name}' missing at timestep {t_idx}")
    entry = pred_map[probe_name]
    x = entry.data if isinstance(entry, probing.DataPoint) else entry
    x = _to_numpy(x)

    # Ensure batch axis is leading.
    if B_from_lengths is not None and x.ndim >= 1:
      b_axis = None
      try:
        b_axis = next((ax for ax in range(x.ndim) if x.shape[ax] == B_from_lengths), None)
      except Exception:
        b_axis = None

      if b_axis is not None and b_axis != 0:
        x = np.moveaxis(x, b_axis, 0)
      elif b_axis is None:
        x = np.expand_dims(x, axis=0)  # assume unbatched
    else:
      x = np.expand_dims(x, axis=0)  # unknown batch -> assume B=1

    # Squeeze non-batch singleton axes to canonicalize (avoid (B,N,1) vs (B,N))
    if x.ndim > 1:
      axes = tuple(i for i in range(1, x.ndim) if x.shape[i] == 1)
      if axes:
        x = np.squeeze(x, axis=axes)

    pred_frames.append(x)

  if not pred_frames:
    raise ValueError(f"No prediction frames for probe '{probe_name}'")

  # Minimize non-batch rank if frames vary (keep the minimum non-batch rank)
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

  # Ensure consistent per-frame shape
  base_shape = pred_frames[0].shape
  for idx, a in enumerate(pred_frames):
    if a.shape != base_shape:
      raise ValueError(
        f"Inconsistent pred frame shapes over time for probe '{probe_name}': "
        f"t={idx} has shape {a.shape}, expected {base_shape}."
      )

  pred_bt = np.stack(pred_frames, axis=1)  # [B,T,...]

  # Determine probe type
  ttype = truth_dp.type_

  # Type-aware postprocessing on predictions (logits -> probs), before alignment
  if postprocess != "none":
    if ttype == specs.Type.MASK:
      pred_bt = _sigmoid(pred_bt)
    elif ttype in (specs.Type.MASK_ONE, specs.Type.CATEGORICAL, specs.Type.POINTER):
      pred_bt = _softmax(pred_bt, axis=-1)
    # SCALAR -> identity

  # --- Generic FIX: if truth is indices but pred is distribution, one-hot truth ---
  truth_bt = _maybe_one_hot_truth_to_match_pred(truth_bt, pred_bt, ttype)

  # Time alignment between truth and pred
  T_true = truth_bt.shape[1]
  T_pred = pred_bt.shape[1]
  if T_true == T_pred + 1:
    if align_mode == "pad_from_truth_first":
      # Pad predictions with first truth frame (already in correct domain/shape)
      pad = np.take(truth_bt, indices=[0], axis=1)
      pred_bt = np.concatenate([pad, pred_bt], axis=1)
    elif align_mode == "pad_pred_first":
      pad = np.take(pred_bt, indices=[0], axis=1)
      pred_bt = np.concatenate([pad, pred_bt], axis=1)
    elif align_mode == "drop_truth_first":
      truth_bt = truth_bt[:, 1:]
    # else: none

  # Trim to common time length
  T = min(truth_bt.shape[1], pred_bt.shape[1])
  truth_bt = truth_bt[:, :T]
  pred_bt = pred_bt[:, :T]

  # Remove trailing singleton dims (keep at least [B,T])
  while pred_bt.ndim > 2 and pred_bt.shape[-1] == 1:
    pred_bt = pred_bt.squeeze(axis=-1)
  while truth_bt.ndim > 2 and truth_bt.shape[-1] == 1:
    truth_bt = truth_bt.squeeze(axis=-1)

  # Axes metadata
  axes: List[str] = ["B", "T"]
  if truth_bt.ndim == 3:
    axes += ["N"]
  elif truth_bt.ndim == 4:
    # If POINTER, it's commonly [B,T,N,N]; label it explicitly
    if ttype == specs.Type.POINTER:
      axes += ["N", "N"]
    else:
      axes += ["H", "W"]
  else:
    axes += [f"D{i}" for i in range(truth_bt.ndim - 2)]

  out: Dict[str, Any] = {
    "true": truth_bt.tolist(),
    "pred": pred_bt.tolist(),
    "axes": axes,
    "shape": list(truth_bt.shape),
  }
  if lengths_list:
    out["lengths"] = lengths_list
  return out


def save_probe_to_json(path: str, probe_data: Dict[str, Any]) -> None:
  """JSON-safe serialization."""
  with open(path, "w", encoding="utf-8") as f:
    json.dump(probe_data, f, ensure_ascii=False, indent=2)


def extract_all_probes(
    preds: Sequence[Mapping[str, probing.DataPoint]],
    feedback: samplers.Feedback,
    *,
    align_mode: str = "pad_from_truth_first",
    postprocess: str = "auto",
) -> Dict[str, Any]:
  """Extracts ALL probes present in `feedback` into a single structure.

  Returns:
    {
      "algorithm": ...,
      "lengths": [...],  # optional
      "probes": {
        probe_name: { true, pred, axes, shape, type } OR { error }
      }
    }
  """
  # Collect lengths first
  lengths_list: List[int] = []
  try:
    lengths_arr = _to_numpy(feedback.features.lengths)
    if lengths_arr.ndim == 0:
      lengths_list = [int(lengths_arr.item())]
    else:
      lengths_list = [int(x) for x in list(lengths_arr.flatten())]
  except Exception:
    lengths_list = []

  # Determine algorithm name if present
  algo_name = getattr(getattr(feedback, 'features', None), 'algo', None)
  if algo_name is None:
    algo_name = getattr(getattr(feedback, 'features', None), 'name', None)
  if algo_name is None:
    algo_name = "unknown"

  probes: Dict[str, Any] = {}
  for dp in feedback.features.hints:
    try:
      entry = extract_probe(
        preds=preds,
        feedback=feedback,
        probe_name=dp.name,
        align_mode=align_mode,
        postprocess=postprocess,
      )
      # Attach type info for the viewer
      t = dp.type_
      try:
        t_name = specs.Type(t).name if isinstance(t, int) else t.name
      except Exception:
        t_name = str(t)
      entry["type"] = t_name
      probes[dp.name] = entry
    except Exception as e:
      probes[dp.name] = {"error": str(e)}

  out: Dict[str, Any] = {"algorithm": str(algo_name), "probes": probes}
  if lengths_list:
    out["lengths"] = lengths_list
    # Best-effort per-batch termination flags for the viewer: is_last[b][t]
    # We construct a jagged list where each row length equals lengths[b],
    # and the last valid timestep (lengths[b]-1) is marked True.
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


def save_algo_to_json(path: str, algo_data: Dict[str, Any]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump(algo_data, f, ensure_ascii=False, indent=2)


__all__ = [
  "extract_probe",
  "extract_all_probes",
  "save_probe_to_json",
  "save_algo_to_json",
]
