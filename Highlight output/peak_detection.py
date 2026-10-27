"""
Stage 9 - Peak detection & ranking (basic signal processing)

Input : one excitement score per line (Head A, or the weak labels) + a mask of lines that may be
        highlights (line_kind == live).
Output: the indices of the highlight peaks, best first, and the build-up span around each.

Steps
  1. smooth       optional moving average (default off; the model is already smooth over a window)
  2. local maxima a line higher than both neighbours. A flat top (several equal lines) counts once
  3. threshold    keep peaks with score >= threshold
  4. de-duplicate non-maximum suppression: go through peaks from the highest down and drop any
                  peak closer than min_distance lines to one already kept. This is what merges
                  "GOAL!!!", "it's a goal" and "celebrations" lines into ONE highlight.
  5. top-k        keep at most top_k per match
Pure numpy; no scipy needed.

Choosing the knobs (they are the first things you will tune):
    threshold     0.5   raise it for fewer, bigger moments; lower it to include cards and chances
    min_distance  5     lines. About the length of one passage of play. Too small -> duplicates
    top_k         12    per match
"""
from __future__ import annotations

import numpy as np


def smooth(scores: np.ndarray, k: int = 1) -> np.ndarray:
    """Centered moving average with window k (odd). k <= 1 returns the scores unchanged."""
    scores = np.asarray(scores, dtype=float)
    if k <= 1:
        return scores.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    padded = np.pad(scores, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def local_maxima(scores: np.ndarray, valid: np.ndarray, plateau: str = "center") -> list[int]:
    """Indices of local maxima among valid lines. Invalid lines act as -inf, so a valid line next
    to a prematch/recap line can still be a peak. A run of equal scores is reported once
    (plateau = 'center' or 'first')."""
    s = np.where(valid, np.asarray(scores, dtype=float), -np.inf)
    T = len(s)
    peaks, i = [], 0
    while i < T:
        j = i
        while j + 1 < T and s[j + 1] == s[i]:
            j += 1
        left = s[i - 1] if i > 0 else -np.inf
        right = s[j + 1] if j + 1 < T else -np.inf
        whole = i == 0 and j == T - 1          # a completely flat curve has no peak
        if np.isfinite(s[i]) and s[i] > left and s[i] > right and not whole:
            peaks.append(i if plateau == "first" else (i + j) // 2)
        i = j + 1
    return peaks


def find_peaks(scores: np.ndarray, valid: np.ndarray, threshold: float = 0.5,
               min_distance: int = 5, top_k: int | None = 12, smooth_k: int = 1,
               plateau: str = "center") -> list[int]:
    """Highlight peaks, ordered best first (ties: earlier line first)."""
    s = smooth(scores, smooth_k)
    cand = [i for i in local_maxima(s, valid, plateau) if s[i] >= threshold]
    cand.sort(key=lambda i: (-s[i], i))
    kept: list[int] = []
    for i in cand:
        if all(abs(i - k) >= min_distance for k in kept):
            kept.append(i)
        if top_k is not None and len(kept) >= top_k:
            break
    return kept


def build_span(scores: np.ndarray, valid: np.ndarray, peak: int, other_peaks: set[int],
               floor: float = 0.25, max_before: int = 10, max_after: int = 3) -> tuple[int, int]:
    """Lines around a peak that belong to the same moment: contiguous valid lines whose score
    stays >= floor, at most max_before lines before and max_after after, never crossing another
    highlight's peak. Returns (start, end), both inclusive; start..peak-1 is the build-up."""
    T = len(scores)
    start = peak
    while (start - 1 >= 0 and valid[start - 1] and scores[start - 1] >= floor
           and peak - (start - 1) <= max_before and (start - 1) not in other_peaks):
        start -= 1
    end = peak
    while (end + 1 < T and valid[end + 1] and scores[end + 1] >= floor
           and (end + 1) - peak <= max_after and (end + 1) not in other_peaks):
        end += 1
    return start, end
