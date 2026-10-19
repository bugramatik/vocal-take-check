import os

import librosa
import numpy as np

from rvc import Config
from vc_infer_pipeline import VC

SAMPLE_RATE = 16000
HOP_LENGTH = 160
F0_MIN = 50
F0_MAX = 1100
DRIFT_FLAG_CENTS = 60.0
MIN_FLAGGED_SECONDS = 0.25


def _pitch_track(audio_path, f0_method='rmvpe', device='cpu', is_half=False):
    """Return the f0 track of an audio file in Hz, 0 where the frame is unvoiced."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        raise ValueError(f'{audio_path} contains no audio samples.')

    config = Config(device, is_half)
    vc = VC(SAMPLE_RATE, config)
    p_len = audio.shape[0] // HOP_LENGTH
    _, f0 = vc.get_f0(audio_path, audio, p_len, 0, f0_method, 3, 128)
    return np.asarray(f0, dtype=np.float64)


def _cents(reference, generated):
    """Signed distance in cents between two f0 values, 0 where either is unvoiced."""
    voiced = (reference > 0) & (generated > 0)
    cents = np.zeros_like(reference)
    cents[voiced] = 1200.0 * np.log2(generated[voiced] / reference[voiced])
    return cents, voiced


def _ranges(mask, frame_seconds, min_seconds=MIN_FLAGGED_SECONDS):
    """Turn a boolean per-frame mask into a list of (start, end) second ranges."""
    ranges = []
    start = None
    for index, flagged in enumerate(mask):
        if flagged and start is None:
            start = index
        elif not flagged and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))

    merged = []
    for begin, end in ranges:
        if (end - begin) * frame_seconds < min_seconds:
            continue
        if merged and begin - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((begin, end))

    return [(round(begin * frame_seconds, 2), round(end * frame_seconds, 2)) for begin, end in merged]


def score_take(source_path, generated_path, f0_method='rmvpe', device='cpu', is_half=False):
    """Compare a generated vocal against its source and score how well it holds pitch.

    Returns a dict with the score (0 to 1), the median absolute drift in cents, the
    pitch correlation, the per-frame drift and the flagged time ranges in seconds.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f'Source vocal not found: {source_path}')
    if not os.path.exists(generated_path):
        raise FileNotFoundError(f'Generated vocal not found: {generated_path}')

    source_f0 = _pitch_track(source_path, f0_method, device, is_half)
    generated_f0 = _pitch_track(generated_path, f0_method, device, is_half)

    length = min(source_f0.shape[0], generated_f0.shape[0])
    if length == 0:
        raise ValueError('Pitch tracking returned no frames for one of the takes.')
    source_f0 = source_f0[:length]
    generated_f0 = generated_f0[:length]

    drift, voiced = _cents(source_f0, generated_f0)
    voiced_frames = int(voiced.sum())
    if voiced_frames == 0:
        raise ValueError('No frame is voiced in both takes, nothing to compare.')

    abs_drift = np.abs(drift[voiced])
    median_drift = float(np.median(abs_drift))
    if voiced_frames > 1 and np.std(source_f0[voiced]) > 0 and np.std(generated_f0[voiced]) > 0:
        correlation = float(np.corrcoef(source_f0[voiced], generated_f0[voiced])[0, 1])
    else:
        correlation = 0.0
    correlation = max(0.0, min(1.0, correlation))

    frame_seconds = HOP_LENGTH / SAMPLE_RATE
    flagged = _ranges(np.abs(drift) > DRIFT_FLAG_CENTS, frame_seconds)
    flagged_seconds = sum(end - begin for begin, end in flagged)
    voiced_seconds = voiced_frames * frame_seconds

    drift_penalty = min(1.0, median_drift / (DRIFT_FLAG_CENTS * 2))
    flagged_penalty = min(1.0, flagged_seconds / voiced_seconds) if voiced_seconds else 0.0
    score = 0.6 * correlation + 0.4 * (1.0 - drift_penalty)
    score = max(0.0, min(1.0, score - 0.5 * flagged_penalty))

    return {
        'score': round(score, 3),
        'median_drift_cents': round(median_drift, 1),
        'pitch_correlation': round(correlation, 3),
        'voiced_seconds': round(voiced_seconds, 2),
        'flagged_seconds': round(flagged_seconds, 2),
        'flagged_ranges': flagged,
        'drift': drift,
        'frame_seconds': frame_seconds,
    }
