"""
Live RTSP -> Demucs -> horn stem -> live COLREG detection + playback.

Requirements:
- ffmpeg & ffplay installed (ffplay used to play the horn stem)
- torch, demucs, numpy, librosa installed
- A Demucs checkpoint saved in CHECKPOINT_PATH
"""

import asyncio
import numpy as np
import torch
import time
import sys
import platform
from demucs.apply import apply_model
import librosa
import logging

# -------------------------
# Config
# -------------------------
RTSP_URL = "rtsp://127.0.0.1:8554/mystream"
CHECKPOINT_PATH = "outputs/xps/97d170e1/best.th"

SAMPLE_RATE = 44100
CHANNELS = 2
CHUNK_DURATION = 10.0      # Long enough for prolonged blasts
OVERLAP = 0.25
SLIDING_WINDOW = 30.0      # Seconds for detection buffer

FRAME_LENGTH = 2048
HOP_LENGTH = 512
ENERGY_THRESH = 0.1        # Matches file-based code
MIN_BLAST = 0.2
MIN_SILENCE = 0.3
SHORT_MAX = 1.0            # Matches file-based code
GAP_TO_NEW_SIGNAL = 3.0    # Gap >3s starts a new signal

# Derived
SAMPLES_PER_CHUNK = int(SAMPLE_RATE * CHUNK_DURATION)
BYTES_PER_SAMPLE = 4
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * CHANNELS * BYTES_PER_SAMPLE
WINDOW_SAMPLES = int(SAMPLE_RATE * SLIDING_WINDOW)

# Added for overlapping Demucs processing to reduce chunk boundary artifacts
OVERLAP_SECONDS = 2.0
SAMPLES_OVERLAP = int(SAMPLE_RATE * OVERLAP_SECONDS)

# COLREG patterns
COLREG_SIGNALS = {
    "short": "I am altering my course to starboard",
    "short short": "I am altering my course to port",
    "short short short": "I am operating astern propulsion",
    "prolonged": "Warning / underway in fog",
    "prolonged prolonged": "I am a vessel being towed / making way",
    "prolonged short": "I intend to overtake you on your starboard side",
    "prolonged prolonged short short": "I intend to overtake you on your port side",
}

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("colreg_detection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# -------------------------
# Utilities for segmentation
# -------------------------
def segment_blasts(y, sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
                   energy_thresh=ENERGY_THRESH, min_blast=MIN_BLAST,
                   min_silence=MIN_SILENCE, start_time=0.0):
    """Return list of (start, end, duration) blasts detected in mono signal y."""
    if len(y) < 2:
        logger.warning("Empty or too short signal for segmentation")
        return [], None, None

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length) + start_time

    rms_norm = rms / (np.max(rms) + 1e-10)
    rms_smooth = np.convolve(rms_norm, np.ones(5)/5, mode="same")  # Match file-based smoothing

    active = rms_smooth > energy_thresh

    blasts = []
    start = None
    last_end = start_time

    for i, flag in enumerate(active):
        t = float(times[i])
        if flag and start is None and (t - last_end) >= min_silence:
            start = t
            logger.debug(f"Blast started at {t:.2f}s")
        elif not flag and start is not None:
            end = t
            duration = end - start
            if duration >= min_blast:
                blasts.append((start, end, duration))
                logger.debug(f"Blast ended: {start:.2f}s to {end:.2f}s, duration {duration:.2f}s")
            start = None
            last_end = end

    if start is not None:
        end = float(times[-1])
        duration = end - start
        if duration >= min_blast:
            blasts.append((start, end, duration))
            logger.debug(f"Blast ended at signal end: {start:.2f}s to {end:.2f}s, duration {duration:.2f}s")

    return blasts, rms_smooth, times

def classify_blast(duration, short_max=SHORT_MAX):
    label = "short" if duration <= short_max else "prolonged"
    logger.debug(f"Classified blast duration {duration:.2f}s as {label}")
    return label

# -------------------------
# Model loader & inference
# -------------------------
def load_demucs_model(checkpoint_path, device):
    pkg = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    logger.info(f"Model keys: {pkg.keys()}")
    klass = pkg["klass"]
    args = pkg.get("args", [])
    kwargs = pkg.get("kwargs", {})
    model = klass(*args, **kwargs)
    model.load_state_dict(pkg["state"])
    model.to(device)
    model.eval()
    return model

def run_demucs_inference(model, wav_tensor, device, horn_idx):
    """Run demucs and return mono horn numpy array (float32)."""
    try:
        wav = wav_tensor.to(device)
        with torch.no_grad():
            out = apply_model(model, wav, split=True, overlap=OVERLAP, device=device)
            horn = out[0, horn_idx]
            if horn.dim() == 2:
                horn_mono = horn.mean(0).cpu().numpy().astype(np.float32)
            else:
                horn_mono = horn.cpu().numpy().astype(np.float32)
            return horn_mono
    except Exception as e:
        logger.error(f"Demucs inference error: {repr(e)}")
        return np.zeros(wav_tensor.shape[-1], dtype=np.float32)  # Return zeros of input length

# -------------------------
# FFmpeg / FFplay helpers
# -------------------------
async def start_ffmpeg_reader(rtsp_url, sample_rate, channels):
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vn",
        "-acodec", "pcm_f32le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "f32le",
        "pipe:1",
        "-loglevel", "error",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    return proc

async def start_ffplay_player(sample_rate):
    cmd = [
        "ffplay",
        "-autoexit",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-nodisp",
        "-",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    return proc

# -------------------------
# Main live processing
# -------------------------
async def process_live():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    model = load_demucs_model(CHECKPOINT_PATH, device=device)

    sources = getattr(model, "sources", None)
    horn_idx = 0
    if sources:
        try:
            horn_idx = list(sources).index("horns")
            logger.info(f"Using horn index: {horn_idx}")
        except ValueError:
            logger.warning("'horns' not found in sources; using index 0")

    player_proc = await start_ffplay_player(SAMPLE_RATE)
    logger.info(f"ffplay started (pid {player_proc.pid})")

    restart_delay = 2.0

    while True:
        proc = None
        stream_start_time = time.time()
        samples_processed = 0
        sliding_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        active_signal = None
        processed_until_time = 0.0
        previous_raw = np.zeros((SAMPLES_OVERLAP, CHANNELS), dtype=np.float32)
        logger.info("Starting new stream loop")

        try:
            proc = await start_ffmpeg_reader(RTSP_URL, SAMPLE_RATE, CHANNELS)
            logger.info(f"FFmpeg reader started (pid {proc.pid})")

            while True:
                raw = await proc.stdout.readexactly(BYTES_PER_CHUNK)
                arr = np.frombuffer(raw, dtype=np.float32)
                if arr.size != SAMPLES_PER_CHUNK * CHANNELS:
                    arr = np.pad(arr, (0, SAMPLES_PER_CHUNK * CHANNELS - arr.size)) if arr.size < SAMPLES_PER_CHUNK * CHANNELS else arr[:SAMPLES_PER_CHUNK * CHANNELS]

                chunk_stereo = arr.reshape(-1, CHANNELS)
                chunk_start_time = stream_start_time + (samples_processed / SAMPLE_RATE)
                chunk_end_time = chunk_start_time + CHUNK_DURATION
                samples_processed += SAMPLES_PER_CHUNK

                # Add overlap from previous raw for continuous Demucs processing
                extended_chunk = np.concatenate((previous_raw, chunk_stereo), axis=0)
                wav = torch.from_numpy(extended_chunk.copy()).float().permute(1, 0).unsqueeze(0)
                horn_mono_full = await asyncio.to_thread(run_demucs_inference, model, wav, device, horn_idx)
                horn_mono = horn_mono_full[SAMPLES_OVERLAP:]
                previous_raw = chunk_stereo[-SAMPLES_OVERLAP:] if SAMPLES_OVERLAP < len(chunk_stereo) else chunk_stereo

                if len(horn_mono) != SAMPLES_PER_CHUNK:
                    logger.warning(f"Horn mono length mismatch: {len(horn_mono)} vs {SAMPLES_PER_CHUNK}. Padding/truncating.")
                    if len(horn_mono) < SAMPLES_PER_CHUNK:
                        horn_mono = np.pad(horn_mono, (0, SAMPLES_PER_CHUNK - len(horn_mono)))
                    else:
                        horn_mono = horn_mono[:SAMPLES_PER_CHUNK]

                # Playback
                try:
                    player_proc.stdin.write(horn_mono.tobytes())
                    await player_proc.stdin.drain()
                except Exception:
                    logger.warning("ffplay error, restarting player")
                    try: player_proc.kill()
                    except: pass
                    player_proc = await start_ffplay_player(SAMPLE_RATE)
                    logger.info(f"ffplay restarted (pid {player_proc.pid})")

                # Sliding buffer update
                n = len(horn_mono)
                if n >= WINDOW_SAMPLES:
                    sliding_buffer = horn_mono[-WINDOW_SAMPLES:]
                    buffer_start_time = chunk_end_time - SLIDING_WINDOW
                else:
                    sliding_buffer = np.roll(sliding_buffer, -n)
                    sliding_buffer[-n:] = horn_mono
                    buffer_start_time = max(0.0, chunk_end_time - SLIDING_WINDOW)

                # Blast segmentation
                blasts, _, _ = segment_blasts(sliding_buffer, SAMPLE_RATE,
                                              frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
                                              energy_thresh=ENERGY_THRESH, min_blast=MIN_BLAST,
                                              min_silence=MIN_SILENCE, start_time=buffer_start_time)
                blasts.sort(key=lambda x: x[0])

                # Process blasts
                for bstart, bend, bdur in blasts:
                    if bend <= processed_until_time + 1e-6:
                        continue  # Skip old blasts
                    if bstart <= processed_until_time + 1e-6 < bend:
                        # Update ongoing blast
                        if active_signal is not None and active_signal["blasts"]:
                            old_start, _, _ = active_signal["blasts"][-1]
                            if (bstart - old_start) <= (SHORT_MAX + MIN_SILENCE):
                                active_signal["blasts"][-1] = (old_start, bend, bend - old_start)
                                active_signal["last_end"] = bend
                                logger.debug(f"Extended blast: {old_start:.2f}s to {bend:.2f}s, duration {bend - old_start:.2f}s")
                            else:
                                logger.warning(f"Unexpected ongoing blast at {bstart:.2f}s, treating as new")
                                active_signal["blasts"].append((bstart, bend, bdur))
                                active_signal["last_end"] = bend
                        processed_until_time = max(processed_until_time, bend)
                    else:
                        # New blast
                        if active_signal is None:
                            active_signal = {"blasts": [(bstart, bend, bdur)], "first_start": bstart, "last_end": bend}
                            logger.debug(f"New pattern started with blast: {bstart:.2f}s to {bend:.2f}s")
                        else:
                            gap = bstart - active_signal["last_end"]
                            if gap >= GAP_TO_NEW_SIGNAL:
                                # Finalize current pattern
                                pattern = " ".join([classify_blast(d) for (_, _, d) in active_signal["blasts"]])
                                meaning = COLREG_SIGNALS.get(pattern, "Unknown signal")
                                logger.info(f"FINAL SIGNAL @ {active_signal['first_start']:.2f}s -> pattern: '{pattern}' -> {meaning}")
                                # Start new pattern
                                active_signal = {"blasts": [(bstart, bend, bdur)], "first_start": bstart, "last_end": bend}
                                logger.debug(f"New pattern started after gap {gap:.2f}s")
                            else:
                                # Append to current pattern
                                active_signal["blasts"].append((bstart, bend, bdur))
                                active_signal["last_end"] = bend
                                logger.debug(f"Added blast to pattern: {bstart:.2f}s to {bend:.2f}s, gap {gap:.2f}s")
                        processed_until_time = max(processed_until_time, bend)

                # Finalize pattern if idle
                if active_signal is not None and (chunk_end_time - active_signal["last_end"] >= GAP_TO_NEW_SIGNAL):
                    pattern = " ".join([classify_blast(d) for (_, _, d) in active_signal["blasts"]])
                    meaning = COLREG_SIGNALS.get(pattern, "Unknown signal")
                    logger.info(f"FINAL SIGNAL @ {active_signal['first_start']:.2f}s -> pattern: '{pattern}' -> {meaning}")
                    active_signal = None

                await asyncio.sleep(0.001)

        except (asyncio.IncompleteReadError, EOFError):
            logger.warning("Stream interrupted. Restarting FFmpeg reader...")
            if proc:
                proc.kill()
            await asyncio.sleep(restart_delay)
            continue
        except Exception as e:
            logger.error(f"Error in main loop: {repr(e)}")
            if proc:
                proc.kill()
            await asyncio.sleep(restart_delay)
            continue

# -------------------------
# Entrypoint
# -------------------------
def main():
    if platform.system() == "Emscripten":
        raise RuntimeError("Unsupported platform")
    asyncio.run(process_live())

if __name__ == "__main__":
    main()