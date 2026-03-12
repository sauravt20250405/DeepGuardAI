"""
DeepGuard AI — Audio Forensic Analysis Module
Extracts audio from video files and analyzes spectral features to detect synthetic/manipulated speech.
"""

import subprocess
import tempfile
import os
import numpy as np

def extract_audio_from_video(video_path, output_path=None):
    """Extract audio track from a video file using ffmpeg/ffprobe."""
    if output_path is None:
        output_path = tempfile.mktemp(suffix='.wav')
    
    try:
        # Use ffmpeg to extract audio as WAV
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vn',           # no video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '22050',  # 22kHz sample rate (good for speech analysis)
            '-ac', '1',      # mono
            '-y',            # overwrite
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
        return None
    except Exception as e:
        print(f"Audio extraction error: {e}")
        return None


def analyze_audio(audio_path):
    """
    Analyze audio for signs of synthetic speech using spectral forensics.
    Returns a dict with score (0-100, higher = more likely fake) and verdict.
    """
    try:
        import librosa
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        
        if len(y) < sr * 0.5:  # Less than 0.5 seconds of audio
            return {"score": 0, "verdict": "Insufficient Audio", "details": {}}
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        if len(y_trimmed) < sr * 0.3:
            return {"score": 0, "verdict": "No Speech Detected", "details": {}}
        
        # ============ SPECTRAL FEATURES ============
        
        # 1. Spectral Flatness — Synthetic speech tends to have lower spectral flatness
        # (more tonal, less noise-like than natural speech)
        spectral_flatness = librosa.feature.spectral_flatness(y=y_trimmed)[0]
        avg_flatness = np.mean(spectral_flatness)
        flatness_std = np.std(spectral_flatness)
        # Natural speech has varied flatness; synthetic is more uniform
        flatness_score = max(0, 15 - flatness_std * 500) + max(0, 10 - avg_flatness * 200)  # 0-25
        
        # 2. MFCC Variance — Natural speech has high MFCC variance across time
        # Synthetic speech tends to be more consistent
        mfccs = librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13)
        mfcc_vars = np.var(mfccs, axis=1)
        avg_mfcc_var = np.mean(mfcc_vars)
        # Low variance = more uniform = more suspicious
        mfcc_score = max(0, 25 - avg_mfcc_var / 5)  # 0-25
        
        # 3. Pitch (F0) Consistency — AI voices have unnaturally consistent pitch
        f0, voiced_flag, _ = librosa.pyin(y_trimmed, fmin=60, fmax=400, sr=sr)
        f0_clean = f0[voiced_flag] if voiced_flag is not None else f0[~np.isnan(f0)]
        f0_clean = f0_clean[~np.isnan(f0_clean)]
        
        if len(f0_clean) > 5:
            pitch_std = np.std(f0_clean)
            pitch_range = np.ptp(f0_clean)
            # Natural speech: pitch_std ~20-60 Hz, AI: ~5-15 Hz
            pitch_score = max(0, 20 - pitch_std / 2)  # 0-20
        else:
            pitch_score = 10  # Neutral
        
        # 4. Zero Crossing Rate Variance — Natural speech has varied ZCR
        zcr = librosa.feature.zero_crossing_rate(y_trimmed)[0]
        zcr_var = np.var(zcr)
        zcr_score = max(0, 15 - zcr_var * 5000)  # 0-15
        
        # 5. Spectral Rolloff Consistency — AI audio has more uniform rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr)[0]
        rolloff_std = np.std(rolloff)
        rolloff_score = max(0, 15 - rolloff_std / 500)  # 0-15
        
        # ============ COMBINE ============
        total_score = flatness_score + mfcc_score + pitch_score + zcr_score + rolloff_score
        total_score = max(5.0, min(95.0, total_score))
        
        is_fake = total_score > 55.0
        
        details = {
            "flatness": round(flatness_score, 1),
            "mfcc_var": round(mfcc_score, 1),
            "pitch": round(pitch_score, 1),
            "zcr": round(zcr_score, 1),
            "rolloff": round(rolloff_score, 1),
            "pitch_std_hz": round(float(pitch_std), 1) if len(f0_clean) > 5 else 0,
        }
        
        return {
            "score": round(total_score, 1),
            "verdict": "Synthetic Voice Detected" if is_fake else "Natural Voice Pattern",
            "details": details
        }
        
    except Exception as e:
        print(f"Audio analysis error: {e}")
        return {"score": 0, "verdict": f"Analysis Failed: {str(e)[:50]}", "details": {}}


def analyze_video_audio(video_path):
    """
    Full pipeline: extract audio from video → analyze → return results.
    """
    audio_path = extract_audio_from_video(video_path)
    
    if audio_path is None:
        return {"score": 0, "verdict": "No Audio Track", "details": {}}
    
    try:
        result = analyze_audio(audio_path)
    finally:
        # Clean up temp audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
    
    return result
