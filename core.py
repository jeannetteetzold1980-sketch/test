import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from num2words import num2words
import re
from sklearn.linear_model import LogisticRegression
import os
import datetime
import shutil
import traceback

GERMAN_INITIAL_PROMPT = (
    "Dies ist eine Transkription auf Deutsch. Sie enthält Umlaute wie ä, ö, ü und auch das ß."
)
BASE_OUTPUT_DIR = "results"

class VoiceGenderClassifier:
    def __init__(self):
        # Standardmodell (Fallback)
        self._weights = np.array(
            [
                -1.88,
                -0.06,
                0.46,
                0.14,
                0.04,
                -0.01,
                0.01,
                -0.01,
                -0.16,
                -0.05,
                0.0,
                -0.01,
                -0.13,
                -0.11,
            ]
        )
        self._bias = 0.81
        self._mean_features = np.array(
            [
                1.51e02,
                -2.13e01,
                2.38e01,
                -4.13e00,
                6.78e00,
                -2.28e00,
                2.65e00,
                -2.53e00,
                -2.07e-01,
                -2.12e00,
                -8.76e-01,
                -1.13e00,
                -1.97e00,
                -2.87e00,
            ]
        )
        self._std_dev_features = np.array(
            [
                4.47e01,
                1.25e01,
                8.87e00,
                6.42e00,
                5.02e00,
                4.34e00,
                3.82e00,
                3.52e00,
                3.42e00,
                3.03e00,
                2.94e00,
                2.76e00,
                2.69e00,
                2.52e00,
            ]
        )
        self.custom_model = None

    def _extract_features(self, y, sr):
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        valid_f0 = f0[~np.isnan(f0)]
        pitch = np.mean(valid_f0) if len(valid_f0) > 0 else 150.0
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        return np.hstack(([pitch], mfccs))

    def predict(self, y, sr):
        features = self._extract_features(y, sr).reshape(1, -1)

        if self.custom_model:
            return self.custom_model.predict(features)[0]

        # Fallback
        features_scaled = (features - self._mean_features) / self._std_dev_features
        z = np.dot(self._weights, features_scaled.flatten()) + self._bias
        probability = 1 / (1 + np.exp(-z))
        return "männlich" if probability > 0.5 else "weiblich"

    def calibrate(self, labeled_data):
        X, y = [], []
        for filepath, label in labeled_data:
            try:
                audio, sr = librosa.load(filepath, sr=16000, mono=True)
                feats = self._extract_features(audio, sr)
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"Fehler beim Laden {filepath}: {e}")

        if len(X) >= 5:
            model = LogisticRegression(max_iter=200)
            model.fit(X, y)
            self.custom_model = model
            return True
        return False

def normalize_text(text):
    def number_to_words(match):
        return num2words(int(match.group(0)), lang="de")

    text = re.sub(r"\d+", number_to_words, text)
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def transcribe_and_normalize(segment_filepath, model, log_callback):
    try:
        result = model.transcribe(
            segment_filepath, language="de", initial_prompt=GERMAN_INITIAL_PROMPT
        )
        raw_transcript = result["text"].strip()
        log_callback("  Segment transkribiert.")
        if raw_transcript:
            return normalize_text(raw_transcript)
        return None
    except Exception as e:
        log_callback(f"  WARNUNG: Transkriptionsfehler: {e}")
        return None

def process_files(
    filepaths,
    model,
    classifier,
    gender_choice,
    stop_event,
    pause_event,
    progress_callback,
    log_callback,
):
    session_path = None
    try:
        session_folder = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_path = os.path.join(BASE_OUTPUT_DIR, f"session_{session_folder}")
        wavs_folder = os.path.join(session_path, "wavs")
        os.makedirs(wavs_folder, exist_ok=True)
        metadata_collector = []
        total_segments_usable = 0
        files_skipped = 0

        for i, filepath in enumerate(filepaths):
            if stop_event.is_set():
                return
            while pause_event.is_set():
                stop_event.wait(0.1)

            progress_callback(i, len(filepaths), os.path.basename(filepath))
            log_callback(f"\n----- {i+1}/{len(filepaths)}: Analysiere '{os.path.basename(filepath)}' -----")

            try:
                y, sr = librosa.load(filepath, sr=16000, mono=True)
            except Exception as e:
                log_callback(f"  FEHLER: Konnte '{os.path.basename(filepath)}' nicht laden: {e}")
                continue

            if gender_choice != "alle":
                detected_gender = classifier.predict(y, sr)
                log_callback(f"  Analyse: Stimme als '{detected_gender}' erkannt.")
                if detected_gender != gender_choice:
                    log_callback(f"-> WIRD ÜBERSPRUNGEN (Filter: '{gender_choice}')")
                    files_skipped += 1
                    continue

            audio = AudioSegment(
                data=(y * (2**15)).astype(np.int16).tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            segments = split_on_silence(
                audio, min_silence_len=700, silence_thresh=-40, keep_silence=300
            )
            if not segments:
                segments = [audio]

            for j, segment in enumerate(segments):
                if stop_event.is_set():
                    return
                while pause_event.is_set():
                    stop_event.wait(0.1)

                duration = len(segment) / 1000.0
                if 2.5 < duration < 12.0 and segment.dBFS > -35.0:
                    unique_wav_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{j+1:03d}.wav"
                    full_wav_path = os.path.join(wavs_folder, unique_wav_filename)
                    segment.export(full_wav_path, format="wav")
                    normalized_transcript = transcribe_and_normalize(
                        full_wav_path, model, log_callback
                    )
                    if normalized_transcript:
                        metadata_collector.append(
                            f"{unique_wav_filename}|{normalized_transcript}"
                        )
                        total_segments_usable += 1
                    else:
                        os.remove(full_wav_path)

        zip_path = shutil.make_archive(
            os.path.join(BASE_OUTPUT_DIR, session_folder), "zip", session_path
        )
        shutil.rmtree(session_path)
        return {
            "total_files": len(filepaths),
            "total_segments_usable": total_segments_usable,
            "files_skipped": files_skipped,
            "zip_path": zip_path,
        }

    except Exception:
        error_details = traceback.format_exc()
        return {"error": error_details}
    finally:
        if stop_event.is_set() and session_path and os.path.exists(session_path):
            shutil.rmtree(session_path)
            log_callback("\nINFO: Verarbeitung abgebrochen.")