import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import whisper
from num2words import num2words
import re
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
import os
import datetime
import shutil
import traceback

GERMAN_INITIAL_PROMPT = (
    "Dies ist eine Transkription auf Deutsch. Sie enthält Umlaute wie ä, ö, ü und auch das ß."
)
BASE_OUTPUT_DIR = "results"

class VoiceGenderClassifier:
    def __init__(self, n_components=5):
        # Initialisiere GMMs für männliche und weibliche Stimmen
        self.male_gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.female_gmm = GaussianMixture(n_components=n_components, random_state=42)
        self.is_fitted = False
        self._train_default_models()

    def _train_default_models(self):
        """
        Trainiert die GMMs mit einem einfachen, repräsentativen Datensatz,
        um ein robustes Standardverhalten sicherzustellen.
        """
        # Repräsentative Merkmale für männliche und weibliche Stimmen
        # Diese Werte simulieren typische Verteilungen für Pitch und MFCCs.
        # Männlich: tiefere Tonhöhe, andere spektrale Eigenschaften
        male_features = np.array([
            [120, -15, 20, -5, 5, -2, 2, -2, 0, -2, -1, -1, -2, -3],
            [130, -18, 22, -4, 6, -3, 3, -3, -1, -2, -1, -1, -2, -3],
            [110, -20, 18, -6, 4, -1, 1, -1, 1, -3, -2, -2, -1, -2]
        ] * 10) # Multipliziere, um genügend Daten für das Training zu haben

        # Weiblich: höhere Tonhöhe, andere spektrale Eigenschaften
        female_features = np.array([
            [200, -10, 25, 0, 10, 0, 5, 0, 2, 0, 1, 1, 0, -1],
            [210, -12, 28, 1, 12, 1, 6, 1, 3, 1, 2, 2, 1, 0],
            [190, -8, 22, -1, 8, -1, 4, -1, 1, -1, 0, 0, -1, -2]
        ] * 10)

        try:
            self.male_gmm.fit(male_features)
            self.female_gmm.fit(female_features)
            self.is_fitted = True
            print("Standard-GMMs für Geschlechtsbestimmung erfolgreich trainiert.")
        except Exception as e:
            print(f"Fehler beim Trainieren der Standard-GMMs: {e}")

    def _extract_features(self, y, sr):
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
        )
        valid_f0 = f0[~np.isnan(f0)]
        # Verwende Median statt Mittelwert für mehr Robustheit
        pitch = np.median(valid_f0) if len(valid_f0) > 0 else 150.0
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        return np.hstack(([pitch], mfccs))

    def predict(self, y, sr):
        if not self.is_fitted:
            return "Modell nicht trainiert"

        features = self._extract_features(y, sr).reshape(1, -1)

        # Berechne die Wahrscheinlichkeit für jedes Modell
        male_score = self.male_gmm.score(features)
        female_score = self.female_gmm.score(features)

        # Gib das Geschlecht des Modells mit der höheren Wahrscheinlichkeit zurück
        return "weiblich" if female_score > male_score else "männlich"

    def calibrate(self, labeled_data):
        """
        Trainiert die GMMs mit benutzerdefinierten, gelabelten Daten neu.
        """
        male_features = []
        female_features = []

        for file_path, label in labeled_data:
            try:
                y, sr = librosa.load(file_path, sr=None)
                features = self._extract_features(y, sr)
                if label.lower() == "männlich":
                    male_features.append(features)
                elif label.lower() == "weiblich":
                    female_features.append(features)
            except Exception as e:
                print(f"Fehler beim Verarbeiten von {file_path} für die Kalibrierung: {e}")

        if not male_features or not female_features:
            print("Nicht genügend Daten für die Kalibrierung. Mindestens eine männliche und eine weibliche Datei erforderlich.")
            return

        try:
            self.male_gmm.fit(np.array(male_features))
            self.female_gmm.fit(np.array(female_features))
            self.is_fitted = True
            print("GMMs erfolgreich mit benutzerdefinierten Daten kalibriert.")
        except Exception as e:
            print(f"Fehler bei der GMM-Kalibrierung: {e}")

def normalize_text(text):
    def number_to_words(match):
        return num2words(int(match.group(0)), lang="de")

    text = re.sub(r"\d+", number_to_words, text)
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def enhance_audio(y, sr, log_callback):
    try:
        log_callback("  Verbessere Audioqualität: Reduziere Rauschen...")
        # Konvertiere zu float32, falls es nicht schon so ist
        if y.dtype != np.float32:
            y = y.astype(np.float32)
        
        # Führe Rauschreduktion durch
        reduced_noise_y = noisereduce.reduce_noise(y=y, sr=sr, stationary=True)
        
        log_callback("  Verbessere Audioqualität: Normalisiere Lautstärke...")
        # Konvertiere zu pydub AudioSegment für die Normalisierung
        audio_segment = AudioSegment(
            data=(reduced_noise_y * (2**15)).astype(np.int16).tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1
        )
        
        # Normalisiere die Lautstärke auf -20 dBFS
        normalized_segment = audio_segment.normalize()
        
        # Konvertiere zurück zu numpy array
        y_normalized = np.array(normalized_segment.get_array_of_samples(), dtype=np.float32) / (2**15)
        
        log_callback("  Audio-Verbesserung abgeschlossen.")
        return y_normalized, sr
    except Exception as e:
        log_callback(f"  WARNUNG: Fehler bei der Audio-Verbesserung: {e}")
        # Gib das Original-Audio zurück, wenn ein Fehler auftritt
        return y, sr

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
            
            # 1. Audio verbessern
            y_enhanced, sr_enhanced = enhance_audio(y, sr, log_callback)

            # 2. Transkribieren mit Wort-Zeitstempeln
            log_callback("  Transkribiere Audio für die logische Segmentierung...")
            try:
                result = model.transcribe(
                    y_enhanced,
                    language="de",
                    initial_prompt=GERMAN_INITIAL_PROMPT,
                    word_timestamps=True,
                )
                word_segments = result.get("segments", [{"words": result.get("words")}])
                if not word_segments or not word_segments[0].get('words'):
                    log_callback("  WARNUNG: Keine Wörter in der Transkription gefunden. Datei wird übersprungen.")
                    continue
            except Exception as e:
                log_callback(f"  FEHLER bei der Transkription für Segmentierung: {e}")
                continue

            # 3. Logische Segmente erstellen
            log_callback("  Erstelle logische Segmente basierend auf Pausen...")
            audio_full = AudioSegment(
                data=(y_enhanced * (2**15)).astype(np.int16).tobytes(),
                sample_width=2, frame_rate=sr_enhanced, channels=1
            )
            
            current_segment_words = []
            current_segment_start_time = word_segments[0]['words'][0]['start']
            
            all_words = [word for seg in word_segments for word in seg.get('words', [])]

            for j in range(len(all_words) - 1):
                word = all_words[j]
                next_word = all_words[j+1]
                
                current_segment_words.append(word['word'])
                pause_duration = next_word['start'] - word['end']

                # Logische Trennung bei Pausen > 0.7s oder am Ende
                if pause_duration > 0.7:
                    segment_end_time = word['end']
                    transcript_text = "".join(current_segment_words)
                    
                    # Segment extrahieren und verarbeiten
                    segment_audio = audio_full[current_segment_start_time*1000:segment_end_time*1000]
                    duration = len(segment_audio) / 1000.0

                    if 2.5 < duration < 12.0 and segment_audio.dBFS > -35.0:
                        normalized_transcript = normalize_text(transcript_text)
                        if normalized_transcript:
                            unique_wav_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{total_segments_usable+1:03d}.wav"
                            full_wav_path = os.path.join(wavs_folder, unique_wav_filename)
                            segment_audio.export(full_wav_path, format="wav")
                            metadata_collector.append(f"{unique_wav_filename}|{normalized_transcript}")
                            total_segments_usable += 1
                    
                    # Nächstes Segment starten
                    current_segment_words = []
                    current_segment_start_time = next_word['start']

            # Letztes Segment verarbeiten
            if current_segment_words:
                current_segment_words.append(all_words[-1]['word'])
                segment_end_time = all_words[-1]['end']
                transcript_text = "".join(current_segment_words)
                segment_audio = audio_full[current_segment_start_time*1000:segment_end_time*1000]
                duration = len(segment_audio) / 1000.0
                if 2.5 < duration < 12.0 and segment_audio.dBFS > -35.0:
                    normalized_transcript = normalize_text(transcript_text)
                    if normalized_transcript:
                        unique_wav_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_{total_segments_usable+1:03d}.wav"
                        full_wav_path = os.path.join(wavs_folder, unique_wav_filename)
                        segment_audio.export(full_wav_path, format="wav")
                        metadata_collector.append(f"{unique_wav_filename}|{normalized_transcript}")
                        total_segments_usable += 1

        if not metadata_collector:
            log_callback("\nWARNUNG: Keine verwertbaren Segmente nach der Verarbeitung gefunden.")
            return {"error": "No usable segments found."}

        # Metadaten-Datei schreiben
        with open(os.path.join(session_path, "metadata.csv"), "w", encoding="utf-8") as f:
            for line in metadata_collector:
                f.write(line + "\n")

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
        log_callback(f"\nFATALER FEHLER: {error_details}")
        return {"error": error_details}
    finally:
        if stop_event.is_set() and session_path and os.path.exists(session_path):
            shutil.rmtree(session_path)
            log_callback("\nINFO: Verarbeitung abgebrochen.")