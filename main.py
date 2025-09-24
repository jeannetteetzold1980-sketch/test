# main.py (Version 9.3.0 - Optimiert mit Kalibrierung)

import os
import sys
import datetime
import shutil
import re
import PySimpleGUI as sg
import threading
import queue
import numpy as np
import traceback

try:
    import librosa
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
    import whisper
    from num2words import num2words
    from sklearn.linear_model import LogisticRegression
except ImportError as e:
    sg.popup_error(
        f"FEHLER: Eine wichtige Bibliothek fehlt: {e}\n\n"
        f"Stellen Sie sicher, dass alle Bibliotheken in 'requirements.txt' installiert sind."
    )
    sys.exit(1)

# --- KONFIGURATION ---
BASE_OUTPUT_DIR = "results"
GERMAN_INITIAL_PROMPT = (
    "Dies ist eine Transkription auf Deutsch. Sie enthält Umlaute wie ä, ö, ü und auch das ß."
)


# --- KLASSE: Gender-Classifier ---
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
        return "weiblich" if probability > 0.5 else "männlich"

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


# --- TEXTHILFSFUNKTION ---
def normalize_text(text):
    def number_to_words(match):
        return num2words(int(match.group(0)), lang="de")

    text = re.sub(r"\d+", number_to_words, text)
    text = text.lower()
    text = re.sub(r"[^a-zäöüß\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# --- TRANSKRIPTION ---
def transcribe_and_normalize(segment_filepath, model, update_queue):
    try:
        result = model.transcribe(
            segment_filepath, language="de", initial_prompt=GERMAN_INITIAL_PROMPT
        )
        raw_transcript = result["text"].strip()
        update_queue.put(("log", "  Segment transkribiert."))
        if raw_transcript:
            return normalize_text(raw_transcript)
        return None
    except Exception as e:
        update_queue.put(("log", f"  WARNUNG: Transkriptionsfehler: {e}"))
        return None


# --- HAUPTVERARBEITUNG ---
def process_files_worker(
    filepaths, model, update_queue, gender_choice, stop_event, pause_event, classifier
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
                update_queue.put(("stopped",))
                return
            while pause_event.is_set():
                stop_event.wait(0.1)

            update_queue.put(
                ("update_overall_progress", i, len(filepaths), os.path.basename(filepath))
            )
            update_queue.put(
                ("log", f"\n----- {i+1}/{len(filepaths)}: Analysiere '{os.path.basename(filepath)}' -----")
            )

            try:
                y, sr = librosa.load(filepath, sr=16000, mono=True)
            except Exception as e:
                update_queue.put(
                    ("log", f"  FEHLER: Konnte '{os.path.basename(filepath)}' nicht laden: {e}")
                )
                continue

            if gender_choice != "alle":
                detected_gender = classifier.predict(y, sr)
                update_queue.put(("log", f"  Analyse: Stimme als '{detected_gender}' erkannt."))
                if detected_gender != gender_choice:
                    update_queue.put(("log", f"-> WIRD ÜBERSPRUNGEN (Filter: '{gender_choice}')"))
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
                    update_queue.put(("stopped",))
                    return
                while pause_event.is_set():
                    stop_event.wait(0.1)

                duration = len(segment) / 1000.0
                if 2.5 < duration < 12.0 and segment.dBFS > -35.0:
                    unique_wav_filename = (
                        f"{os.path.splitext(os.path.basename(filepath))[0]}_{j+1:03d}.wav"
                    )
                    full_wav_path = os.path.join(wavs_folder, unique_wav_filename)
                    segment.export(full_wav_path, format="wav")
                    normalized_transcript = transcribe_and_normalize(
                        full_wav_path, model, update_queue
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
        update_queue.put(
            (
                "processing_complete",
                len(filepaths),
                total_segments_usable,
                files_skipped,
                zip_path,
            )
        )

    except Exception:
        error_details = traceback.format_exc()
        update_queue.put(("error", error_details))
    finally:
        if stop_event.is_set() and session_path and os.path.exists(session_path):
            shutil.rmtree(session_path)
            update_queue.put(("log", "\nINFO: Verarbeitung abgebrochen."))


# --- KALIBRIERUNGSDIALOG ---
def calibration_dialog(parent_window, classifier):
    layout = [
        [sg.Text("Bitte wählen Sie 10–20 Audio-Dateien und ordnen Sie das Geschlecht zu.")],
        [sg.Text("Ausgewählte Dateien:")],
        [sg.Listbox(values=[], size=(60, 10), key="-CAL_FILELIST-")],
        [
            sg.FilesBrowse(
                "Dateien auswählen",
                key="-CAL_FILES-",
                file_types=(("Audio-Dateien", "*.wav;*.mp3;*.flac"),),
                files_delimiter=";",
            ),
            sg.Button("Laden", key="-CAL_LOAD-"),
        ],
        [
            sg.Frame(
                "Label",
                [
                    [sg.Radio("Männlich", "CAL_GENDER", key="-CAL_M-"),
                     sg.Radio("Weiblich", "CAL_GENDER", key="-CAL_F-")]
                ],
            )
        ],
        [
            sg.Button("Datei hinzufügen", key="-CAL_ADD-"),
            sg.Button("Training starten", key="-CAL_TRAIN-", disabled=True),
            sg.Button("Abbrechen"),
        ],
    ]
    window = sg.Window("Kalibrierung", layout, modal=True)

    labeled_data = []

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, "Abbrechen"):
            break
        elif event == "-CAL_LOAD-":
            if values["-CAL_FILES-"]:
                filepaths = values["-CAL_FILES-"].split(";")
                window["-CAL_FILELIST-"].update(filepaths)
        elif event == "-CAL_ADD-":
            selected_files = values["-CAL_FILELIST-"]
            if selected_files:
                gender = (
                    "männlich" if values["-CAL_M-"] else "weiblich" if values["-CAL_F-"] else None
                )
                if gender:
                    for f in selected_files:
                        labeled_data.append((f, gender))
                    sg.popup("Dateien hinzugefügt.")
                    window["-CAL_TRAIN-"].update(disabled=(len(labeled_data) < 5))
                else:
                    sg.popup_error("Bitte Geschlecht auswählen.")
        elif event == "-CAL_TRAIN-":
            if labeled_data:
                success = classifier.calibrate(labeled_data)
                if success:
                    sg.popup("Kalibrierung abgeschlossen!")
                else:
                    sg.popup_error("Nicht genug Daten für Kalibrierung.")
                break

    window.close()


# --- GUI ---
def create_window():
    sg.theme("DarkGrey13")
    file_list_column = [
        [sg.Text("Zu verarbeitende Dateien:")],
        [
            sg.Listbox(
                values=[],
                select_mode=sg.LISTBOX_SELECT_MODE_EXTENDED,
                key="-FILE_LIST-",
                size=(60, 15),
                enable_events=True,
            )
        ],
        [
            sg.Button("Hinzufügen", key="-ADD-"),
            sg.Button("Auswahl entfernen", key="-REMOVE-"),
            sg.Button("Alle entfernen", key="-CLEAR-"),
        ],
        [
            sg.FilesBrowse(
                target=sg.ThisRow,
                key="-FILE_BROWSER-",
                file_types=(("Audio Files", "*.*"), ("All Files", "*.*")),
                enable_events=True,
            )
        ],
    ]
    settings_column = [
        [
            sg.Frame(
                "Geschlechter-Filter",
                [
                    [sg.Radio("Alle", "G", k="-G_A-", default=True)],
                    [sg.Radio("Männlich", "G", k="-G_M-")],
                    [sg.Radio("Weiblich", "G", k="-G_F-")],
                ],
            )
        ],
        [sg.VPush()],
        [sg.Button("Kalibrieren", key="-CALIBRATE-")],
        [sg.Button("TTS-Datensatz generieren", key="-START-", size=(25, 2), disabled=True)],
    ]
    action_buttons = [
        [
            sg.Button("Pause", key="-PAUSE-", size=(12, 2), visible=False),
            sg.Button("Stopp", key="-STOP-", size=(12, 2), visible=False),
        ]
    ]
    layout = [
        [sg.Text("Coqui TTS Datensatz-Formatter", font=("Helvetica", 20))],
        [
            sg.Text(
                "⬇️ Dateien oder Ordner hierher ziehen ⬇️",
                size=(85, 2),
                justification="center",
                pad=(0, 5),
                background_color="gray",
                text_color="white",
            )
        ],
        [sg.Column(file_list_column), sg.VSeperator(), sg.Column(settings_column, element_justification="center")],
        [sg.Frame("Aktionen", action_buttons, key="-ACTION_FRAME-", visible=False, element_justification="center")],
        [sg.Frame("Logs", [[sg.Multiline(size=(85, 12), key="-LOG-", autoscroll=True, reroute_stdout=True, reroute_stderr=True, disabled=True)]])],
    ]
    return sg.Window("TTS-Toolkit v9.3.0 (Optimiert)", layout, finalize=True)


# --- MAIN ---
def main():
    window = create_window()
    model, classifier = None, None
    filepaths = set()
    worker_thread = None
    stop_event, pause_event = None, None

    def update_file_list_and_buttons():
        sorted_paths = sorted(list(filepaths))
        window["-FILE_LIST-"].update(sorted_paths)
        window["-START-"].update(disabled=(not sorted_paths or not (model and classifier)))

    def add_files_to_list(paths_to_add):
        initial_count = len(filepaths)
        for path in paths_to_add:
            if os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if not file.startswith("."):
                            filepaths.add(os.path.join(root, file))
            elif os.path.isfile(path):
                if not os.path.basename(path).startswith("."):
                    filepaths.add(path)
        if len(filepaths) > initial_count:
            print(f"{len(filepaths) - initial_count} neue Datei(en) hinzugefügt.")
            update_file_list_and_buttons()

    def load_resources():
        nonlocal model, classifier
        print("INFO: Initialisiere Gender-Klassifikator...")
        classifier = VoiceGenderClassifier()
        print("INFO: Lade Whisper-Modell 'medium'...")
        model = whisper.load_model("medium")
        print("INFO: Alle Ressourcen geladen. Bereit.")
        window.write_event_value("-MODEL_LOADED-", "")

    threading.Thread(target=load_resources, daemon=True).start()

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "-ADD-":
            window["-FILE_BROWSER-"].click()
        elif event == "-FILE_BROWSER-":
            add_files_to_list(values["-FILE_BROWSER-"].split(";"))
        elif event.endswith("+DRAG_DROP+"):
            add_files_to_list(values[event])
        elif event == "-REMOVE-":
            for item in values["-FILE_LIST-"]:
                filepaths.discard(item)
            update_file_list_and_buttons()
        elif event == "-CLEAR-":
            filepaths.clear()
            update_file_list_and_buttons()
        elif event == "-MODEL_LOADED-":
            update_file_list_and_buttons()
        elif event == "-CALIBRATE-":
            if classifier:
                calibration_dialog(window, classifier)
        elif event == "-START-":
            window["-START-"].update(disabled=True)
            window["-ACTION_FRAME-"].update(visible=True)
            current_file_list = window["-FILE_LIST-"].get_list_values()
            gender_choice = (
                "männlich"
                if values["-G_M-"]
                else "weiblich"
                if values["-G_F-"]
                else "alle"
            )
            print(
                f"INFO: Verarbeitung von {len(current_file_list)} Dateien gestartet. Filter: '{gender_choice}'"
            )
            stop_event, pause_event = threading.Event(), threading.Event()
            update_queue = queue.Queue()
            worker_thread = threading.Thread(
                target=process_files_worker,
                args=(
                    current_file_list,
                    model,
                    update_queue,
                    gender_choice,
                    stop_event,
                    pause_event,
                    classifier,
                ),
                daemon=True,
            )
            worker_thread.start()

    window.close()


if __name__ == "__main__":
    main()
