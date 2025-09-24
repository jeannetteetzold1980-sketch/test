import os
import sys
import PySimpleGUI as sg
import threading
import queue
import whisper

from core import VoiceGenderClassifier, process_files

def process_files_worker(
    filepaths, model, update_queue, gender_choice, stop_event, pause_event, classifier
):
    def log_callback(message):
        update_queue.put(("log", message))

    def progress_callback(current, total, filename):
        update_queue.put(("update_overall_progress", current, total, filename))

    result = process_files(
        filepaths,
        model,
        classifier,
        gender_choice,
        stop_event,
        pause_event,
        progress_callback,
        log_callback,
    )

    if result and "error" in result:
        update_queue.put(("error", result["error"]))
    elif result:
        update_queue.put(
            (
                "processing_complete",
                result["total_files"],
                result["total_segments_usable"],
                result["files_skipped"],
                result["zip_path"],
            )
        )
    else:
        update_queue.put(("stopped",))


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
        try:
            model = whisper.load_model("medium")
            print("INFO: Alle Ressourcen geladen. Bereit.")
            window.write_event_value("-MODEL_LOADED-", "")
        except Exception as e:
            sg.popup_error(f"Fehler beim Laden des Whisper-Modells: {e}")
            window.close()

    threading.Thread(target=load_resources, daemon=True).start()

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "-ADD-":
            window["-FILE_BROWSER-"].click()
        elif event == "-FILE_BROWROWSER-":
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