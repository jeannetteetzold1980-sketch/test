
import argparse
import os
import sys
import whisper
from core import VoiceGenderClassifier, process_files
import threading

def log_to_console(message):
    """Gibt eine Nachricht auf der Konsole aus."""
    print(message, file=sys.stdout)

def progress_dummy(current, total, filename):
    """Eine Dummy-Funktion für den Fortschritt, die nichts tut."""
    pass

def main():
    """
    Hauptfunktion für das Kommandozeilen-Tool zur Verarbeitung von Audiodateien
    für das TTS-Datensatz-Training.
    """
    parser = argparse.ArgumentParser(
        description="Verarbeitet Audiodateien, um einen Datensatz für Coqui TTS zu formatieren."
    )
    parser.add_argument(
        "input_dir",
        help="Pfad zum Eingabeverzeichnis mit den Audiodateien."
    )
    parser.add_argument(
        "output_dir",
        help="Pfad zum Ausgabeverzeichnis, in dem der formatierte Datensatz gespeichert wird."
    )
    parser.add_argument(
        "--gender",
        choices=["männlich", "weiblich", "alle"],
        default="alle",
        help="Filtert die Stimmen nach Geschlecht. Standard: 'alle'."
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Name des zu verwendenden Whisper-Modells. Standard: 'large-v3'."
    )

    args = parser.parse_args()

    # Überprüfen, ob die Verzeichnisse existieren
    if not os.path.isdir(args.input_dir):
        log_to_console(f"FEHLER: Das Eingabeverzeichnis '{args.input_dir}' wurde nicht gefunden.")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)

    log_to_console("Initialisiere Modelle...")
    try:
        model = whisper.load_model(args.model)
        classifier = VoiceGenderClassifier()
    except Exception as e:
        log_to_console(f"FEHLER beim Initialisieren der Modelle: {e}")
        sys.exit(1)

    log_to_console(f"Suche nach Audiodateien in: {args.input_dir}")
    filepaths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    
    if not filepaths:
        log_to_console("Keine Dateien im Eingabeverzeichnis gefunden.")
        sys.exit(0)

    log_to_console(f"{len(filepaths)} Dateien gefunden. Starte Verarbeitung...")
    log_to_console(f"Geschlechterfilter: {args.gender}")

    # Dummy-Events für die Kompatibilität mit process_files
    stop_event = threading.Event()
    pause_event = threading.Event()

    # Starte die Verarbeitung
    result = process_files(
        filepaths=filepaths,
        model=model,
        classifier=classifier,
        gender_choice=args.gender,
        stop_event=stop_event,
        pause_event=pause_event,
        progress_callback=progress_dummy,
        log_callback=log_to_console,
    )

    if result and "error" in result:
        log_to_console(f"\nEin Fehler ist aufgetreten:\n{result['error']}")
        sys.exit(1)
    elif result:
        log_to_console("\nVerarbeitung erfolgreich abgeschlossen.")
        log_to_console(f"  - Verarbeitete Dateien: {result.get('total_files', 0)}")
        log_to_console(f"  - Verwertbare Segmente: {result.get('total_segments_usable', 0)}")
        log_to_console(f"  - Übersprungene Dateien (Gender-Filter): {result.get('files_skipped', 0)}")
        log_to_console(f"  - Datensatz gespeichert als: {result.get('zip_path', 'N/A')}")
    else:
        log_to_console("\nVerarbeitung abgeschlossen, aber es wurden keine Ergebnisse zurückgegeben.")

if __name__ == "__main__":
    main()
