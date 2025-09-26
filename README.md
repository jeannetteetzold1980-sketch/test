# TTS Datensatz Formatter (Docker-Version)

Dieses Tool verarbeitet einen Ordner mit Audiodateien, filtert sie nach Qualität und Geschlecht und erstellt einen perfekt formatierten Datensatz für das Training mit Coqui TTS.

## Voraussetzungen

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) muss installiert und gestartet sein.

## 1. Bau des Docker-Images

Öffne eine Kommandozeile (Terminal) in diesem Projektverzeichnis und führe den folgenden Befehl aus. Der Name `tts-formatter` ist frei wählbar. Der Build-Prozess wird einige Zeit dauern, da das große Whisper-Modell heruntergeladen wird.

```bash
docker build -t tts-formatter .```

## 2. Ausführung des Containers

Um das Tool zu verwenden, musst du deine lokalen Ordner mit den Ordnern im Container verbinden.

### Standard-Ausführung (verarbeitet alle Geschlechter)

Erstelle auf deinem Computer einen Ordner für die Eingabe-Audiodateien (z.B. `C:\Audio\Input`) und einen für die Ausgabe (z.B. `C:\Audio\Output`).

```bash
docker run --rm -v "C:\Audio\Input":/app/input -v "C:\Audio\Output":/app/output tts-formatter
```

### Ausführung mit Gender-Filter

Um nur männliche oder weibliche Stimmen zu verarbeiten, füge einfach das `--gender` Flag am Ende des Befehls hinzu:

**Nur männliche Stimmen:**
```bash
docker run --rm -v "C:\Audio\Input":/app/input -v "C:\Audio\Output":/app/output tts-formatter python main_cli.py /app/input /app/output --gender männlich
```

**Nur weibliche Stimmen:**
```bash
docker run --rm -v "C:\Audio\Input":/app/input -v "C:\Audio\Output":/app/output tts-formatter python main_cli.py /app/input /app/output --gender weiblich
```

---
**Hinweise zu den Befehlen:**
- `--rm`: Löscht den Container automatisch nach der Ausführung, um Müll zu vermeiden.
- `-v "DEIN_PFAD":/app/input`: Verbindet (`-v` für Volume) deinen lokalen Ordner mit dem `/app/input`-Ordner *innerhalb* des Containers.
- `tts-formatter`: Der Name des Images, das du in Schritt 1 gebaut hast.