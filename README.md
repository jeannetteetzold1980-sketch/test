# TTS Datensatz Formatter (Docker-Version)

Dieses Tool verarbeitet einen Ordner mit Audiodateien, filtert sie nach Qualit�t und Geschlecht und erstellt einen perfekt formatierten Datensatz f�r das Training mit Coqui TTS.

## Voraussetzungen

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) muss installiert und gestartet sein.

## 1. Bau des Docker-Images

�ffne eine Kommandozeile (Terminal) in diesem Projektverzeichnis und f�hre den folgenden Befehl aus. Der Name `tts-formatter` ist frei w�hlbar. Der Build-Prozess wird einige Zeit dauern, da das gro�e Whisper-Modell heruntergeladen wird.

```bash
docker build -t tts-formatter .```

## 2. Ausf�hrung des Containers

Um das Tool zu verwenden, musst du deine lokalen Ordner mit den Ordnern im Container verbinden.

### Standard-Ausf�hrung (verarbeitet alle Geschlechter)

Erstelle auf deinem Computer einen Ordner f�r die Eingabe-Audiodateien (z.B. `C:\Audio\Input`) und einen f�r die Ausgabe (z.B. `C:\Audio\Output`).

```bash
docker run --rm -v "C:\Audio\Input":/app/input -v "C:\Audio\Output":/app/output tts-formatter
```

### Ausf�hrung mit Gender-Filter

Um nur m�nnliche oder weibliche Stimmen zu verarbeiten, f�ge einfach das `--gender` Flag am Ende des Befehls hinzu:

**Nur m�nnliche Stimmen:**
```bash
docker run --rm -v "C:\Audio\Input":/app/input -v "C:\Audio\Output":/app/output tts-formatter python main_cli.py /app/input /app/output --gender m�nnlich
```

**Nur weibliche Stimmen:**
```bash
docker run --rm -v "C:\Audio\Input":/app/input -v "C:\Audio\Output":/app/output tts-formatter python main_cli.py /app/input /app/output --gender weiblich
```

---
**Hinweise zu den Befehlen:**
- `--rm`: L�scht den Container automatisch nach der Ausf�hrung, um M�ll zu vermeiden.
- `-v "DEIN_PFAD":/app/input`: Verbindet (`-v` f�r Volume) deinen lokalen Ordner mit dem `/app/input`-Ordner *innerhalb* des Containers.
- `tts-formatter`: Der Name des Images, das du in Schritt 1 gebaut hast.