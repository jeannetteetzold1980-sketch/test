# Intelligenter TTS-Datensatz-Prozessor

Dieses Tool automatisiert die Erstellung hochwertiger Datensätze für das Training von Text-to-Speech (TTS)-Modellen wie Coqui TTS. Es durchläuft einen mehrstufigen Prozess, um aus rohen Audiodateien saubere, segmentierte und transkribierte Daten zu generieren.

## Verarbeitungsschritte

Der Prozess ist darauf ausgelegt, die bestmögliche Datenqualität zu gewährleisten:

1.  **Geschlechtserkennung:** Zuerst wird das Geschlecht der sprechenden Person (männlich/weiblich) in jeder Audiodatei identifiziert. Dies ermöglicht die Erstellung geschlechtsspezifischer Datensätze und die Filterung nach Wunsch.

2.  **Qualitätsanalyse und -verbesserung:** Anschließend wird die Audioqualität bewertet. Das Tool versucht, häufige Probleme wie Hintergrundgeräusche oder geringe Lautstärke zu erkennen und automatisch zu korrigieren, um die Klarheit der Stimme zu verbessern.

3.  **Intelligente Segmentierung:** Anstatt die Audiodateien in feste Zeitabschnitte zu zerlegen, segmentiert das Tool die Aufnahmen basierend auf natürlichen Sprechpausen und Satzenden. Dadurch wird sichergestellt, dass Sätze und Gedanken logisch zusammenhängen und nicht abrupt abgeschnitten werden.

4.  **Transkription:** Jedes einzelne logische Segment wird mithilfe des leistungsstarken Whisper-Modells von OpenAI präzise transkribiert.

5.  **TTS-konforme Ausgabe:** Zum Schluss werden die Audio-Segmente zusammen mit ihren Transkriptionen in einem Format gespeichert, das den Anforderungen von Coqui TTS entspricht (typischerweise eine `metadata.csv`-Datei und ein Ordner mit `.wav`-Dateien).

## Voraussetzungen

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) muss installiert und gestartet sein.

## 1. Bau des Docker-Images

Öffne eine Kommandozeile (Terminal) in diesem Projektverzeichnis und führe den folgenden Befehl aus. Der Name `tts-formatter` ist frei wählbar. Der Build-Prozess wird einige Zeit dauern, da das große Whisper-Modell heruntergeladen wird.

```bash
docker build -t tts-formatter .
```

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
