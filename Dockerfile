# Stage 1: Build-Stage mit Whisper-Modell
FROM python:3.12-slim as builder

# Setze Arbeitsverzeichnis
WORKDIR /app

# Installiere notwendige Build-Tools und Whisper-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Installiere Python-Abhängigkeiten
COPY requirements.txt .

# Installiere torch explizit als CPU-Version, dann die restlichen Abhängigkeiten
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Lade das kleine Whisper-Modell herunter und cache es
RUN python -c "import whisper; whisper.load_model('base')"

# Stage 2: Finale, schlanke Ausführungs-Stage
FROM python:3.12-slim

# Setze Arbeitsverzeichnis
WORKDIR /app

# Installiere nur die notwendigen Laufzeit-Abhängigkeiten
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Kopiere die installierten Python-Pakete aus der Build-Stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Kopiere den Whisper-Modell-Cache aus der Build-Stage
COPY --from=builder /root/.cache/whisper /root/.cache/whisper

# Kopiere den Anwendungscode
COPY . .

# Definiere den Standard-Einstiegspunkt.
# Dieser wird ausgeführt, wenn keine weiteren Argumente an `docker run` übergeben werden.
# Er verarbeitet alle Dateien im Input-Ordner mit dem Standard-Gender-Filter "alle".
CMD ["python", "main_cli.py", "/app/input", "/app/output"]
