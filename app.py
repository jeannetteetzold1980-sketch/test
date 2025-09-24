from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import threading
import queue
import uuid
import whisper
from core import VoiceGenderClassifier, process_files

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory task storage
tasks = {}

# Load models on startup
model = whisper.load_model("medium")
classifier = VoiceGenderClassifier()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist("files")
    gender_choice = request.form.get("gender")

    if not files:
        return jsonify({"error": "No files selected"}), 400

    filepaths = []
    for file in files:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        filepaths.append(filepath)

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"state": "pending", "log": ""}

    # Run processing in a background thread
    thread = threading.Thread(
        target=process_files_wrapper,
        args=(task_id, filepaths, model, classifier, gender_choice)
    )
    thread.start()

    return jsonify({"task_id": task_id})

@app.route('/status/<task_id>')
def get_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory('results', filename, as_attachment=True)

def process_files_wrapper(task_id, filepaths, model, classifier, gender_choice):
    log_queue = queue.Queue()
    stop_event = threading.Event()
    pause_event = threading.Event()

    def progress_callback(current, total, filename):
        tasks[task_id]["state"] = "processing"
        tasks[task_id]["current"] = current
        tasks[task_id]["total"] = total
        tasks[task_id]["filename"] = filename

    def log_callback(message):
        tasks[task_id]["log"] += message + "\n"

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
        tasks[task_id]["state"] = "error"
        tasks[task_id]["log"] += result["error"]
    elif result:
        tasks[task_id]["state"] = "complete"
        tasks[task_id]["zip_path"] = os.path.basename(result["zip_path"])
    else:
        tasks[task_id]["state"] = "stopped"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
