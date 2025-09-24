import pytest
from unittest.mock import patch, MagicMock, call
import numpy as np
import threading
from core import normalize_text, VoiceGenderClassifier, transcribe_and_normalize, process_files


@pytest.fixture
def process_files_mocks():
    with (patch('core.librosa.load') as mock_load,
          patch('core.AudioSegment') as mock_audio_segment,
          patch('core.split_on_silence') as mock_split_on_silence,
          patch('core.transcribe_and_normalize') as mock_transcribe,
          patch('core.shutil.make_archive') as mock_make_archive,
          patch('core.os.path.exists') as mock_exists,
          patch('core.os.makedirs'),
          patch('core.shutil.rmtree') as mock_rmtree):

        # Setup mock behaviors
        mock_load.return_value = (np.random.randn(16000), 16000)
        segment_mock = MagicMock()
        segment_mock.dBFS = -10.0
        segment_mock.__len__.return_value = 5000
        mock_split_on_silence.return_value = [segment_mock]
        mock_audio_segment.return_value = segment_mock
        mock_transcribe.return_value = "test transcript"
        mock_make_archive.return_value = "/path/to/zipfile.zip"
        mock_exists.return_value = True

        yield {
            "mock_load": mock_load,
            "mock_audio_segment": mock_audio_segment,
            "mock_split_on_silence": mock_split_on_silence,
            "mock_transcribe": mock_transcribe,
            "mock_make_archive": mock_make_archive,
            "mock_exists": mock_exists,
            "mock_rmtree": mock_rmtree
        }

def test_normalize_text_with_numbers():
    assert normalize_text("123") == "einhundertdreiundzwanzig"

def test_normalize_text_with_uppercase():
    assert normalize_text("TEST") == "test"

def test_normalize_text_with_punctuation():
    assert normalize_text("test, test.") == "test test"

def test_normalize_text_with_extra_spaces():
    assert normalize_text("  test  test  ") == "test test"

def test_normalize_text_with_umlaute():
    assert normalize_text("äöüß") == "äöüß"

def test_normalize_text_empty():
    assert normalize_text("") == ""

def test_normalize_text_combined():
    assert normalize_text("Test 123, mit ÄÖÜß.") == "test einhundertdreiundzwanzig mit äöüß"


@patch('core.librosa.pyin')
@patch('core.librosa.feature.mfcc')
def test_voice_gender_classifier_predict_male(mock_mfcc, mock_pyin):
    # Mocking librosa functions
    mock_pyin.return_value = (np.array([100.0]), None, None)
    mock_mfcc.return_value = np.random.rand(13, 10)

    classifier = VoiceGenderClassifier()
    # Create a dummy audio signal
    y = np.random.randn(16000)
    sr = 16000
    gender = classifier.predict(y, sr)
    assert gender == "männlich"

@patch('core.librosa.pyin')
@patch('core.librosa.feature.mfcc')
def test_voice_gender_classifier_predict_female(mock_mfcc, mock_pyin):
    # Mocking librosa functions
    mock_pyin.return_value = (np.array([200.0]), None, None)
    mock_mfcc.return_value = np.random.rand(13, 10)

    classifier = VoiceGenderClassifier()
    # Create a dummy audio signal
    y = np.random.randn(16000)
    sr = 16000
    gender = classifier.predict(y, sr)
    assert gender == "weiblich"

@patch('core.librosa.load')
@patch('core.VoiceGenderClassifier._extract_features')
def test_voice_gender_classifier_calibrate(mock_extract_features, mock_load):
    # Mocking librosa.load and _extract_features
    mock_load.return_value = (np.random.randn(16000), 16000)
    mock_extract_features.return_value = np.random.rand(14)

    classifier = VoiceGenderClassifier()
    labeled_data = [
        ("file1.wav", "männlich"),
        ("file2.wav", "weiblich"),
        ("file3.wav", "männlich"),
        ("file4.wav", "weiblich"),
        ("file5.wav", "männlich"),
    ]
    assert classifier.calibrate(labeled_data)
    assert classifier.custom_model is not None

@patch('core.whisper')
def test_transcribe_and_normalize_success(mock_whisper):
    mock_whisper.transcribe.return_value = {"text": "Das ist ein Test 123"}
    model = MagicMock()
    model.transcribe = mock_whisper.transcribe
    log_callback = MagicMock()

    result = transcribe_and_normalize("dummy.wav", model, log_callback)

    assert result == "das ist ein test einhundertdreiundzwanzig"
    log_callback.assert_called_with("  Segment transkribiert.")

@patch('core.whisper')
def test_transcribe_and_normalize_empty(mock_whisper):
    mock_whisper.transcribe.return_value = {"text": ""}
    model = MagicMock()
    model.transcribe = mock_whisper.transcribe
    log_callback = MagicMock()

    result = transcribe_and_normalize("dummy.wav", model, log_callback)

    assert result is None
    log_callback.assert_called_with("  Segment transkribiert.")

@patch('core.whisper')
def test_transcribe_and_normalize_exception(mock_whisper):
    mock_whisper.transcribe.side_effect = Exception("Test Exception")
    model = MagicMock()
    model.transcribe = mock_whisper.transcribe
    log_callback = MagicMock()

    result = transcribe_and_normalize("dummy.wav", model, log_callback)

    assert result is None
    log_callback.assert_called_with("  WARNUNG: Transkriptionsfehler: Test Exception")

def test_process_files_success(process_files_mocks):
    filepaths = ["file1.wav"]
    model = MagicMock()
    classifier = MagicMock()
    classifier.predict.return_value = "männlich"
    gender_choice = "alle"
    stop_event = threading.Event()
    pause_event = threading.Event()
    progress_callback = MagicMock()
    log_callback = MagicMock()

    result = process_files(
        filepaths, model, classifier, gender_choice, stop_event, pause_event, progress_callback, log_callback
    )

    assert result["total_files"] == 1
    assert result["total_segments_usable"] == 1
    assert result["files_skipped"] == 0
    assert result["zip_path"] == "/path/to/zipfile.zip"
    progress_callback.assert_called_once()
    assert log_callback.call_count > 0

def test_process_files_skip_gender(process_files_mocks):
    filepaths = ["file1.wav"]
    model = MagicMock()
    classifier = MagicMock()
    classifier.predict.return_value = "weiblich"
    gender_choice = "männlich"
    stop_event = threading.Event()
    pause_event = threading.Event()
    progress_callback = MagicMock()
    log_callback = MagicMock()

    result = process_files(
        filepaths, model, classifier, gender_choice, stop_event, pause_event, progress_callback, log_callback
    )

    assert result["total_files"] == 1
    assert result["total_segments_usable"] == 0
    assert result["files_skipped"] == 1
    progress_callback.assert_called_once()
    log_callback.assert_any_call("-> WIRD ÜBERSPRUNGEN (Filter: 'männlich')")

def test_process_files_load_error(process_files_mocks):
    process_files_mocks["mock_load"].side_effect = Exception("Load Error")
    filepaths = ["file1.wav"]
    model = MagicMock()
    classifier = MagicMock()
    gender_choice = "alle"
    stop_event = threading.Event()
    pause_event = threading.Event()
    progress_callback = MagicMock()
    log_callback = MagicMock()

    result = process_files(
        filepaths, model, classifier, gender_choice, stop_event, pause_event, progress_callback, log_callback
    )

    assert result["total_files"] == 1
    assert result["total_segments_usable"] == 0
    assert result["files_skipped"] == 0
    log_callback.assert_any_call("  FEHLER: Konnte 'file1.wav' nicht laden: Load Error")

def test_process_files_stop_event(process_files_mocks):
    filepaths = ["file1.wav", "file2.wav"]
    model = MagicMock()
    classifier = MagicMock()
    gender_choice = "alle"
    stop_event = threading.Event()
    pause_event = threading.Event()
    progress_callback = MagicMock()
    log_callback = MagicMock()

    def stop_after_first_file(*args, **kwargs):
        stop_event.set()

    progress_callback.side_effect = stop_after_first_file

    result = process_files(
        filepaths, model, classifier, gender_choice, stop_event, pause_event, progress_callback, log_callback
    )

    assert result is None
    progress_callback.assert_called_once()