
import pytest
from core import normalize_text


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
