
import numpy as np
from unittest.mock import patch
from core import VoiceGenderClassifier

def run_test():
    """
    Führt Tests für den VoiceGenderClassifier aus, um die korrekte Erkennung
    von männlichen und weiblichen Stimmen basierend auf der Tonhöhe zu überprüfen.
    """
    classifier = VoiceGenderClassifier()
    sr = 16000
    # Erstelle ein Dummy-Audiosignal, der Inhalt ist für den Test nicht relevant,
    # da die Feature-Extraktion gemockt wird.
    dummy_audio = np.random.randn(sr) 

    print("Starte Tests für die Geschlechtsbestimmung...")

    # --- Test 1: Weibliche Stimme (hohe Tonhöhe) ---
    # Wir simulieren, dass _extract_features eine hohe Tonhöhe (200 Hz) zurückgibt.
    simulated_female_features = np.array([200.0] + [0.0]*13)
    
    with patch.object(classifier, '_extract_features', return_value=simulated_female_features):
        print("\nTeste Erkennung für weibliche Stimme (simulierte Tonhöhe: 200 Hz)...")
        gender_female = classifier.predict(dummy_audio, sr)
        print(f"Erkanntes Geschlecht: {gender_female}")
        
        assert gender_female == "weiblich", f"Fehler: Weibliche Stimme als '{gender_female}' erkannt!"
        print("✅ Test für weibliche Stimme erfolgreich!")

    # --- Test 2: Männliche Stimme (tiefe Tonhöhe) ---
    # Wir simulieren, dass _extract_features eine tiefe Tonhöhe (100 Hz) zurückgibt.
    simulated_male_features = np.array([100.0] + [0.0]*13)

    with patch.object(classifier, '_extract_features', return_value=simulated_male_features):
        print("\nTeste Erkennung für männliche Stimme (simulierte Tonhöhe: 100 Hz)...")
        gender_male = classifier.predict(dummy_audio, sr)
        print(f"Erkanntes Geschlecht: {gender_male}")
        
        assert gender_male == "männlich", f"Fehler: Männliche Stimme als '{gender_male}' erkannt!"
        print("✅ Test für männliche Stimme erfolgreich!")

    print("\nAlle Tests erfolgreich abgeschlossen!")

if __name__ == "__main__":
    run_test()
