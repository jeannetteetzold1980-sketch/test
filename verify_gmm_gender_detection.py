
import numpy as np
from unittest.mock import patch
from core import VoiceGenderClassifier

def run_gmm_test():
    """
    Führt Tests für den neuen GMM-basierten VoiceGenderClassifier aus,
    um die Zuverlässigkeit der Geschlechtsbestimmung zu überprüfen.
    """
    print("Initialisiere den GMM-basierten VoiceGenderClassifier...")
    # Beim Initialisieren werden die Standard-Modelle trainiert.
    # Die Ausgabe des Trainings wird hier erwartet.
    classifier = VoiceGenderClassifier()
    
    print("\nStarte Tests für die GMM-Geschlechtsbestimmung...")

    # --- Test 1: Weibliche Stimme (hohe Tonhöhe, typische MFCCs) ---
    # Wir simulieren die Merkmale, die _extract_features für eine weibliche Stimme zurückgeben würde.
    # Diese Werte sind an die Trainingsdaten des Modells angelehnt.
    simulated_female_features = np.array([210, -12, 28, 1, 12, 1, 6, 1, 3, 1, 2, 2, 1, 0]).reshape(1, -1)
    
    print("\nTeste Erkennung für weibliche Stimme...")
    # Wir müssen die Feature-Extraktion nicht mocken, da wir die Scores direkt vergleichen.
    # Stattdessen rufen wir die score-Methode der GMMs direkt auf.
    male_score_female_voice = classifier.male_gmm.score(simulated_female_features)
    female_score_female_voice = classifier.female_gmm.score(simulated_female_features)

    print(f"Score für männliches Modell: {male_score_female_voice:.2f}")
    print(f"Score für weibliches Modell: {female_score_female_voice:.2f}")
    
    is_female = female_score_female_voice > male_score_female_voice
    detected_gender_female = "weiblich" if is_female else "männlich"
    print(f"-> Erkanntes Geschlecht: {detected_gender_female}")
    
    assert is_female, f"Fehler: Weibliche Stimme als '{detected_gender_female}' erkannt!"
    print("✅ Test für weibliche Stimme erfolgreich!")

    # --- Test 2: Männliche Stimme (tiefe Tonhöhe, typische MFCCs) ---
    # Wir simulieren die Merkmale für eine männliche Stimme.
    simulated_male_features = np.array([120, -15, 20, -5, 5, -2, 2, -2, 0, -2, -1, -1, -2, -3]).reshape(1, -1)

    print("\nTeste Erkennung für männliche Stimme...")
    male_score_male_voice = classifier.male_gmm.score(simulated_male_features)
    female_score_male_voice = classifier.female_gmm.score(simulated_male_features)

    print(f"Score für männliches Modell: {male_score_male_voice:.2f}")
    print(f"Score für weibliches Modell: {female_score_male_voice:.2f}")

    is_male = male_score_male_voice > female_score_male_voice
    detected_gender_male = "männlich" if is_male else "weiblich"
    print(f"-> Erkanntes Geschlecht: {detected_gender_male}")
    
    assert is_male, f"Fehler: Männliche Stimme als '{detected_gender_male}' erkannt!"
    print("✅ Test für männliche Stimme erfolgreich!")

    print("\nAlle GMM-Tests erfolgreich abgeschlossen!")

if __name__ == "__main__":
    run_gmm_test()
