import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import secrets
from serial import Serial
from serial.tools.list_ports import comports

def simulate_signal(duree, fs, niveau_bruit, frequence_onde_carree1, frequence_onde_carree2, frequence_sinusoidale1, frequence_sinusoidale2):
    t = np.linspace(0, duree, int(fs * duree), endpoint=False)
    
    # Création des ondes carrées et sinusoïdales
    onde_carree1 = signal.square(2 * np.pi * frequence_onde_carree1 * t)
    onde_carree2 = signal.square(2 * np.pi * frequence_onde_carree2 * t)
    onde_sinusoidale1 = np.sin(2 * np.pi * frequence_sinusoidale1 * t)
    onde_sinusoidale2 = np.sin(2 * np.pi * frequence_sinusoidale2 * t)
    
    # Générer un tableau ("array") de bits aléatoires
    octets_aleatoires = secrets.token_bytes(len(t))

    # Convertir les bits en tableau NumPy de valeurs flottantes entre 0 et 1
    tableau_aleatoire = np.frombuffer(octets_aleatoires, dtype=np.uint8) / 255.0

    # Ajouter du bruit au signal
    signal_avec_bruit = (np.convolve(onde_carree1, onde_sinusoidale1, mode='same') +
                        np.convolve(onde_carree2, onde_sinusoidale2, mode='same')) + (2 * niveau_bruit * tableau_aleatoire)
    
    return t, signal_avec_bruit, onde_carree1, onde_carree2, onde_sinusoidale1, onde_sinusoidale2

def analyser_signal(t, signal_avec_bruit, fs, frequence_onde_carree1, frequence_onde_carree2):
    # Calculer la FFT
    resultat_fft = fft.fft(signal_avec_bruit)
    freqs = fft.fftfreq(len(resultat_fft), 1/fs)
    
    # Trouver le pic représentant le rythme cardiaque
    indice_rythme_cardiaque = np.argmax(np.abs(resultat_fft))
    rythme_cardiaque = np.abs(freqs[indice_rythme_cardiaque]) * 60
    
    # Déconvolution pour obtenir l'amplitude
    _, deconv1 = signal.deconvolve(signal_avec_bruit, signal.square(2 * np.pi * frequence_onde_carree1 * t))
    _, deconv2 = signal.deconvolve(signal_avec_bruit, signal.square(2 * np.pi * frequence_onde_carree2 * t))
    
    amplitude1 = np.max(deconv1)
    amplitude2 = np.max(deconv2)
    
    return rythme_cardiaque, amplitude1, amplitude2, deconv1, deconv2, freqs, resultat_fft

def calculer_saturation_oxygene(amplitude1, amplitude2):
    # Calcul du double ratio (R) du principe de Beer-Lambert
    R =  (amplitude1) / (amplitude2)
    
    # Estimation de la saturation de l'oxygène en utilisant la formule empirique
    SO2 = 34.74 * R**2 + 44.73 * R + 12.25
    
    return SO2

def tracer_signaux(t, onde_carree1, onde_sinusoidale1, onde_carree2, onde_sinusoidale2, signal_avec_bruit):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 2, 1)
    plt.plot(t, onde_carree1, label='Onde Carrée 1')
    plt.title('Onde Carrée 1')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(t, onde_sinusoidale1, label='Onde Sinusoïdale 1')
    plt.title('Onde Sinusoïdale 1')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(t, onde_carree2, label='Onde Carrée 2')
    plt.title('Onde Carrée 2')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(t, onde_sinusoidale2, label='Onde Sinusoïdale 2')
    plt.title('Onde Sinusoïdale 2')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, signal_avec_bruit, label='Signal Simulé avec Bruit')
    plt.title('Signal Simulé avec Bruit')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

def tracer_ffts(freqs, resultat_fft, onde_carree1, onde_sinusoidale1):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(freqs, np.abs(resultat_fft), label='FFT du Signal avec Bruit')
    plt.title('FFT du Signal avec Bruit')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(freqs, np.abs(fft.fft(onde_carree1)), label='FFT de l\'Onde Carrée 1')
    plt.plot(freqs, np.abs(fft.fft(onde_sinusoidale1)), label='FFT de l\'Onde Sinusoïdale 1')
    plt.title('FFT de l\'Onde Carrée et Sinusoïdale')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.tight_layout()
    plt.show()

duree = 10  # secondes
fs = 1000  # fréquence d'échantillonnage
niveau_bruit = np.random.randint(12, 22)  # paramètre du bruit
# signal_avec_bruit = donnees
t = np.linspace(0, duree, int(fs * duree), endpoint=False)
# Fréquences
frequence_onde_carree1 = 3
frequence_onde_carree2 = 5

frequence_sinusoidale1 = 1 / np.random.randint(1, 10) + 1
frequence_sinusoidale2 = frequence_sinusoidale1

t, signal_avec_bruit, onde_carree1, onde_carree2, onde_sinusoidale1, onde_sinusoidale2 = simulate_signal(
    duree, fs, niveau_bruit, frequence_onde_carree1, frequence_onde_carree2, frequence_sinusoidale1, frequence_sinusoidale2)

rythme_cardiaque, amplitude1, amplitude2, deconv1, deconv2, freqs, resultat_fft = analyser_signal(
    t, signal_avec_bruit, fs, frequence_onde_carree1, frequence_onde_carree2
)

print(f"Rythme Cardiaque : {rythme_cardiaque :.2f} bpm")
print(f"Amplitude de la Diode 1: {amplitude1:.2f}")
print(f"Amplitude de la Diode 2: {amplitude2:.2f}")
SO2 = calculer_saturation_oxygene(amplitude2, amplitude1)
print(f"Saturation en Oxygène: {SO2:.2f}%")

tracer_signaux(t, onde_carree1, onde_sinusoidale1, onde_carree2, onde_sinusoidale2, signal_avec_bruit)


tracer_ffts(freqs, resultat_fft, onde_carree1, onde_sinusoidale1)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t, deconv1, label='Signal Déconvolué 1')
plt.title('Signal Déconvolué 1')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, deconv2, label='Signal Déconvolué 2')
plt.title('Signal Déconvolué 2')
plt.xlabel('Temps (s)')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(t, rythme_cardiaque*np.ones_like(t), label='Rythme Cardiaque')
plt.title('Rythme Cardiaque au Fil du Temps')
plt.xlabel('Temps (s)')
plt.ylabel('Rythme Cardiaque (Hz)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, SO2*np.ones_like(t), label='Saturation en Oxygène')
plt.title('Saturation en Oxygène au Fil du Temps')
plt.xlabel('Temps (s)')
plt.ylabel('Saturation en Oxygène (%)')
plt.legend()

plt.tight_layout()
plt.show()
