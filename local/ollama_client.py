import subprocess
import sys

class OllamaClient:
    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    def verify_ollama(self):
        """Vérifie qu'Ollama est installé et accessible"""
        try:
            subprocess.run(['ollama', '--version'], capture_output=True, check=True)
            print(f"✅ Ollama détecté - Modèle: {self.model}\n")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Erreur: Ollama n'est pas installé ou pas dans le PATH")
            print("Installez Ollama depuis: https://ollama.ai")
            sys.exit(1)

    def call_ollama(self, prompt: str, timeout: int = 180) -> str:
        """Appelle Ollama avec gestion d'encodage Windows"""
        try:
            # Forcer l'encodage UTF-8 pour Windows
            result = subprocess.run(
                ['ollama', 'run', self.model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'  # Remplace les caractères problématiques
            )

            # Si ollama renvoie une erreur, l'afficher
            if result.returncode != 0:
                err = (result.stderr or "").strip()
                return f"Erreur Ollama (code {result.returncode}): {err[:500]}"

            response = (result.stdout or "").strip()

            # Vérifier si la réponse est vide
            if not response or len(response) < 5:
                return "Erreur: réponse vide d'Ollama"

            return response

        except subprocess.TimeoutExpired:
            return f"Timeout après {timeout}s - Le modèle prend trop de temps"
        except UnicodeDecodeError as e:
            return f"Erreur d'encodage: {str(e)}"
        except Exception as e:
            return f"Erreur: {str(e)}"
