import subprocess
import sys
import re
import requests

class OllamaClient:
    def __init__(self, model: str = "gemma3:12b"):
        self.model = model

    def verify_ollama(self):
        """Checks that Ollama is installed and accessible."""
        try:
            subprocess.run(['ollama', '--version'], capture_output=True, check=True)
            print(f"✅ Ollama detected - Model: {self.model}\n")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Error: Ollama is not installed or not in PATH")
            print("Install Ollama from: https://ollama.ai")
            sys.exit(1)

    def call_ollama(self, prompt: str, timeout: int = 180) -> str:
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )

            if response.status_code != 200:
                return f"Ollama error (code {response.status_code}): {response.text[:500]}"

            answer = response.json().get("response", "").strip()

            if not answer or len(answer) < 5:
                return "Error: empty response from Ollama"

            return answer

        except requests.Timeout:
            return f"Timeout after {timeout}s - The model took too long"
        except Exception as e:
            return f"Error: {str(e)}"
