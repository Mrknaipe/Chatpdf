import subprocess
import sys

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
        """Calls Ollama with Windows encoding handling."""
        try:
            # Force UTF-8 encoding on Windows.
            result = subprocess.run(
                ['ollama', 'run', self.model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace'  # Replaces problematic characters.
            )

            # If ollama returns an error, display it.
            if result.returncode != 0:
                err = (result.stderr or "").strip()
                return f"Ollama error (code {result.returncode}): {err[:500]}"

            response = (result.stdout or "").strip()

            # Check whether the response is empty.
            if not response or len(response) < 5:
                return "Error: empty response from Ollama"

            return response

        except subprocess.TimeoutExpired:
            return f"Timeout after {timeout}s - The model took too long"
        except UnicodeDecodeError as e:
            return f"Encoding error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"
