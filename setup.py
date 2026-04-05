"""
setup.py — One-time project setup.
Builds the dataset from real weather data and trains the ML model.
Run once from the project root before launching the app.
"""
import subprocess
import sys

def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)

if __name__ == "__main__":
    print("=" * 60)
    print("  Smart Urban Noise Analyzer — Setup")
    print("=" * 60)

    print("\n[1/2] Building dataset (fetching real weather from Open-Meteo)...")
    run(f"{sys.executable} -c \"from src.data_processing import build_dataset, load_and_process_data; build_dataset(); load_and_process_data()\"")

    print("\n[2/2] Training ML models...")
    run(f"{sys.executable} -c \"from src.model_training import main; main()\"")

    print("\n" + "=" * 60)
    print("  Setup complete. Run the app with:")
    print("  streamlit run app.py")
    print("=" * 60)
