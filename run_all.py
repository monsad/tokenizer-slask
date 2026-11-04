"""Cały potok jedną komendą: korpus -> trening -> metryki -> pytania A–D.

    uv run --no-project --with-requirements requirements.txt python run_all.py

Wymaga pobranych danych w data/raw (patrz README, sekcja „Dane”).
"""
import subprocess
import sys

KROKI = ["korpus.czysc", "eksperymenty.e1_trening", "eksperymenty.e2_slownik",
         "eksperymenty.e3_pytanie_a", "eksperymenty.e4_pytanie_b",
         "eksperymenty.e5_pytanie_c", "eksperymenty.e6_pytanie_d"]

for krok in KROKI:
    print(f"\n=== {krok} ===", flush=True)
    subprocess.run([sys.executable, "-m", krok], check=True)
