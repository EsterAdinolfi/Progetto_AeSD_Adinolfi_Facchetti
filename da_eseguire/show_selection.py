#!/usr/bin/env python3
# =============================================================================
# show_selection.py - Visualizzazione del catalogo delle matrici selezionate
# =============================================================================
"""
Script semplice per stampare a video le matrici selezionate per i test.

Viene letto il file 'selection.json' prodotto da matrices_selection.py e vengono mostrati:
    - Nome del file
    - Numero di righe (N)
    - Numero di colonne originali (M)
    - Numero di colonne ridotte (M_ridotto)
    - Densità della matrice
    - Categoria di complessità

Utilizzo:
    python show_selection.py
"""

import json

# Carica il file di selezione
with open("selection.json", "r", encoding="utf-8") as f:
    sel = json.load(f)

print(f'TOTALE: {len(sel)} matrici\n')

for cat in ['trivial', 'tiny', 'small', 'medium', 'large', 'xlarge']:
    matrices = [m for m in sel if m['categoria'] == cat]
    if matrices:
        print(f'\n{cat.upper()}:')
        for m in matrices:
            print(f'  {m["origine"]:25} | N={m["N"]:2} M={m["M"]:3} M_r={m["M_ridotto"]:3} dens={m["densita"]:.3f}')
