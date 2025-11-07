#!/usr/bin/env python3
# =============================================================================
# cleanup_misplaced.py - Sposta file .mhs mal posizionati
#
# Disponibile come file di utilità da richiamare in caso di necessità
# =============================================================================
"""
Script per trovare e spostare file .mhs che si trovano nella cartella
'selezionate' invece che nella cartella 'risultati_auto'.

Utilizzo:
    python cleanup_misplaced.py [opzioni]

Opzioni:
    --selected-dir DIR     Directory delle matrici selezionate (default: selezionate)
    --results-dir DIR      Directory dei risultati (default: risultati_auto)
"""

import os
import shutil
import argparse

def cleanup_misplaced_mhs(selected_dir="selezionate", results_dir="risultati_auto", verbose=True):
    """
    Trova e sposta file .mhs mal posizionati.
    
    Args:
        selected_dir: directory delle matrici selezionate
        results_dir: directory dei risultati
+        
    Returns:
        tuple: (file_spostati, errori)
    """
    if verbose:
        print("="*70)
        print("  PULIZIA FILE .MHS MAL POSIZIONATI")
        print("="*70)
        print()
    
    total_moved = 0
    total_errors = 0
    
    # Verifica benchmarks1 e benchmarks2
    for bench in ["benchmarks1", "benchmarks2"]:
        src_dir = os.path.join(selected_dir, bench)
        out_dir = os.path.join(results_dir, bench)
        
        if not os.path.exists(src_dir):
            continue
        
        # Crea la directory di destinazione se non esiste
        os.makedirs(out_dir, exist_ok=True)
        
        # Cerca file .mhs nella cartella sorgente
        mhs_files = [f for f in os.listdir(src_dir) if f.endswith(".mhs")]
        
        if not mhs_files:
            if verbose:
                print(f"{bench}/: nessun file .mhs mal posizionato")
            continue
        
        if verbose:
            print(f"Attenzione: {bench}/: trovati {len(mhs_files)} file .mhs mal posizionati")
        
        for mhs_file in mhs_files:
            src_path = os.path.join(src_dir, mhs_file)
            dest_path = os.path.join(out_dir, mhs_file)
            
            if verbose:
                print(f"  Sposto: {mhs_file}")
                print(f"    Da:  {src_path}")
                print(f"    A:   {dest_path}")
            
            try:
                # Sovrascrive automaticamente eventuali file esistenti
                shutil.move(src_path, dest_path)
                if verbose:
                    print(f"    Spostato correttamente")
                total_moved += 1
                
            except Exception:
                if verbose:
                    print(f"    ERRORE: impossibile spostare {mhs_file}")
                total_errors += 1
        
        if verbose:
            print()
    
    # Riepilogo
    if verbose:
        print("="*70)
        print("RIEPILOGO PULIZIA")
        print("="*70)
        print(f"File spostati:  {total_moved}")
        print(f"Errori:         {total_errors}")
        print("="*70)
        print()
    
    return (total_moved, total_errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sposta file .mhs mal posizionati")
    parser.add_argument("--selected-dir", default="selezionate", 
                       help="Directory delle matrici selezionate (default: selezionate)")
    parser.add_argument("--results-dir", default="risultati_auto",
                       help="Directory dei risultati (default: risultati_auto)")
    args = parser.parse_args()
    
    cleanup_misplaced_mhs(
        selected_dir=args.selected_dir,
        results_dir=args.results_dir,
        verbose=True
    )
