#!/usr/bin/env python3
# =============================================================================
# reprocess_missing.py - Riprocessa le matrici mancanti
#
# Disponibile come file di utilità da richiamare in caso di necessità
# =============================================================================
"""
Script per identificare e riprocessare le matrici che non hanno
un file .mhs corrispondente nella cartella risultati.

Utilizzo:
    python reprocess_missing.py [--timeout=N] [--parallel] [--serial]
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def find_missing_matrices(selected_dir="selezionate", results_dir="risultati_auto"):
    """
    Trova le matrici che non hanno un file .mhs corrispondente.
    
    Returns:
        dict: dizionario con chiave = subdirectory, valore = lista file mancanti
    """
    missing = {}
    
    # Verifica benchmarks1 e benchmarks2
    for bench in ["benchmarks1", "benchmarks2"]:
        src_dir = os.path.join(selected_dir, bench)
        out_dir = os.path.join(results_dir, bench)
        
        if not os.path.exists(src_dir):
            continue
            
        matrix_files = [f for f in os.listdir(src_dir) if f.endswith(".matrix")]
        
        missing_in_bench = []
        for matrix_file in matrix_files:
            base_name = os.path.splitext(matrix_file)[0]
            mhs_file = os.path.join(out_dir, f"{base_name}.mhs")
            
            if not os.path.exists(mhs_file):
                missing_in_bench.append(matrix_file)
        
        if missing_in_bench:
            missing[bench] = missing_in_bench
    
    return missing

def reprocess_missing_matrices(selected_dir="selezionate", results_dir="risultati_auto", 
                                timeout=None, solver_mode="auto", memory_monitoring=False, 
                                memory_threshold=95, num_processes=None, 
                                batch_size=None, no_reduction=False, skip_global_dedup=False, verbose=True):
    """
    Riprocessa le matrici mancanti con i parametri specificati.
    
    Args:
        selected_dir: directory delle matrici selezionate
        results_dir: directory dei risultati
        timeout: timeout in secondi per ogni matrice
        solver_mode: "auto", "serial" o "parallel"
        memory_monitoring: abilita monitoraggio RAM
        memory_threshold: soglia percentuale memoria
        num_processes: numero processi paralleli
        batch_size: dimensione batch
        no_reduction: se True, non rimuove colonne vuote
        skip_global_dedup: se True, disabilita deduplicazione globale
        verbose: se False, mostra solo riepilogo finale
        
    Returns:
        tuple: (successi, fallimenti)
    """
    # Trova le matrici mancanti
    missing = find_missing_matrices(selected_dir, results_dir)
    
    if not missing:
        if verbose:
            print("Tutte le matrici sono state processate correttamente!")
            print("  Non ci sono file .mhs mancanti.")
        return (0, 0)
    
    # Conta totale matrici mancanti
    total_missing = sum(len(files) for files in missing.values())
    
    if verbose:
        print(f"Trovate {total_missing} matrici senza file .mhs:")
        print()
        
        for bench, files in missing.items():
            print(f"  {bench}/:")
            for f in files:
                print(f"    - {f}")
            print()
    
    # Costruisci argomenti extra
    extra_args = []
    
    if timeout is not None:
        extra_args.append(f"--timeout={timeout}")
    
    if solver_mode == "serial":
        extra_args.append("--serial")
    elif solver_mode == "parallel":
        extra_args.append("--parallel")
    
    if memory_monitoring and solver_mode != "serial":
        extra_args.append("--memory-monitoring")
        extra_args.append(f"--memory-threshold={memory_threshold}")
    
    if num_processes is not None and solver_mode != "serial":
        extra_args.append(f"--processes={num_processes}")
    
    if batch_size is not None and solver_mode != "serial":
        extra_args.append(f"--batch-size={batch_size}")
    
    # Carica selection.json per ottenere i metadati
    metadata_map = {}
    selection_json = "selection.json"
    if os.path.exists(selection_json):
        try:
            with open(selection_json, 'r') as f:
                selection_data = json.load(f)
                # Crea mappa: filename -> {origine, densita, categoria}
                for entry in selection_data:
                    fname = entry.get("file", "")
                    base_name = os.path.splitext(fname)[0]  # Rimuovi estensione
                    metadata_map[base_name] = {
                        "origine": entry.get("origine", ""),
                        "densita": entry.get("densita", 0.0),
                        "categoria": entry.get("categoria", "")
                    }
        except Exception:
            if verbose:
                print(f"ATTENZIONE: Errore nel caricamento di {selection_json}.")
                print("  Proseguo senza metadati.")
    
    # Riprocessa ogni matrice
    if verbose:
        print()
        print("="*70)
        print("RIPROCESSAMENTO IN CORSO")
        print("="*70)
        print()
    
    success = 0
    failed = 0
    interrupted = False
    
    try:
        for bench, files in missing.items():
            src_dir = os.path.join(selected_dir, bench)
            out_dir = os.path.join(results_dir, bench)
            
            os.makedirs(out_dir, exist_ok=True)
            
            for matrix_file in files:
                matrix_path = os.path.join(src_dir, matrix_file)
                
                if verbose:
                    print(f"\nProcessando: {matrix_path}")
                    print("-"*70)
                
                # Costruisci comando con metadati
                cmd = [sys.executable, "run.py", matrix_path, f"--outdir={out_dir}"]
                cmd.extend(extra_args)
                
                # Aggiungi opzioni di preprocessing
                if no_reduction:
                    cmd.append("--no-reduction")
                if skip_global_dedup:
                    cmd.append("--skip-global-dedup")
                
                # Aggiungi metadati se disponibili
                base_name = os.path.splitext(matrix_file)[0]
                if base_name in metadata_map:
                    meta = metadata_map[base_name]
                    cmd.append(f"--origine={meta['origine']}")
                    cmd.append(f"--densita={meta['densita']}")
                    cmd.append(f"--categoria={meta['categoria']}")
                
                if verbose:
                    print(f"Comando: {' '.join(cmd)}")
                    print()
                
                try:
                    # Esegui il comando e gestisci l'interruzione
                    try:
                        # Cattura sempre stderr per vedere gli errori, anche in verbose mode
                        result = subprocess.run(cmd, capture_output=False, stderr=subprocess.PIPE, text=True)
                        
                        # Controlla se il comando è andato a buon fine
                        if result.returncode != 0:
                            if verbose:
                                print(f"ERRORE: Il comando è fallito con codice di uscita {result.returncode}")
                                if result.stderr:
                                    print(f"Errore del comando:\n{result.stderr}")
                            failed += 1
                            continue  # Salta alla prossima matrice
                            
                    except KeyboardInterrupt: # Ctrl+C premuto durante subprocess
                        raise  # Propaga l'interruzione
                    
                    # Verifica che il file .mhs sia stato creato nella posizione corretta
                    base_name = os.path.splitext(matrix_file)[0]
                    mhs_file = os.path.join(out_dir, f"{base_name}.mhs")
                    
                    # Controlla anche la posizione errata (cartella sorgente)
                    wrong_location = os.path.join(src_dir, f"{base_name}.mhs")
                    
                    if os.path.exists(mhs_file):
                        if verbose:
                            print(f"File .mhs creato: {mhs_file}")
                        success += 1
                    elif os.path.exists(wrong_location):
                        # File creato nella posizione sbagliata, spostiamolo
                        if verbose:
                            print(f"File .mhs trovato in posizione errata: {wrong_location}")
                            print(f"  Lo sposto nella directory corretta: {mhs_file}")
                        try:
                            import shutil
                            shutil.move(wrong_location, mhs_file)
                            if verbose:
                                print(f"File spostato correttamente")
                            success += 1
                        except Exception:
                            if verbose:
                                print(f"ERRORE nello spostamento.")
                            failed += 1
                    else:
                        if verbose:
                            print(f"File .mhs NON creato né in {mhs_file} né in {wrong_location}")
                        failed += 1
                        
                except KeyboardInterrupt:
                    interrupted = True
                    break  # Esce dal ciclo dei file
                except Exception:
                    if verbose:
                        print(f"ERRORE durante il processamento di {matrix_path}.")
                    failed += 1
            
            if interrupted:
                break  # Esce dal ciclo dei benchmark
    
    except KeyboardInterrupt:
        interrupted = True    # Riepilogo
    if verbose:
        print()
        print("="*70)
        print("RIEPILOGO RIPROCESSAMENTO")
        print("="*70)
        print(f"Successi:   {success}")
        print(f"Fallimenti: {failed}")
        print(f"Totale:     {success + failed}")
        print("="*70)
        print()
    
    return (success, failed)

def main():
    """Funzione principale per esecuzione da riga di comando."""
    print("="*70)
    print("  RIPROCESSAMENTO MATRICI MANCANTI")
    print("="*70)
    print()
    
    # Estrai parametri dalla riga di comando
    timeout = None
    solver_mode = "auto"
    memory_monitoring = False
    memory_threshold = 95
    num_processes = None
    batch_size = None
    no_reduction = False
    skip_global_dedup = False
    selected_dir = "selezionate"
    results_dir = "risultati_auto"
    
    i = 0
    while i < len(sys.argv) - 1:  # -1 perché sys.argv[0] è il nome del script
        arg = sys.argv[i + 1]
        i += 1
        
        if arg.startswith("--timeout="):
            timeout = int(arg.split("=", 1)[1])
        elif arg == "--timeout" and i < len(sys.argv) - 1:
            timeout = int(sys.argv[i + 1])
            i += 1
        elif arg == "--parallel":
            solver_mode = "parallel"
        elif arg == "--serial":
            solver_mode = "serial"
        elif arg == "--memory-monitoring":
            memory_monitoring = True
        elif arg.startswith("--memory-threshold="):
            memory_threshold = int(arg.split("=", 1)[1])
        elif arg == "--memory-threshold" and i < len(sys.argv) - 1:
            memory_threshold = int(sys.argv[i + 1])
            i += 1
        elif arg.startswith("--processes="):
            num_processes = int(arg.split("=", 1)[1])
        elif arg == "--processes" and i < len(sys.argv) - 1:
            num_processes = int(sys.argv[i + 1])
            i += 1
        elif arg.startswith("--batch-size="):
            batch_size = int(arg.split("=", 1)[1])
        elif arg == "--batch-size" and i < len(sys.argv) - 1:
            batch_size = int(sys.argv[i + 1])
            i += 1
        elif arg == "--no-reduction":
            no_reduction = True
        elif arg == "--skip-global-dedup":
            skip_global_dedup = True
        elif arg.startswith("--selected-dir="):
            selected_dir = arg.split("=", 1)[1]
        elif arg == "--selected-dir" and i < len(sys.argv) - 1:
            selected_dir = sys.argv[i + 1]
            i += 1
        elif arg.startswith("--results-dir="):
            results_dir = arg.split("=", 1)[1]
        elif arg == "--results-dir" and i < len(sys.argv) - 1:
            results_dir = sys.argv[i + 1]
            i += 1
    
    # Chiama la funzione principale con gestione interruzioni
    try:
        reprocess_missing_matrices(
            selected_dir=selected_dir,
            results_dir=results_dir,
            timeout=timeout,
            solver_mode=solver_mode,
            memory_monitoring=memory_monitoring,
            memory_threshold=memory_threshold,
            num_processes=num_processes,
            batch_size=batch_size,
            no_reduction=no_reduction,
            skip_global_dedup=skip_global_dedup,
            verbose=True
        )
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
