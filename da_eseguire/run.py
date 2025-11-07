#!/usr/bin/env python3
# =============================================================================
# run.py - Wrapper per l'esecuzione del solver MHS
# =============================================================================
"""
Script per eseguire il solver MHS su una singola matrice.

Viene scelto automaticamente tra solver seriale e parallelo in base alle
dimensioni della matrice, oppure è possibile forzare una versione specifica.

Utilizzo:
    python run.py [file_matrice] [opzioni]

Argomenti:
    file_matrice    File .matrix da analizzare (default: esempio.matrix)
    
Opzioni:
    --timeout=N         Timeout in secondi
    --serial            Forza l'utilizzo del solver seriale
    --parallel          Forza l'utilizzo del solver parallelo
    --outdir=DIR        Cartella di output per file .mhs
    --memory-monitoring Monitoraggio memoria con protezione da esaurimento (solo parallelo)
    --memory-threshold=N Soglia percentuale memoria (default: 95, range: 50-99)
    --processes=N       Numero di processi paralleli (default: CPU count - 1, solo parallelo)
    --batch-size=N      Dimensione batch per parallelizzazione (default: 1000, solo parallelo)
    --no-reduction      Non rimuove le colonne vuote dalla matrice (default: rimuove)

Selezione automatica:
    - Matrici piccole: solver seriale (più veloce, meno overhead)
    - Matrici grandi: solver parallelo (sfrutta multicore)
"""

import os
import sys
import time
import argparse
from utility import parse_matrix_file, is_small_matrix
import utility  # Importa utility per accedere a stop_requested

def parse_arguments():
    """
    Parse degli argomenti da riga di comando.
    
    Returns:
        Namespace con gli argomenti parsati
    """
    parser = argparse.ArgumentParser(description='Esegui algoritmo MHS su una matrice specificata')
    parser.add_argument('file', nargs='?', default='esempio.matrix', 
                        help='Il file .matrix da analizzare (default: esempio.matrix)')
    parser.add_argument('--timeout', type=int, default=None,
                        help='Timeout in secondi (default: nessun timeout)')
    parser.add_argument('--serial', action='store_true',
                        help='Forza l\'utilizzo dell\'algoritmo seriale')
    parser.add_argument('--parallel', action='store_true',
                        help='Forza l\'utilizzo dell\'algoritmo parallelo')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Directory di output per il file .mhs generato')
    parser.add_argument('--memory-monitoring', action='store_true',
                        help='Abilita monitoraggio continuo della memoria RAM (solo parallelo)')
    parser.add_argument('--memory-threshold', type=int, default=95,
                        help='Soglia percentuale di memoria per il monitoraggio (default: 95%%)')
    parser.add_argument('--processes', type=int, default=None,
                        help='Numero di processi paralleli (default: CPU count - 1, solo parallelo)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Dimensione batch per parallelizzazione (default: 1000, solo parallelo)')
    parser.add_argument('--no-reduction', action='store_true',
                        help='Non rimuove le colonne vuote dalla matrice')
    parser.add_argument('--skip-global-dedup', action='store_true',
                        help='Salta la deduplicazione globale (solo parallelo, teoricamente non necessaria con succL)')
    # Metadata della matrice (passati da setup.py per essere scritti nel file .mhs)
    parser.add_argument('--origine', type=str, default=None,
                        help='Origine del file (benchmarks1/benchmarks2)')
    parser.add_argument('--densita', type=float, default=None,
                        help='Densità della matrice')
    parser.add_argument('--categoria', type=str, default=None,
                        help='Categoria della matrice (trivial/tiny/small/medium/large/xlarge)')
    
    return parser.parse_args()

def run_solver(file_path, timeout=None, force_serial=False, force_parallel=False, memory_monitoring=False, memory_threshold=95, out_dir=None, num_processes=None, batch_size=None, no_reduction=False, skip_global_dedup=False, origine=None, densita=None, categoria=None):
    """
    Esegue il solver appropriato sul file specificato.
    
    La funzione:
    1. Verifica l'esistenza del file di input
    2. Seleziona automaticamente il solver (seriale o parallelo) in base alle dimensioni della matrice
    3. Consente di forzare manualmente la scelta del solver tramite flag
    4. Costruisce la riga di comando con i parametri appropriati
    5. Esegue il solver e restituisce il tempo di esecuzione
    
    Args:
        file_path: percorso del file .matrix da elaborare
        timeout: timeout in secondi (None = nessun timeout)
        force_serial: se True, forza l'utilizzo dell'algoritmo seriale
        force_parallel: se True, forza l'utilizzo dell'algoritmo parallelo
        memory_monitoring: se True, abilita monitoraggio continuo RAM e protezione da esaurimento nel solver parallelo
        memory_threshold: soglia percentuale di memoria (default: 95)
        out_dir: directory di output personalizzata per il file .mhs generato
        num_processes: numero di processi paralleli (None = auto, solo parallelo)
        batch_size: dimensione batch per parallelizzazione (None = auto, solo parallelo)
        no_reduction: se True, non rimuove le colonne vuote dalla matrice
        skip_global_dedup: se True, salta la deduplicazione globale (solo parallelo)
    
    Returns:
        float: tempo di esecuzione in secondi
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} non trovato.")
        sys.exit(1)
    
    # Determina quale versione utilizzare
    if force_serial:
        print(f"Utilizzo versione seriale (forzata) per {file_path}...")
        from mhs_solver import main as solver_main
        is_parallel = False
    elif force_parallel:
        print(f"Utilizzo versione parallela (forzata) per {file_path}...")
        from mhs_solver_parallel import main as solver_main
        is_parallel = True
    else:
        rows, _ = parse_matrix_file(file_path)
        is_small = is_small_matrix(rows)
        
        if is_small:
            print(f"Data la matrice, utilizzo la versione seriale.")
            from mhs_solver import main as solver_main
            is_parallel = False
        else:
            print(f"Data la matrice, utilizzo la versione parallela.")
            from mhs_solver_parallel import main as solver_main
            is_parallel = True
    
    # Costruisce la lista di argomenti per il solver
    solver_args = [sys.argv[0], file_path]
    
    if timeout is not None:
        solver_args.append(f"--timeout={timeout}")
        print(f"Timeout impostato: {timeout} secondi")
    
    # Monitoraggio memoria solo per solver parallelo
    if memory_monitoring and is_parallel:
        solver_args.append("--memory-monitoring")
        solver_args.append(f"--memory-threshold={memory_threshold}")
        print(f"Abilitato monitoraggio memoria RAM: soglia {memory_threshold}% (protezione da esaurimento)")
    
    # Parametri di parallelizzazione solo per solver parallelo
    if is_parallel:
        if num_processes is not None:
            solver_args.append(f"--processes={num_processes}")
            print(f"Numero di processi paralleli: {num_processes}")
        
        if batch_size is not None:
            solver_args.append(f"--batch-size={batch_size}")
            print(f"Dimensione batch: {batch_size}")
        
        if skip_global_dedup:
            solver_args.append("--skip-global-dedup")
            print("Deduplicazione globale disabilitata (solo ordinamento canonico)")
    
    if no_reduction:
        solver_args.append("--no-reduction")
        print("Riduzione colonne vuote disabilitata (matrice originale)")

    if out_dir:
        solver_args.append(f"--outdir={out_dir}")
        print(f"Directory di output: {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
    
    # Metadata della matrice (passati al solver per essere scritti nel file .mhs)
    if origine is not None:
        solver_args.append(f"--origine={origine}")
    if densita is not None:
        solver_args.append(f"--densita={densita}")
    if categoria is not None:
        solver_args.append(f"--categoria={categoria}")
    
    # Esegue il solver e misura il tempo di esecuzione
    start_time = time.time()
    solver_main(solver_args)
    elapsed_time = time.time() - start_time
    
    print(f"Tempo totale di esecuzione: {elapsed_time:.1f} secondi")
    return elapsed_time

# ===== FUNZIONE MAIN =====

if __name__ == "__main__":
    args = parse_arguments()
    
    # Verifica che non siano specificati entrambi i flag --serial e --parallel
    if args.serial and args.parallel:
        print("Errore: non è possibile specificare sia --serial che --parallel")
        sys.exit(1)
    
    # Se --memory-monitoring viene specificato insieme a --serial, 
    # lo ignoriamo silenziosamente (il solver seriale non supporta il monitoraggio)
    if args.memory_monitoring and args.serial:
        print("Nota: l'opzione --memory-monitoring viene ignorata con --serial (non supportato)")
        args.memory_monitoring = False
    
    # Valida opzioni parallele con --serial
    if args.serial and (args.processes is not None or args.batch_size is not None):
        print("Errore: le opzioni --processes e --batch-size sono supportate solo con --parallel")
        sys.exit(1)
    
    # Valida la soglia memoria
    if args.memory_threshold < 50 or args.memory_threshold > 99:
        print(f"Errore: soglia memoria {args.memory_threshold}% fuori range consentito (50-99%)")
        sys.exit(1)
    
    # Valida num_processes
    if args.processes is not None and args.processes < 1:
        print(f"Errore: numero di processi deve essere >= 1")
        sys.exit(1)
    
    # Valida batch_size
    if args.batch_size is not None and args.batch_size < 1:
        print(f"Errore: dimensione batch deve essere >= 1")
        sys.exit(1)
    
    # Esegui il solver e gestisci eventuali errori
    try:
        run_solver(args.file, args.timeout, args.serial, args.parallel, args.memory_monitoring, args.memory_threshold, args.outdir, args.processes, args.batch_size, args.no_reduction, args.skip_global_dedup, args.origine, args.densita, args.categoria)
    except KeyboardInterrupt:
        sys.exit(1)
    except TimeoutError:
        sys.exit(1)
    except Exception:
        print(f"\nERRORE CRITICO.")
        sys.exit(1)