#!/usr/bin/env python3
"""
Solver SERIALE per il calcolo dei Minimal Hitting Sets (MHS).

Utilizzo:
    python mhs_solver.py input.matrix [opzioni]

Opzioni:
    --timeout=N      Limite massimo in secondi
    --no-reduction   Non rimuove le colonne vuote
    --outdir=DIR     Cartella di output per il file .mhs
"""

import sys
import os
import time
import tracemalloc
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import threading

from utility import Hypothesis, generate_succ_left, parse_matrix_file, get_column_vectors, write_mhs_output, measure_performance, input_listener, periodic_check_with_progress, format_level_summary, create_state_exception, format_parsing_message, format_level_start_message, format_summary_message, format_timeout_message

last_saved_state = None

def mhs_solver(col_vectors: List[int], col_map: List[int], num_rows: int, max_level: int, start_time: float, timeout: int = None):
    """
    Algoritmo seriale per il calcolo dei Minimal Hitting Sets.
    
    Esplora lo spazio delle soluzioni per livelli (BFS) con cardinalità crescente,
    applicando pruning per minimalità.
    
    Args:
        col_vectors: vettori delle colonne non vuote (bitmask delle righe)
        col_map: mapping da indice ridotto a indice originale
        num_rows: numero di righe nella matrice
        max_level: livello massimo di esplorazione
        start_time: timestamp di inizio
        timeout: timeout in secondi (opzionale)
        
    Returns:
        Tupla (found_mhs, stats_per_level, mhs_per_level, max_level_reached)
    """
    all_rows_mask = (1 << num_rows) - 1  # Maschera per tutte le righe
    
    found_mhs = []  # Delta: sequenza contenente tutte le soluzioni trovate finora (MAIN:5)
    found_mhs_sets = set()  # Per pruning: evita duplicati e sovra-insiemi
    current_level_hypotheses = []  # current: sequenza delle ipotesi correnti (MAIN:4)
    stats_per_level = {}
    mhs_per_level = {}
    max_level_reached = 0
    
    # Inizializzazione: ipotesi vuota (livello 0) - MAIN:2-3
    h0 = Hypothesis(0, len(col_vectors))  # ho ← 0 (MAIN:2)
    h0.vector = 0  # SET_FIELDS(ho) - imposta vector a 0 per ipotesi vuota (SET_FIELDS:5)
    current_level_hypotheses.append(h0)  # current ← <ho> (MAIN:4)
    stats_per_level[0] = 1
    
    for level in range(0, max_level + 1):  # repeat loop (MAIN:6-21)
        level_start_time = time.time()
        print(format_level_start_message(level, len(current_level_hypotheses)))
        
        next_level_hypotheses = []  # next ← <> (MAIN:7)
        current_hypotheses_to_process = current_level_hypotheses.copy()
        last_update_time = level_start_time
        
        for h_idx, h in enumerate(current_hypotheses_to_process):  # for each h in current (MAIN:8)
            try:
                check_result, should_print, last_update_time, message = periodic_check_with_progress(
                    h_idx, len(current_hypotheses_to_process), last_update_time,
                    timeout=timeout, start_time=start_time, stop_event=stop_event,
                    check_frequency=10,
                    prefix=f"  Livello {level}: ",
                    extra_info=f"MHS trovati: {len(found_mhs)}"
                )
                
                if should_print:
                    print(message, end="", flush=True)
                    last_saved_state = (found_mhs, level, stats_per_level, mhs_per_level)
            except TimeoutError:
                last_saved_state = (found_mhs, level, stats_per_level, mhs_per_level)
                raise create_state_exception(TimeoutError, found_mhs, level, stats_per_level, mhs_per_level)
            except KeyboardInterrupt:
                last_saved_state = (found_mhs, level, stats_per_level, mhs_per_level)
                raise create_state_exception(KeyboardInterrupt, found_mhs, level, stats_per_level, mhs_per_level)
                
            is_mhs = False
            # if CHECK(h) then (MAIN:9)
            if h.vector == all_rows_mask:  # CHECK(h): verifica se vector non include zeri (CHECK:2-3)
                mhs_indices = [col_map[col_idx] for col_idx in range(h.num_cols) 
                               if (h.bin >> (h.num_cols - 1 - col_idx)) & 1]
                
                mhs_indices_set = frozenset(mhs_indices)
                
                found_mhs.append((mhs_indices, h.card))  # APPEND(Delta, h) (MAIN:10)
                found_mhs_sets.add(mhs_indices_set)
                
                if h.card not in mhs_per_level:
                    mhs_per_level[h.card] = 0
                mhs_per_level[h.card] += 1
                last_saved_state = (found_mhs, level, stats_per_level, mhs_per_level)
                is_mhs = True  
                # (MAIN:11) Non rimuoviamo h da current: non necessario poiché non viene propagato (= non genera figli) e l'iterazione è su una copia => non serve rimuoverlo
            
            if not is_mhs:  # else branch (MAIN:12-19)
                try:
                    # else if h = 0 then APPEND(next, GENERATE_CHILDREN(h)) (MAIN:13)
                    # else if LM1(h) != 1 then ... (MAIN:14-19)
                    # generate_succ_left genera i figli tramite strategia succL (corrisponde a GENERATE_CHILDREN)
                    children = generate_succ_left(h, col_vectors, 
                                                 found_mhs_sets=found_mhs_sets,
                                                 col_map=col_map,
                                                 timeout=timeout,
                                                 start_time=start_time)
                    next_level_hypotheses.extend(children)  # APPEND(next, ...) o MERGE(next, ...)
                except TimeoutError:
                    print(format_timeout_message(
                        f"generazione figli al livello {level}",
                        processed=h_idx+1, 
                        total=len(current_hypotheses_to_process)
                    ))
                    raise create_state_exception(TimeoutError, found_mhs, level, stats_per_level, mhs_per_level)
        
        stats_per_level[level] = len(next_level_hypotheses)
        
        # si potrebbe omettere in quanto l'ordinamento è implicito 
        # Ordinamento canonico: garantisce ordine deterministico per BFS (implicito in MERGE)
        if next_level_hypotheses:
            next_level_hypotheses.sort(key=lambda h: h.bin, reverse=True)
        
        level_elapsed = time.time() - level_start_time
        total_mhs = sum(mhs_per_level.values())
        mhs_this_level = mhs_per_level.get(level, 0)
        print(format_level_summary(
            level, 
            len(current_hypotheses_to_process), 
            level_elapsed, 
            mhs_this_level, 
            total_mhs
        ))
        
        max_level_reached = level
        last_saved_state = (found_mhs, level, stats_per_level, mhs_per_level)
        
        current_level_hypotheses = next_level_hypotheses  # current ← next (MAIN:20)
        
        if not current_level_hypotheses:  # until current = <> (MAIN:21)
            break
    
    found_mhs.sort(key=lambda x: x[1])  # Ordinamento finale delle soluzioni per cardinalità crescente
    
    return found_mhs, stats_per_level, mhs_per_level, max_level_reached  # return Delta (MAIN:22)

def main(argv):
    """
    Entry point del solver seriale.
    
    Gestisce il parsing degli argomenti, l'esecuzione dell'algoritmo,
    e la scrittura dei risultati su file.
    
    Args:
        argv: argomenti da riga di comando
    """
    tracemalloc.start()
    start_wall = time.time()
    start_cpu = time.process_time()

    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path = argv[1]
    if not os.path.isfile(in_path):
        print("File non trovato:", in_path)
        sys.exit(1)
    base, _ = os.path.splitext(in_path)
    out_path = base + ".mhs"

    timeout = None
    reduction = True
    outdir = None
    origine = None
    densita = None
    categoria = None
    for arg in argv[2:]:
        if arg.startswith("--timeout="):
            try:
                timeout = int(arg.split("=")[1])
            except ValueError:
                print("Valore non valido per --timeout")
                sys.exit(1)
        elif arg == "--no-reduction":
            reduction = False
        elif arg.startswith("--outdir="):
            outdir = arg.split("=",1)[1]
        elif arg == "--memory-monitoring":
            print("Errore: l'opzione --memory-monitoring non è supportata con la versione seriale")
            sys.exit(1)
        elif arg.startswith("--origine="):
            origine = arg.split("=")[1]
        elif arg.startswith("--densita="):
            densita = float(arg.split("=")[1])
        elif arg.startswith("--categoria="):
            categoria = arg.split("=")[1]
        else:
            print("Opzione sconosciuta:", arg)
            sys.exit(1)


    base, _ = os.path.splitext(os.path.basename(in_path))  # solo il nome file
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, base + ".mhs")
    else:
        base, _ = os.path.splitext(in_path)
        out_path = base + ".mhs"

    rows, comments = parse_matrix_file(in_path)

    if not rows:
        print("Matrice di input vuota o non valida.")
        # Scrive un file di output vuoto con il riassunto
        write_mhs_output(out_path, [], 0, 0, 0, [], True, start_wall, perf=measure_performance(start_wall, start_cpu), solver_name="mhs_solver.py",
                         origine=origine, densita=densita, categoria=categoria)
        sys.exit(0)

    num_rows = len(rows)
    num_cols_original = len(rows[0])

    col_vectors, col_map = [], []
    if reduction:
        col_vectors, col_map = get_column_vectors(rows)
    else:
        num_cols_original = len(rows[0]) if rows else 0
        col_map = list(range(num_cols_original))
        for j in range(num_cols_original):
            vector = 0
            for i in range(num_rows):
                if rows[i][j] == 1:
                    vector |= (1 << i)
            col_vectors.append(vector)

    num_cols_reduced = len(col_vectors)
    removed_cols = sorted(list(set(range(num_cols_original)) - set(col_map)))
    num_cols_non_empty = sum(1 for v in col_vectors if v != 0)

    print(format_parsing_message(num_rows, num_cols_original, num_cols_reduced, 
                                 removed_empty=len(removed_cols)))
    if removed_cols:
        print(f"Colonne vuote rimosse (indice 0-based): {removed_cols}")
    
    if timeout:
        print(f"Timeout impostato: {timeout} secondi")
    
    max_level = max(num_rows, num_cols_non_empty)
    print(f"Esplorazione fino al livello: {max_level}")
    
    start_datetime = datetime.now()
    print(f"\nInizio calcolo: {start_datetime.strftime('%H:%M:%S')}")
    if timeout:
        end_datetime = start_datetime + timedelta(seconds=timeout)
        print(f"Fine prevista: {end_datetime.strftime('%H:%M:%S')}")
    
    found_mhs = []
    stats_per_level = {}
    mhs_per_level = {}
    completed = False
    level_interrupted = -1
    
    start_time_exec = time.time()
    try:
        found_mhs, stats_per_level, mhs_per_level, max_level_reached = mhs_solver(
            col_vectors, col_map, num_rows, max_level, start_time_exec, timeout
        )
        completed = True
    except (KeyboardInterrupt, TimeoutError, MemoryError) as e:
        if hasattr(e, 'args') and len(e.args) > 0 and isinstance(e.args[0], tuple):
            found_mhs, level_interrupted, stats_per_level, mhs_per_level = e.args[0]
        elif last_saved_state:
            found_mhs, level_interrupted, stats_per_level, mhs_per_level = last_saved_state
            print(f"Recuperato stato salvato al livello {level_interrupted} con {len(found_mhs)} MHS trovati.")
        else:
            print("Nessuno stato salvato trovato.")
        
        if isinstance(e, TimeoutError):
            print(f"\n[TIMEOUT] Timeout di {timeout} secondi raggiunto.")
        elif isinstance(e, KeyboardInterrupt):
            print("\n[INTERRUPT] Calcolo interrotto dall'utente.")
        elif isinstance(e, MemoryError):
            print("\n[MEMORIA] Memoria esaurita.")
        
        completed = False
    except Exception:
        print(f"\n[ERRORE] Errore durante il processamento.")
        completed = False

    perf = measure_performance(start_wall, start_cpu)
    
    write_mhs_output(
        out_path, found_mhs, num_rows, num_cols_original, num_cols_reduced, 
        removed_cols, completed, start_time_exec, level_interrupted, timeout, 
        stats_per_level, perf, mhs_per_level, solver_name="mhs_solver.py",
        origine=origine, densita=densita, categoria=categoria
    )
    
    print(f"\nOutput scritto in {out_path}")
    elapsed_total = time.time() - start_time_exec
    print(format_summary_message(len(found_mhs), completed, elapsed_total,
                                 level=level_interrupted if not completed else max_level_reached))
    
    if mhs_per_level and any(count > 0 for count in mhs_per_level.values()):
        print("Distribuzione MHS per livello:")
        for level in sorted(mhs_per_level.keys()):
            if mhs_per_level[level] > 0:
                print(f"  Livello {level}: {mhs_per_level[level]} MHS")
    
    print(f"\nAlgoritmo {'completato' if completed else 'interrotto'}")


stop_event = threading.Event()
listener_thread = threading.Thread(target=input_listener, args=(stop_event,), daemon=True)
listener_thread.start()

if __name__ == "__main__":
    main(sys.argv)