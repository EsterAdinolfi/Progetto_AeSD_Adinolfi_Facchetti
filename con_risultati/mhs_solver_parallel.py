#!/usr/bin/env python3
"""
Solver PARALLELO per il calcolo dei Minimal Hitting Sets (MHS).

Utilizzo:
    python mhs_solver_parallel.py input.matrix [opzioni]

Opzioni:
    --timeout=N          Limite massimo in secondi
    --no-reduction       Non rimuove le colonne vuote
    --outdir=DIR         Cartella di output
    --processes=N        Numero di processi paralleli (default: CPU count)
    --batch-size=SIZE    Soglia minima dimensione batch (default: 1000)
    --memory-monitoring  Abilita monitoraggio RAM continuo
    --memory-threshold=N Soglia percentuale memoria (default: 95, range: 50-99)
    --skip-global-dedup  Salta deduplicazione globale (solo ordinamento)
"""

import sys
import os
import time
import tracemalloc
import psutil
import gc  # Garbage collector per gestione della memoria 
import multiprocessing
from multiprocessing import Pool, cpu_count
import threading
import signal
from datetime import datetime, timedelta
import inspect

from utility import Hypothesis, parse_matrix_file, get_column_vectors
from utility import generate_succ_left
from utility import write_mhs_output, check_memory_usage, measure_performance, input_listener, handle_sigint
from utility import check_timeout, check_interruption, check_memory_with_cleanup
from utility import check_batch_processing_conditions
from utility import check_all_loop_conditions, check_intensive_loop_conditions
from utility import format_timeout_message, format_memory_message, format_interrupt_message
from utility import format_filter_progress
from utility import update_emergency_data, initialize_emergency_input_data, create_state_exception
from utility import signal_stop_and_save, safe_pool_terminate, should_check_iteration
from utility import format_parsing_message, format_level_start_message, format_completion_message
from utility import format_dedup_progress_message, format_summary_message, calculate_percentage
from utility import calculate_rate, calculate_eta
import utility

GLOBAL_COL_VECTORS = None
GLOBAL_COL_MAP = None
GLOBAL_NUM_ROWS = None
GLOBAL_NUM_COLS = None
GLOBAL_STOP_EVENT = None


def worker_initializer(col_vectors, col_map, num_rows, num_cols, stop_event):
    """
    Inizializza le variabili globali per ciascun processo worker.
    
    Args:
        col_vectors: vettori colonne (bitmask)
        col_map: mapping indici colonne ridotte -> originali
        num_rows: numero righe matrice
        num_cols: numero colonne ridotte
        stop_event: evento per interruzione coordinata
    """
    global GLOBAL_COL_VECTORS, GLOBAL_COL_MAP, GLOBAL_NUM_ROWS, GLOBAL_NUM_COLS, GLOBAL_STOP_EVENT
    GLOBAL_COL_VECTORS = col_vectors
    GLOBAL_COL_MAP = col_map
    GLOBAL_NUM_ROWS = num_rows
    GLOBAL_NUM_COLS = num_cols
    GLOBAL_STOP_EVENT = stop_event

    def worker_signal_handler(signum, frame):
        sys.exit(0)
    
    try:
        signal.signal(signal.SIGINT, worker_signal_handler)
        signal.signal(signal.SIGTERM, worker_signal_handler)
    except Exception:
        pass

def worker_process_batch_bins(args):
    """
    Elabora un batch di ipotesi in parallelo, generando figli tramite succL.
    
    PARALLELIZZAZIONE: Ogni worker esegue GENERATE_CHILDREN su un sottoinsieme
    di ipotesi del livello corrente (batch). Questa funzione implementa il loop
    parallelo di MAIN:8-19, dove ogni worker processa un batch di ipotesi h.
    
    GENERATE_CHILDREN (MAIN:13,19): Chiama generate_succ_left per generare i
    figli di ogni ipotesi h del batch usando la strategia succL. Questa garantisce
    che ogni ipotesi sia generata esattamente una volta DENTRO il batch.
    
    IMPORTANTE: Tra batch diversi potrebbero esserci duplicati in edge-cases,
    quindi può servire una deduplicazione globale alla fine del livello.
    
    Args:
        args: tupla (batch_bins, timeout, start_time, found_mhs_sets)
            - batch_bins: lista di valori binari delle ipotesi da elaborare
            - timeout: timeout in secondi
            - start_time: timestamp di inizio algoritmo
            - found_mhs_sets: set di MHS già trovati (per pruning minimalità)
            
    Returns:
        Tupla (children_tuples, mhs_local, timeout_flag, worker_cpu_time)
    """
    worker_start_cpu = time.process_time()
    
    (batch_bins, timeout, start_time, found_mhs_sets) = args 

    col_vectors = GLOBAL_COL_VECTORS
    col_map = GLOBAL_COL_MAP
    num_rows = GLOBAL_NUM_ROWS
    num_cols = GLOBAL_NUM_COLS
    stop_event = GLOBAL_STOP_EVENT

    children_tuples = []
    mhs_local = []
    
    found_mhs_sets_frozenset = {frozenset(mhs) for mhs in found_mhs_sets} if found_mhs_sets else set()

    timeout_reached, _ = check_timeout(timeout, start_time, margin=0.05)
    if timeout_reached:
        worker_cpu_time = time.process_time() - worker_start_cpu
        return children_tuples, mhs_local, True, worker_cpu_time
    
    batch_size = len(batch_bins)
    sub_batch_size = 250

    try:
        for idx in range(0, batch_size, sub_batch_size):
            # Controllo unificato con funzione modulare
            should_stop, reason, _ = check_batch_processing_conditions(
                timeout, start_time, stop_event, memory_limit=100, 
                iteration=idx, check_frequency=1, margin=0.05
            )
            if should_stop:
                worker_cpu_time = time.process_time() - worker_start_cpu
                return children_tuples, mhs_local, True, worker_cpu_time
                
            end_idx = min(idx + sub_batch_size, batch_size)
            for i in range(idx, end_idx):
                bin_val = batch_bins[i]
                
                # Controllo frequente di stop/timeout ogni 2 iterazioni
                if should_check_iteration(i, 2, offset=idx):
                    should_stop, reason, _ = check_batch_processing_conditions(
                        timeout, start_time, stop_event, memory_limit=100,
                        iteration=i, check_frequency=1, margin=0.05
                    )
                    if should_stop:
                        worker_cpu_time = time.process_time() - worker_start_cpu
                        return children_tuples, mhs_local, True, worker_cpu_time

                # Ricostruzione vector dell'ipotesi h (SET_FIELDS)
                # SET_FIELDS:1-5 - calcola vector(h) come OR bitwise delle colonne selezionate
                vector = 0
                card = bin_val.bit_count() if hasattr(bin_val, 'bit_count') else bin(bin_val).count('1')
                
                # Itera sui bit di bin_val per trovare colonne selezionate
                bit_pos = 0
                temp = bin_val
                while temp:
                    if temp & 1:  # Se colonna bit_pos è selezionata
                        vector |= col_vectors[num_cols - 1 - bit_pos]  # OR con vettore colonna (SET_FIELDS:3, PROPAGATE:2)
                    temp >>= 1
                    bit_pos += 1
                
                if should_check_iteration(i, 100, offset=idx):
                    _, mem_percent, _, _, cleaned = check_memory_with_cleanup(memory_limit_percent=90, force_gc=True)
                
                # Crea ipotesi temporanea per generazione figli
                fake_h = Hypothesis(bin_val, num_cols)
                fake_h.vector = vector
                fake_h.card = card
                
                # GENERATE_CHILDREN(h) - genera i figli usando strategia succL (MAIN:13,19)
                # GENERATE_CHILDREN:1-26 - implementato in generate_succ_left
                new_children = generate_succ_left(fake_h, col_vectors, 
                                                 found_mhs_sets=found_mhs_sets_frozenset,
                                                 col_map=col_map)

                check_interval = 25
                for j, child in enumerate(new_children):
                    if should_check_iteration(j, check_interval):
                        should_stop, reason, _ = check_batch_processing_conditions(
                            timeout, start_time, stop_event, memory_limit=100,
                            iteration=j, check_frequency=1, margin=0.05
                        )
                        if should_stop:
                            worker_cpu_time = time.process_time() - worker_start_cpu
                            return children_tuples, mhs_local, True, worker_cpu_time

                    children_tuples.append((child.bin, child.vector, child.card))

    except MemoryError:
        # MemoryError: ritorna immediatamente senza stampare (risparmiare memoria)
        worker_cpu_time = time.process_time() - worker_start_cpu
        # Ritorna risultati parziali con flag timeout=True per segnalare stop
        return children_tuples, mhs_local, True, worker_cpu_time
    except Exception as e:
        # Altri errori: stampa solo se non è MemoryError (già gestito sopra)
        try:
            print(f"[Worker error] {e}")
        except:
            pass  # Se anche il print fallisce, ignora
        worker_cpu_time = time.process_time() - worker_start_cpu
        return children_tuples, mhs_local, False, worker_cpu_time

    worker_cpu_time = time.process_time() - worker_start_cpu
    return children_tuples, mhs_local, False, worker_cpu_time


def dedup_bitset(all_children, seen_bitset, num_cols, timeout=None, start_time=None, dedup_start_time=None, memory_limit_percent=100):
    """
    Deduplicazione figli usando bitset (per livelli piccoli/medi).
    
    NOTA ALGORITMO: La deduplicazione globale NON è presente nello pseudocodice
    teorico perché con succL non dovrebbero esserci duplicati. Tuttavia,
    nella parallelizzazione possono verificarsi edge-cases, quindi questa
    funzione rimuove eventuali duplicati tra batch diversi.
    
    STRATEGIA BITSET: Usa un array di bit per tracciare ipotesi già viste.
    Adatta per livelli piccoli/medi (num_cols <= 24) dove 2^num_cols è gestibile.
    
    Args:
        all_children: lista di tuple (bin, vector, card)
        seen_bitset: array per tracking duplicati
        num_cols: numero colonne
        timeout: timeout in secondi
        start_time: timestamp inizio
        dedup_start_time: timestamp inizio deduplicazione
        memory_limit_percent: soglia memoria percentuale
        
    Returns:
        Tupla (unique, duplicates_found)
    """
    unique = []
    duplicates_found = 0
    progress_interval = 500000  # Progresso ogni 500k per bitset
    memory_check_interval = 2.0  # Controllo memoria ogni 2 secondi
    last_memory_check = time.time()
    
    for i, (bin_val, vec, card) in enumerate(all_children):
        # Controllo unificato ogni 10000 ipotesi per massima reattività
        if should_check_iteration(i, 10000):
            current_time = time.time()
            check_result = check_intensive_loop_conditions(
                timeout, start_time, memory_limit_percent,
                current_time, last_memory_check, memory_check_interval
            )
            last_memory_check = check_result['last_memory_check']
            
            if check_result['should_stop']:
                elapsed_dedup = time.time() - dedup_start_time
                if check_result['reason'] == 'interrupt':
                    print(format_interrupt_message("deduplicazione bitset", i, len(all_children), elapsed_dedup))
                    raise KeyboardInterrupt
                elif check_result['reason'] == 'timeout':
                    print(format_timeout_message("deduplicazione bitset", i, len(all_children), 
                                                check_result['remaining_time'], elapsed_dedup))
                    raise TimeoutError
                elif check_result['reason'] == 'memory':
                    print(f"\nMemoria insufficiente dopo pulizia ({check_result['mem_percent']:.1f}%)")
                    raise MemoryError(f"Utilizzo memoria troppo elevato durante deduplicazione: {check_result['mem_percent']:.1f}%")
        
        # Progresso per volumi grandi
        if len(all_children) > 1000000 and should_check_iteration(i, progress_interval) and i > 0:
            elapsed_dedup = time.time() - dedup_start_time
            print(f"\r    {format_dedup_progress_message('bitset', i, len(all_children), elapsed_dedup, duplicates_found)}", end="", flush=True)
        
        if not seen_bitset[bin_val]:
            seen_bitset[bin_val] = 1
            unique.append((bin_val, vec, card))
        else:
            duplicates_found += 1
    
    return unique, duplicates_found

def dedup_sorted(all_children, timeout=None, start_time=None, dedup_start_time=None, memory_limit_percent=100):
    """
    Deduplicazione figli usando ordinamento (per livelli grandi).
    
    NOTA ALGORITMO: Deduplicazione non presente nello pseudocodice teorico.
    Necessaria solo per gestire edge-cases della parallelizzazione.
    
    STRATEGIA SORTED: Ordina le ipotesi per bin value, poi rimuove duplicati
    consecutivi in un singolo passaggio. Adatta per livelli medi (24 < num_cols <= 128)
    dove bitset non è applicabile ma il volume è gestibile.
    
    Args:
        all_children: lista di tuple (bin, vector, card)
        timeout: timeout in secondi
        start_time: timestamp inizio
        dedup_start_time: timestamp inizio deduplicazione
        memory_limit_percent: soglia memoria percentuale
        
    Returns:
        Tupla (unique, duplicates_found)
    """
    print(f"    Ordinamento di {len(all_children)} ipotesi...")
    sort_start = time.time()
    
    timeout_reached, remaining = check_timeout(timeout, start_time, margin=1.0)
    if timeout_reached:
        print(format_timeout_message("ordinamento (preparazione)", remaining_time=remaining))
        raise TimeoutError
    
    is_mem_crit, mem_percent, mem_avail_mb, _, cleaned = check_memory_with_cleanup(
        memory_limit_percent, force_gc=True
    )
    if mem_percent >= memory_limit_percent:
        print(format_memory_message("ordinamento", mem_percent, mem_avail_mb, after_cleanup=True))
        raise MemoryError(f"Utilizzo memoria troppo elevato prima dell'ordinamento: {mem_percent:.1f}%")
    
    all_children.sort(key=lambda x: x[0])
    sort_time = time.time() - sort_start
    print(f"    {format_completion_message('Ordinamento', sort_time, count=len(all_children))}")
    
    unique = []
    duplicates_found = 0
    last_bin = None
    progress_interval = 200000
    memory_check_interval = 2.0
    last_memory_check = time.time()
    
    for i, (bin_val, vec, card) in enumerate(all_children):
        # Controllo unificato ogni 10000 ipotesi per massima reattività
        if should_check_iteration(i, 10000):
            current_time = time.time()
            check_result = check_intensive_loop_conditions(
                timeout, start_time, memory_limit_percent,
                current_time, last_memory_check, memory_check_interval
            )
            last_memory_check = check_result['last_memory_check']
            
            if check_result['should_stop']:
                elapsed_dedup = time.time() - dedup_start_time
                if check_result['reason'] == 'interrupt':
                    print(format_interrupt_message("deduplicazione sorted", i, len(all_children), elapsed_dedup))
                    raise KeyboardInterrupt
                elif check_result['reason'] == 'timeout':
                    print(format_timeout_message("deduplicazione sorted", i, len(all_children), 
                                                check_result['remaining_time'], elapsed_dedup))
                    raise TimeoutError
                elif check_result['reason'] == 'memory':
                    print(f"\nMemoria insufficiente dopo pulizia ({check_result['mem_percent']:.1f}%)")
                    raise MemoryError(f"Utilizzo memoria troppo elevato durante deduplicazione sorted: {check_result['mem_percent']:.1f}%")
        
        if len(all_children) > 500000 and should_check_iteration(i, progress_interval) and i > 0:
            elapsed_dedup = time.time() - dedup_start_time
            print(f"\r    {format_dedup_progress_message('sorted', i, len(all_children), elapsed_dedup, duplicates_found)}\t", end="", flush=True)
        
        if bin_val == last_bin:
            duplicates_found += 1
            continue
        last_bin = bin_val
        unique.append((bin_val, vec, card))
        
        # Libera memoria periodicamente rilasciando elementi della lista
        if should_check_iteration(i, 1000000) and i > 0:
            del all_children[:i]
            i = 0
            gc.collect()
    
    return unique, duplicates_found

def dedup_distributed(all_children, num_processes, timeout=None, start_time=None, dedup_start_time=None, memory_limit_percent=100):
    """
    Deduplicazione distribuita (per livelli molto grandi).
    
    NOTA ALGORITMO: Deduplicazione non presente nello pseudocodice teorico.
    Necessaria solo per gestire edge-cases della parallelizzazione.
    
    STRATEGIA DISTRIBUTED: Partiziona le ipotesi in bucket (usando hash modulo),
    poi deduplica ogni bucket separatamente. Adatta per livelli molto grandi
    (num_cols > 128) dove sorted richiederebbe troppa memoria.
    
    Args:
        all_children: lista di tuple (bin, vector, card)
        num_processes: numero di bucket per partizionamento
        timeout: timeout in secondi
        start_time: timestamp inizio
        dedup_start_time: timestamp inizio deduplicazione
        memory_limit_percent: soglia memoria percentuale
        
    Returns:
        Tupla (unique, duplicates_found)
    """
    print(f"    Partizionamento in {num_processes} bucket...")
    partition_start = time.time()
    
    timeout_reached, remaining = check_timeout(timeout, start_time, margin=1.0)
    if timeout_reached:
        print(format_timeout_message("partizionamento (preparazione)", remaining_time=remaining))
        raise TimeoutError
    
    is_mem_crit, mem_percent, mem_avail_mb, _, cleaned = check_memory_with_cleanup(
        memory_limit_percent, force_gc=True
    )
    if mem_percent >= memory_limit_percent:
        print(format_memory_message("partizionamento", mem_percent, mem_avail_mb, after_cleanup=True))
        print(f"Operazione troppo grande per la memoria disponibile")
        raise MemoryError(f"Utilizzo memoria troppo elevato prima del partizionamento: {mem_percent:.1f}% (solo {mem_avail_mb:.0f}MB liberi)")
    
    buckets = [[] for _ in range(num_processes)]
    memory_check_interval = 2.0
    last_memory_check = time.time()
    
    for i, child in enumerate(all_children):
        if should_check_iteration(i, 50000):
            current_time = time.time()
            check_result = check_intensive_loop_conditions(
                timeout, start_time, memory_limit_percent,
                current_time, last_memory_check, memory_check_interval
            )
            last_memory_check = check_result['last_memory_check']
            
            if check_result['should_stop']:
                if check_result['reason'] == 'interrupt':
                    print(format_interrupt_message("partizionamento", i, len(all_children)))
                    raise KeyboardInterrupt
                elif check_result['reason'] == 'timeout':
                    print(format_timeout_message("partizionamento", i, len(all_children), 
                                                check_result['remaining_time']))
                    raise TimeoutError
                elif check_result['reason'] == 'memory':
                    print(f"\nMemoria insufficiente dopo pulizia ({check_result['mem_percent']:.1f}%)")
                    raise MemoryError(f"Utilizzo memoria troppo elevato durante partizionamento: {check_result['mem_percent']:.1f}%")
            
        bin_val = child[0]
        buckets[bin_val % num_processes].append(child)
    
    partition_time = time.time() - partition_start
    bucket_sizes = [len(b) for b in buckets]
    print(f"    {format_completion_message('Partizionamento', partition_time, count=len(all_children))}")
    print(f"    Dimensioni bucket: min={min(bucket_sizes)}, max={max(bucket_sizes)}, avg={sum(bucket_sizes)/len(bucket_sizes):.0f}")
    
    del all_children
    gc.collect()
    
    unique = []
    total_duplicates = 0
    
    for bucket_idx, bucket in enumerate(buckets):
        if not bucket:
            continue
            
        if timeout:
            timeout_reached, remaining_time = check_timeout(timeout, start_time, margin=0.5)
            if timeout_reached:
                print(f"\nTimeout imminente durante elaborazione bucket {bucket_idx+1}/{num_processes} ({remaining_time:.1f}s rimanenti)")
                raise TimeoutError
        
        is_mem_crit, mem_percent, mem_avail_mb, _, cleaned = check_memory_with_cleanup(memory_limit_percent, force_gc=True)
        if mem_percent >= memory_limit_percent:
            print(f"\nMemoria ancora critica dopo pulizia ({mem_percent:.1f}%), procedendo comunque...")
        
        seen = set()
        bucket_dups = 0
        for bin_val, vec, card in bucket:
            if bin_val not in seen:
                seen.add(bin_val)
                unique.append((bin_val, vec, card))
            else:
                bucket_dups += 1
        
        total_duplicates += bucket_dups
        
        # Libera memoria dopo ogni bucket
        buckets[bucket_idx] = None  # Rilascia la memoria di questo bucket
        gc.collect()
        
        # Progresso bucket
        if len(buckets) > 4:  # Solo se abbiamo molti bucket
            print(f"    Bucket {bucket_idx+1}/{num_processes}: {len(bucket)} -> {len(bucket)-bucket_dups} unici ({bucket_dups} dup)")
    
    return unique, total_duplicates


_emergency_data = {'found_mhs': [], 'stats': {}, 'mhs_per_level': {}, 'level': 0} # campi mutabili 

def get_emergency_data():
    global _emergency_data
    return _emergency_data

def mhs_solver_parallel(col_vectors, col_map, num_rows,
                         max_level, start_time, timeout=None,
                         batch_size=None, num_processes=None, memory_limit_percent=100,
                         skip_global_dedup=False):
    """
    Algoritmo PARALLELO per il calcolo dei Minimal Hitting Sets (MHS).
    
    Implementa lo pseudocodice MAIN:1-22 con PARALLELIZZAZIONE per livelli.
    
    STRUTTURA ALGORITMO (seguendo MAIN):
    MAIN:2-5  - Inizializzazione: ho ← 0, current ← <ho>, Delta ← <>
    MAIN:6-21 - Loop repeat per esplorazione livelli BFS
    MAIN:7    - next ← <> (ipotesi livello successivo)
    MAIN:8    - for each h in current (PARALLELIZZATO su worker)
    MAIN:9-11 - CHECK(h): se h è soluzione, APPEND(Delta, h)
    MAIN:13,19- GENERATE_CHILDREN(h): genera figli con succL (nei worker)
    MAIN:20   - current ← next (passa al livello successivo)
    MAIN:21   - until current = <> (termina se nessun figlio)
    MAIN:22   - return Delta (restituisce MHS trovati)
    
    PARALLELIZZAZIONE:
    - Ogni livello k viene suddiviso in BATCH di ipotesi
    - Ogni batch viene assegnato a un WORKER che esegue:
      * CHECK(h) per ogni h del batch
      * GENERATE_CHILDREN(h) per ogni h non-MHS usando succL
    - Raccolta risultati: i figli di tutti i worker formano next
    - Deduplicazione globale: rimuove eventuali duplicati tra batch
      (non necessaria teoricamente con succL, ma utile per edge-cases)
      * Strategie: bitset (piccoli), sorted (medi), distributed (grandi)
    
    Args:
        col_vectors: vettori colonne (bitmask righe)
        col_map: mapping indici ridotti -> originali
        num_rows: numero righe matrice
        max_level: livello massimo esplorazione
        start_time: timestamp inizio
        timeout: timeout in secondi (opzionale)
        batch_size: dimensione batch per parallelizzazione
        num_processes: numero processi worker
        memory_limit_percent: soglia memoria percentuale
        skip_global_dedup: se True, salta deduplicazione globale (solo ordinamento)
        
    Returns:
        Tupla (found_mhs, stats_per_level, mhs_per_level, max_level_reached, all_worker_cpu_times)
    """
    if utility.stop_requested:
        raise KeyboardInterrupt
    
    def emergency_exit(signum, frame):
        print("\nInterruzione richiesta dall'utente (Ctrl+C). Salvataggio immediato...")
        utility.stop_requested = True
        
        if timeout:
            elapsed = time.time() - start_time
            remaining_timeout = timeout - elapsed
            if remaining_timeout <= 0.5:
                print(f"Timeout rilevato - terminazione rapida senza salvataggio complesso (trascorso: {elapsed:.1f}s, rimanente: {remaining_timeout:.1f}s)")
                raise KeyboardInterrupt
        
        try:
            current_level = get_emergency_data().get('level', 0)
            
            if current_level == 0:
                try:
                    stack = inspect.stack()
                    for frame_info in stack:
                        if 'level' in frame_info[0].f_locals:
                            current_level = frame_info[0].f_locals['level']
                            break
                except:
                    pass
                    
            if current_level > 0:
                print(f"Interruzione durante l'elaborazione del livello {current_level}")
            
            input_file = None
            for arg in sys.argv:
                if arg.endswith('.matrix'):
                    input_file = arg
                    break
            
            if input_file:
                input_dir = os.path.dirname(input_file)
                base = os.path.splitext(os.path.basename(input_file))[0]
                out_path = os.path.join(input_dir, base + ".mhs") if input_dir else base + ".mhs"
                
                elapsed = time.time() - start_time
                
                emergency_data_obj = get_emergency_data()
                # campi immutabili (dati di input)
                n_rows = emergency_data_obj.get('num_rows', 0)
                m_original = emergency_data_obj.get('num_cols_original', 0)
                m_reduced = len(emergency_data_obj.get('col_vectors', []))
                removed_cols_emergency = emergency_data_obj.get('removed_cols', [])
                origine = emergency_data_obj.get('origine')
                densita = emergency_data_obj.get('densita')
                categoria = emergency_data_obj.get('categoria')
                completed = False
                
                found_mhs = []
                stats = {}
                mhs_per_level = {}
                worker_cpu_times_emergency = []
                level_interrupted = current_level
                
                if 'found_mhs' in emergency_data_obj and emergency_data_obj['found_mhs']:
                    found_mhs = emergency_data_obj['found_mhs']
                
                if 'stats' in emergency_data_obj and emergency_data_obj['stats']:
                    stats = emergency_data_obj['stats']
                    
                if 'mhs_per_level' in emergency_data_obj and emergency_data_obj['mhs_per_level']:
                    mhs_per_level = emergency_data_obj['mhs_per_level']
                
                if 'worker_cpu_times' in emergency_data_obj and emergency_data_obj['worker_cpu_times']:
                    worker_cpu_times_emergency = emergency_data_obj['worker_cpu_times']
                
                # Gli MHS contengono già indici originali (mappati tramite col_map nel worker quindi non serve filtrarli)
                valid_mhs = found_mhs
                
                # Calcola le performance per il salvataggio di emergenza
                perf_emergency = measure_performance(start_time, time.process_time(), worker_cpu_times_emergency)
                
                utility.write_mhs_output(
                    out_path=out_path,
                    found_mhs=valid_mhs,
                    n_rows=n_rows,
                    m_original=m_original,
                    m_reduced=m_reduced,
                    removed_cols=removed_cols_emergency,  
                    completed=completed,
                    start_time=start_time,
                    level_interrupted=level_interrupted,
                    timeout=timeout,
                    stats_per_level=stats,
                    mhs_per_level=mhs_per_level,
                    perf=perf_emergency,
                    origine=origine,
                    densita=densita,
                    categoria=categoria
                )
                
                print(f"\nSalvati {len(valid_mhs)} MHS parziali nel file {out_path}")
                print(f"Tempo totale: {elapsed:.1f}s, Livello interrotto: {level_interrupted}")
                if len(valid_mhs) < len(found_mhs):
                    print(f"Attenzione: {len(found_mhs) - len(valid_mhs)} MHS sono stati filtrati perché contenevano indici di colonna non validi")
                
                # Verifica che il file sia stato effettivamente scritto
                if os.path.exists(out_path):
                    print(f"File salvato correttamente in: {os.path.abspath(out_path)}")
                else:
                    print(f"ATTENZIONE: il file non sembra essere stato salvato in: {os.path.abspath(out_path)}")
        except Exception:
            print(f"Errore nel salvataggio di emergenza.")
        
        raise KeyboardInterrupt
    
    try:
        signal.signal(signal.SIGINT, emergency_exit)
    except:
        pass
        
    gc.collect()

    # MAIN:5 - Delta ← <> (sequenza soluzioni trovate)
    found_mhs = []  # Delta: lista di MHS trovati (colonne, cardinalità)
    stats_per_level = {}  # Statistiche per livello (numero ipotesi generate)
    mhs_per_level = {}  # MHS trovati per livello
    found_mhs_sets = set()  # Set per pruning: evita duplicati e sovra-insiemi
    max_level_reached = 0
    
    all_worker_cpu_times = []
    pool_already_terminated = False  # Flag per tracciare se il pool è già stato terminato

    num_cols = len(col_vectors)

    if num_cols <= 24:
        dedup_mode = "bitset"
    elif num_cols <= 128:
        dedup_mode = "sorted"
    else:
        dedup_mode = "distributed"

    if skip_global_dedup:
        print(f"Deduplicazione globale: DISABILITATA (solo ordinamento canonico)")
        print(f"NOTA: Teoricamente non necessaria con succL, ma può causare duplicati in edge-cases")
    else:
        print(f"Deduplicazione globale: ATTIVA (strategia: {dedup_mode})")

    # MAIN:2-4 - Inizializzazione con ipotesi vuota
    # ho ← 0 (MAIN:2), SET_FIELDS(ho) (MAIN:3), current ← <ho> (MAIN:4)
    h0 = (0, 0, 0)  # (bin, vector, card) - ipotesi vuota
    current_level = [h0]  # current: sequenza ipotesi livello corrente (MAIN:4)
    stats_per_level[0] = 1

    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    if batch_size is None:
        batch_size = 1000
    stop_event = multiprocessing.Event()
    pool = Pool(processes=num_processes, initializer=worker_initializer,
                initargs=(col_vectors, col_map, num_rows, num_cols, stop_event))

    memory_check_interval = 0.5  # Controllo utilizzo memoria ogni mezzo secondo
    timeout_check_interval = 0.2  # Controllo timeout ogni 0.2 secondi
    last_memory_check = time.time()
    last_timeout_check = time.time()

    try:
        # MAIN:6-21 - repeat loop: coordinamento esplorazione livelli BFS
        # Ogni iterazione processa un livello k, generando le ipotesi del livello k+1
        for level in range(0, max_level + 1):  # repeat (MAIN:6)
            # Controllo unificato COMPLETO (timeout + interruzione + memoria)
            current_time = time.time()
            check_result = check_all_loop_conditions(
                timeout, start_time, stop_event, memory_limit_percent,
                current_time, last_timeout_check, last_memory_check,
                timeout_check_interval, memory_check_interval
            )
            
            # Aggiorna i timestamp
            last_timeout_check = check_result['last_timeout_check']
            last_memory_check = check_result['last_memory_check']
            
            # Gestione risultato del controllo
            if check_result['should_stop']:
                emergency_data_obj = get_emergency_data()
                update_emergency_data(emergency_data_obj, found_mhs, level - 1, stats_per_level, mhs_per_level, all_worker_cpu_times)
                
                if check_result['reason'] == 'timeout':
                    print(format_timeout_message(f"livello {level}", remaining_time=check_result['remaining_time']))
                    print("Interruzione preventiva per salvare risultati")
                    raise create_state_exception(TimeoutError, found_mhs, level - 1, stats_per_level, mhs_per_level)
                elif check_result['reason'] == 'memory':
                    print(f"\nUtilizzo memoria critico ({check_result['mem_percent']:.1f}%, solo {check_result['mem_avail_mb']:.1f}MB liberi)")
                    raise MemoryError(f"Utilizzo memoria troppo elevato: {check_result['mem_percent']:.1f}%")
                elif check_result['reason'] == 'interrupt':
                    raise KeyboardInterrupt
            
            emergency_data_obj = get_emergency_data()
            update_emergency_data(emergency_data_obj, found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
            
            if check_interruption(stop_event):
                raise KeyboardInterrupt
            if timeout:
                elapsed = time.time() - start_time
                remaining_timeout = timeout - elapsed
                if remaining_timeout <= 0.1:
                    print(f"Timeout principale: trascorso {elapsed:.1f}s, rimanente {remaining_timeout:.1f}s")
                    raise TimeoutError((found_mhs, level - 1, stats_per_level, mhs_per_level))

            print(format_level_start_message(level, len(current_level)))

            if level >= 2 and timeout:
                elapsed = time.time() - start_time
                remaining = timeout - elapsed
                elapsed_pct = calculate_percentage(elapsed, timeout)
                remaining_pct = max(0, calculate_percentage(remaining, timeout))
                if remaining > 0:
                    estimated_end = datetime.now() + timedelta(seconds=remaining)
                    print(f"    Tempo: {elapsed:.1f}s/{timeout}s "
                          f"({elapsed_pct:.1f}% trascorso, {remaining_pct:.1f}% rimanente)")
                    print(f"    Fine stimata: {estimated_end.strftime('%H:%M:%S')} (fra {remaining:.1f}s)")
                else:
                    print(f"    Tempo: {elapsed:.1f}s/{timeout}s (SCADUTO)")
                    
            if not current_level:
                print("Nessuna ipotesi rimasta.")
                break

            # Maschera per CHECK(h): tutte le righe coperte
            all_rows_mask = (1 << num_rows) - 1  # CHECK:2 - 2^|N| - 1
            current_level_non_mhs = []  # Ipotesi non-MHS da passare ai worker
            
            # MAIN:8 - for each h in current (controllo soluzioni PRIMA dei worker)
            # Separa MHS da non-MHS: solo i non-MHS vanno ai worker per generare figli
            for bin_val, vector, card in current_level:
                if timeout:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        print(f"\nTimeout durante controllo MHS")
                        raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
                
                # MAIN:9 - if CHECK(h) then
                # CHECK:2-3 - verifica se vector(h) copre tutte le righe
                if vector == all_rows_mask:  # CHECK(h): tutte le righe coperte?
                    # Estrae indici colonne originali da bin_val
                    mhs_cols = [col_map[j] for j in range(num_cols) if (bin_val >> (num_cols - 1 - j)) & 1]
                    mhs_cols_set = frozenset(mhs_cols)
                    
                    # MAIN:10 - APPEND(Delta, h)
                    found_mhs.append((mhs_cols, card))  # APPEND(Delta, h) (MAIN:10)
                    found_mhs_sets.add(mhs_cols_set)  # Per pruning
                    if card not in mhs_per_level:
                        mhs_per_level[card] = 0
                    mhs_per_level[card] += 1
                    # MAIN:11 - remove h from current (implicito: h non va ai worker)
                else:
                    # MAIN:12-19 - else: h non è MHS, genera figli
                    # h va ai worker per GENERATE_CHILDREN (MAIN:13,19)
                    current_level_non_mhs.append((bin_val, vector, card))
            
            current_level = current_level_non_mhs
            
            # MAIN:21 - until current = <>
            # Se current_level è vuoto dopo aver rimosso gli MHS, termina
            if not current_level:
                print(f"Tutte le ipotesi del livello {level} erano MHS, nessun figlio da generare.")
                break  # until current = <> (MAIN:21)

            batch_start_time = time.time()
            print(f"\n  Preparazione batches per {len(current_level)} ipotesi...")
            
            progress_update_timer = time.time()
            
            bins_start = time.time()
            print(f"    Calcolo set bins ({len(current_level)} elementi)...", end="", flush=True)
            current_bins = {h[0] for h in current_level}
            bins_time = time.time() - bins_start
            print(f" {format_completion_message('calcolo', bins_time)}")
            
            num_hyp = len(current_level)
            
            # Calcolo dimensione batch adattiva basata sul livello corrente
            # La dimensione batch è sempre calcolata adattativamente considerando:
            # - Numero di ipotesi da processare (num_hyp)
            # - Numero di processi paralleli disponibili (num_processes)
            # - Fattore adattivo crescente con il livello (adaptive_factor)
            # Il parametro batch_size dell'utente funge da SOGLIA MINIMA per garantire
            # che i batch non siano troppo piccoli (evitando overhead eccessivo).
            adaptive_factor = min(3, 1 + level // 3)  # Aumenta gradualmente con il livello
            batch_size_local = max(batch_size, (num_hyp // (num_processes * adaptive_factor)))
            
            # Bilanciamento del carico: cerca di creare batch di dimensioni simili
            # con un numero di batch che è un multiplo di num_processes
            # NOTA: Se il numero di ipotesi è molto piccolo (< batch_size), non ha senso
            # creare più batch di quanti ne servono effettivamente
            if num_hyp < batch_size_local:
                # Per insiemi molto piccoli, crea un solo batch
                num_batches = 1
                batch_size_local = num_hyp
            else:
                num_batches = max(num_processes, (num_hyp + batch_size_local - 1) // batch_size_local)
                if num_batches % num_processes != 0:
                    num_batches = ((num_batches // num_processes) + 1) * num_processes
                
                batch_size_local = (num_hyp + num_batches - 1) // num_batches
            
            # Mostra progresso iniziale per creazione batch
            if num_hyp > 5000:
                print(f"  Preparazione batch per {num_hyp} ipotesi...")
                
            batches = []
            
            update_interval = max(1, num_hyp // 100000)
            last_update = time.time()
            
            batch_creation_start = time.time()
            progress_update_timer = time.time()
            progress_update_interval = 5.0
            print(f"    Creazione di {num_batches} batch per {num_hyp} ipotesi...")
            
            for i in range(0, num_hyp, batch_size_local):
                if timeout and should_check_iteration(i, 1000):
                    remaining_time = timeout - (time.time() - start_time)
                    if remaining_time <= 1.0:
                        print(f"\n  Timeout imminente durante la creazione dei batch ({remaining_time:.1f}s rimanenti), interruzione")
                        update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                        raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
                
                end_idx = min(i + batch_size_local, num_hyp)
                batches.append([h[0] for h in current_level[i:end_idx]])
                
                current_time = time.time()
                if (i % update_interval == 0 and current_time - last_update > 0.5) or (current_time - progress_update_timer >= progress_update_interval):
                    progress = calculate_percentage(i, num_hyp)
                    elapsed = current_time - batch_creation_start
                    rate = calculate_rate(i, elapsed)
                    eta_remaining = calculate_eta(num_hyp - i, rate)
                    remaining = eta_remaining if eta_remaining else 0
                    
                    remaining_time = timeout - (time.time() - start_time) if timeout else float('inf')
                    timeout_str = f" | Timeout globale: {remaining_time:.1f}s" if timeout and remaining_time < 60 else ""

                    print(f"\r    Creazione batch: {i}/{num_hyp} ({progress:.1f}%) - Tempo creazione: {elapsed:.1f}s, stima completamento: {remaining:.1f}s{timeout_str}", end="", flush=True)
                    last_update = current_time
                    progress_update_timer = current_time
                    
                    update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                    
                    remaining_time = timeout - (time.time() - start_time) if timeout else float('inf')
                    if timeout and remaining_time <= 1.0:
                        print(f"\n  Timeout imminente durante la creazione dei batch ({remaining_time:.1f}s rimanenti), interruzione")
                        raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
            
            batch_duration = time.time() - batch_start_time
            print(f"\n    {format_completion_message('Creazione batch', batch_duration, count=len(batches))}")
            
            print(f"  Suddivisione in {len(batches)} batch (batch_size={batch_size_local}) su {num_processes} processi. Tempo: {batch_duration:.1f}s")

            # Controllo memoria PREVENTIVO prima di lanciare il batch
            is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb = check_memory_usage()
            # Usa una soglia leggermente più bassa (5% meno) per avere margine
            preventive_threshold = max(memory_limit_percent - 5, 80)
            if mem_percent >= preventive_threshold:
                print(f"\n  [PREVENZIONE] Memoria alta rilevata PRIMA del batch: {mem_percent:.1f}% (soglia preventiva: {preventive_threshold:.0f}%)")
                print(f"  Tentativo pulizia memoria prima di procedere...")
                gc.collect()
                time.sleep(0.1)
                is_mem_crit, mem_percent_after, mem_avail_mb_after, mem_used_mb_after = check_memory_usage()
                print(f"  Memoria dopo pulizia: {mem_percent_after:.1f}% ({mem_avail_mb_after:.0f}MB liberi)")
                
                if mem_percent_after >= memory_limit_percent:
                    print(f"\n  MEMORIA CRITICA: impossibile procedere con batch (rischio esaurimento)")
                    update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                    raise MemoryError(f"Memoria critica prima del batch: {mem_percent_after:.1f}%")

            # Prepara snapshot MHS per pruning nei worker
            found_mhs_snapshot = [m for m, _ in found_mhs]
            task_args = [(b, timeout, start_time, found_mhs_snapshot) for b in batches]

            # PARALLELIZZAZIONE: MAIN:8-19 distribuito su worker
            # Ogni worker esegue for each h in batch:
            #   - CHECK(h) (MAIN:9)
            #   - GENERATE_CHILDREN(h) (MAIN:13,19)
            async_res = pool.map_async(worker_process_batch_bins, task_args)

            status_update_interval = 1.0
            memory_check_interval = 0.2
            last_status = time.time()
            last_memory_check = time.time()
            while not async_res.ready():
                time.sleep(0.05)  # Check ogni 50ms per maggiore responsività
                
                # Controlli di interruzione e timeout 
                if check_interruption(stop_event):
                    print(f"\nInterruzione rilevata durante batch processing al livello {level}")
                    print("Terminazione immediata - salvando risultati parziali...")
                    stop_event.set()  
                    safe_pool_terminate(pool, "interruzione batch")
                    # Salvataggio esplicito prima di sollevare l'eccezione
                    update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                    # NON aspettare pool.join() per terminazione immediata
                    raise KeyboardInterrupt
                if timeout:
                    elapsed = time.time() - start_time
                    remaining_timeout = timeout - elapsed
                    # Margine (1s) nel polling per interrompere prima delle fasi costose
                    if remaining_timeout <= 1.0:
                        print(f"\nTimeout imminente, interrompendo elaborazione al livello {level}")
                        print(f"Tempo trascorso: {elapsed:.1f}s, Rimanente: {remaining_timeout:.1f}s")
                        stop_event.set()  # Segnala ai worker di terminare immediatamente
                        safe_pool_terminate(pool, "timeout polling")
                        # Salvataggio esplicito prima di sollevare l'eccezione
                        update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                        # NON aspettare pool.join() per terminazione immediata
                        raise TimeoutError((found_mhs, level - 1, stats_per_level, mhs_per_level))
                
                # Controllo memoria FREQUENTE (ogni 200ms) - separato dallo status update
                # Questo è critico per prevenire che la memoria raggiunga il 100%
                if time.time() - last_memory_check >= memory_check_interval:
                    is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb = check_memory_usage() 
                    
                    if mem_percent >= memory_limit_percent:
                        print(f"\nMEMORIA CRITICA RILEVATA: {mem_percent:.1f}% (soglia: {memory_limit_percent:.0f}%)")
                        
                        # Salva immediatamente lo stato prima di qualsiasi operazione che possa fallire
                        update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                        print(f"Stato salvato: {len(found_mhs)} MHS trovati al livello {level}")
                        
                        # Segnala stop PRIMA di terminare il pool
                        print(f"Segnalazione stop ai worker...")
                        try:
                            stop_event.set()
                        except:
                            pass
                        
                        # Terminazione IMMEDIATA e AGGRESSIVA senza aspettare risposte
                        print(f"Terminazione IMMEDIATA del pool...")
                        try:
                            # Non usare close() che aspetta i task, usa solo terminate()
                            pool.terminate()
                            pool_already_terminated = True  # Segnala che il pool è già terminato
                            # Non aspettare join() che può bloccarsi se i worker sono in MemoryError
                        except:
                            # Ignora TUTTI gli errori durante terminate - se fallisce è già troppo tardi
                            pool_already_terminated = True  # Comunque segnala come terminato
                            pass
                        
                        # Cleanup aggressivo della memoria SENZA delay
                        print(f"Pulizia memoria emergenza...")
                        try:
                            # Cancella le variabili più grandi prima del GC
                            del async_res
                            del batches
                        except:
                            pass
                        
                        # GC multipli rapidi
                        for _ in range(5):
                            try:
                                gc.collect()
                            except:
                                pass
                        
                        # Verifica finale (best effort)
                        try:
                            is_mem_crit, mem_percent_after, mem_avail_mb_after, _, cleaned = check_memory_with_cleanup(memory_limit_percent, force_gc=False)
                            print(f"Memoria dopo cleanup: {mem_percent_after:.1f}% ({mem_avail_mb_after:.0f}MB liberi)")
                        except:
                            print(f"Memoria cleanup completato (verifica fallita)")
                        
                        print(f"Interruzione per esaurimento memoria. MHS parziali: {len(found_mhs)}")
                        print(f"Procedendo con salvataggio risultati parziali...\n", flush=True)
                        
                        raise MemoryError(f"Utilizzo memoria troppo elevato: {mem_percent:.1f}%")
                    
                    last_memory_check = time.time()
                
                if time.time() - last_status >= status_update_interval:
                    elapsed = time.time() - start_time
                    elapsed_batch = time.time() - (last_status - status_update_interval)
                    is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb = check_memory_usage()
                    remaining = timeout - elapsed if timeout else None
                    rem_str = f"{remaining:.1f}s" if remaining and remaining > 0 else "N/A"
                    
                    batch_info = ""
                    try:
                        completed = async_res._number_left
                        if hasattr(async_res, '_number_left') and hasattr(async_res, '_chunksize'):
                            total = len(batches)
                            done = total - async_res._number_left
                            if done > 0:
                                batch_info = f", Batch: {done}/{total} ({calculate_percentage(done, total):.1f}%)"
                    except:
                        pass

                    batch_rate = f", {elapsed_batch:.1f}s/batch" if elapsed_batch > 0 else ""
                    
                    print(f"\r  [Livello {level}] In corso. Trascorsi: {elapsed:.1f}s, Rimanenti: {rem_str}{batch_info}{batch_rate}, Memoria: {mem_percent:.1f}% ({mem_avail_mb:.0f}MB liberi)", end="", flush=True)
                    
                    if timeout and remaining is not None and remaining <= 1.0:
                        print(f"\nTimeout imminente ({remaining:.1f}s rimanenti), interruzione forzata...")
                        stop_event.set()
                        update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                        raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
                    
                    last_status = time.time()
            
            print("")  
            
            try:
                remaining_time = timeout - (time.time() - start_time) if timeout else None
                get_timeout = max(0.5, remaining_time) if remaining_time else None
                results = async_res.get(timeout=get_timeout)
            except multiprocessing.TimeoutError:
                print(f"\nTimeout durante attesa risultati batch al livello {level}")
                stop_event.set()
                # Tenta di recuperare risultati parziali per salvare worker_cpu_times
                partial_results = []
                try:
                    if async_res.ready():
                        partial_results = async_res.get(timeout=0.1)
                except:
                    pass
                # Estrai worker_cpu_times dai risultati parziali se disponibili
                if partial_results:
                    worker_cpu_times_partial = []
                    for result in partial_results:
                        if len(result) >= 4:
                            worker_cpu_times_partial.append(result[3])
                    if worker_cpu_times_partial:
                        all_worker_cpu_times.append(worker_cpu_times_partial)
                        print(f"Recuperati tempi CPU da {len(worker_cpu_times_partial)} worker completati")
                try:
                    pool.terminate()
                    pool_already_terminated = True
                    pool.close()
                except:
                    pool_already_terminated = True
                    pass  # Ignora errori durante terminate (Windows/Python 3.12)
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
            except (AssertionError, OSError) as e:
                print(f"\n[Errore interno multiprocessing durante recupero risultati - salvando risultati parziali]")
                stop_event.set()
                # Tenta di recuperare risultati parziali per salvare worker_cpu_times
                partial_results = []
                try:
                    if async_res.ready():
                        partial_results = async_res.get(timeout=0.1)
                except:
                    pass
                # Estrai worker_cpu_times dai risultati parziali se disponibili
                if partial_results:
                    worker_cpu_times_partial = []
                    for result in partial_results:
                        if len(result) >= 4:
                            worker_cpu_times_partial.append(result[3])
                    if worker_cpu_times_partial:
                        all_worker_cpu_times.append(worker_cpu_times_partial)
                        print(f"Recuperati tempi CPU da {len(worker_cpu_times_partial)} worker completati")
                try:
                    pool.terminate()
                    pool_already_terminated = True
                    pool.close()
                except:
                    pool_already_terminated = True
                    pass
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
            except MemoryError:
                print(f"\nMemoryError durante recupero risultati batch al livello {level}")
                # Tenta recupero MINIMO worker_cpu_times (operazione leggera)
                try:
                    if async_res.ready():
                        partial_results = async_res.get(timeout=0.1)
                        if partial_results:
                            worker_cpu_times_partial = []
                            for result in partial_results:
                                if len(result) >= 4:
                                    worker_cpu_times_partial.append(result[3])
                            if worker_cpu_times_partial:
                                all_worker_cpu_times.append(worker_cpu_times_partial)
                                print(f"Recuperati tempi CPU da {len(worker_cpu_times_partial)} worker")
                except:
                    pass  # Se fallisce, ignora (memoria critica)
                # NON tentare altre operazioni che richiedono memoria
                try:
                    stop_event.set()
                except:
                    pass
                try:
                    pool.terminate()  # Solo terminate, no close/join
                    pool_already_terminated = True
                except:
                    pass
                # Cleanup aggressivo
                try:
                    del async_res
                    del batches
                    gc.collect()
                except:
                    pass
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                raise MemoryError(f"Memoria esaurita durante recupero risultati al livello {level}")
            except KeyboardInterrupt:
                print(f"\nInterruzione durante recupero risultati batch al livello {level}")
                # Tenta recupero rapido worker_cpu_times
                try:
                    if async_res.ready():
                        partial_results = async_res.get(timeout=0.1)
                        if partial_results:
                            worker_cpu_times_partial = []
                            for result in partial_results:
                                if len(result) >= 4:
                                    worker_cpu_times_partial.append(result[3])
                            if worker_cpu_times_partial:
                                all_worker_cpu_times.append(worker_cpu_times_partial)
                                print(f"Recuperati tempi CPU da {len(worker_cpu_times_partial)} worker")
                except:
                    pass  # Ignora errori per velocità
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                # Skip terminate per velocità - salvataggio immediato
                raise KeyboardInterrupt
            
            # Controllo timeout modulare dopo aver ottenuto i risultati
            timeout_reached, _ = check_timeout(timeout, start_time, margin=0.0)
            if timeout_reached:
                print(f"\nTimeout raggiunto durante elaborazione risultati")
                stop_event.set()  # Segnala ai worker di terminare
                # Skip terminate per terminazione immediata
                # Salva lo stato attuale nei dati di emergenza
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
            
            remaining_time = timeout - (time.time() - start_time) if timeout else float('inf')
            if timeout and remaining_time <= 1.0:
                print(f"\nTimeout imminente prima dell'elaborazione risultati ({remaining_time:.1f}s rimanenti), interruzione immediata")
                stop_event.set()
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
            
            found_mhs_dict = {frozenset(m): True for m, _ in found_mhs}
            new_mhs_count = 0
            
            worker_cpu_times = []
            for result in results:
                if len(result) >= 4:
                    worker_cpu_times.append(result[3])
            
            all_worker_cpu_times.append(worker_cpu_times)
            
            # Elaborazione risultati worker: raccolta figli e MHS
            # MAIN:7 - next ← <> (implicitamente in current_level)
            if skip_global_dedup:
                # Modalità SENZA deduplicazione globale (solo ordinamento)
                print(f"  Elaborazione risultati da {len(results)} batch (costruzione diretta, no deduplica)...")
                current_level = []  # next ← <> (MAIN:7)
                
                for i, (children_tuples, mhs_local, _, worker_cpu) in enumerate(results):
                    if check_interruption(stop_event):
                        print(f"\nInterruzione durante elaborazione risultati ({i+1}/{len(results)} batch)")
                        # Elabora batch rimanenti per salvare MHS
                        for j in range(i, len(results)):
                            _, mhs_batch, _, _ = results[j]
                            for cols, card in mhs_batch:
                                fs = frozenset(cols)
                                if fs not in found_mhs_dict:
                                    found_mhs.append((cols, card))
                                    found_mhs_dict[fs] = True
                                    new_mhs_count += 1
                        print(f"Salvati {new_mhs_count} nuovi MHS dai batch completati")
                        update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                        raise KeyboardInterrupt
                    
                    remaining_time = timeout - (time.time() - start_time) if timeout else float('inf')
                    if timeout and remaining_time <= 1.0:
                        print(f"\nTimeout imminente durante elaborazione MHS ({i+1}/{len(results)} batch, {remaining_time:.1f}s rimanenti)")
                        # Elabora batch rimanenti per salvare MHS
                        for j in range(i, len(results)):
                            _, mhs_batch, _, _ = results[j]
                            for cols, card in mhs_batch:
                                fs = frozenset(cols)
                                if fs not in found_mhs_dict:
                                    found_mhs.append((cols, card))
                                    found_mhs_dict[fs] = True
                                    new_mhs_count += 1
                        print(f"Salvati {new_mhs_count} nuovi MHS dai batch completati")
                        update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                        raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
                    
                    # MAIN:10 - APPEND(Delta, h) per MHS trovati dai worker
                    for cols, card in mhs_local:
                        fs = frozenset(cols)
                        if fs not in found_mhs_dict:
                            found_mhs.append((cols, card))  # APPEND(Delta, h) (MAIN:10)
                            found_mhs_dict[fs] = True
                            new_mhs_count += 1
                    
                    # MAIN:13,19 - APPEND(next, GENERATE_CHILDREN(h))
                    # I figli generati dai worker vengono aggiunti a next (current_level)
                    current_level.extend(children_tuples)  # APPEND/MERGE(next, children) (MAIN:13,19)
                    
                    if len(results) > 10 and should_check_iteration(i, 5, offset=0) and i > 0:
                        print(f"\r  Elaborazione batch: {i}/{len(results)} ({calculate_percentage(i, len(results)):.1f}%)...", end="", flush=True)
                
                print(f"\r  Elaborazione batch completata: {len(results)} batch, {len(current_level)} ipotesi raccolte (no deduplica)", end="")
                
                update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                
                # Ordinamento canonico: garantisce ordine decrescente per BFS
                # MERGE implicitamente mantiene ordine (MAIN:19)
                print(f"\n  Ordinamento di {len(current_level)} ipotesi per ordine canonico succL...")
                sort_start = time.time()
                current_level.sort(key=lambda x: x[0], reverse=True)  # Ordine decrescente bin values
                sort_time = time.time() - sort_start
                print(f"  {format_completion_message('Ordinamento', sort_time, count=len(current_level))}")
                
                stats_per_level[level] = len(current_level)
                
            else:
                # Modalità CON deduplicazione globale
                # NOTA: Non presente in pseudocodice teorico, ma utile per edge-cases parallelizzazione
                print(f"  Elaborazione risultati da {len(results)} batch (deduplicazione incrementale)...")
                
                if dedup_mode == "bitset":
                    max_size = 1 << num_cols
                    seen_bitset = bytearray(max_size)
                    print(f"    Bitset inizializzato per {max_size} posizioni ({max_size//8} bytes)")
                    unique_children = []
                    duplicates_found = 0
                    
                    for i, (children_tuples, mhs_local, _, worker_cpu) in enumerate(results):
                        # Check interruzione e timeout
                        if check_interruption(stop_event):
                            print(f"\nInterruzione durante deduplicazione incrementale bitset ({i+1}/{len(results)} batch)")
                            update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                            raise KeyboardInterrupt
                        
                        remaining_time = timeout - (time.time() - start_time) if timeout else float('inf')
                        if timeout and remaining_time <= 1.0:
                            print(f"\nTimeout durante deduplicazione incrementale bitset ({i+1}/{len(results)} batch, {remaining_time:.1f}s)")
                            update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                            raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
                        
                        # MAIN:10 - APPEND(Delta, h) per MHS trovati
                        for cols, card in mhs_local:
                            fs = frozenset(cols)
                            if fs not in found_mhs_dict:
                                found_mhs.append((cols, card))  # APPEND(Delta, h) (MAIN:10)
                                found_mhs_dict[fs] = True
                                new_mhs_count += 1
                        
                        # Deduplicazione incrementale con bitset (non in pseudocodice)
                        # Rimuove duplicati tra batch per garantire unicità
                        for bin_val, vec, card in children_tuples:
                            if not seen_bitset[bin_val]:
                                seen_bitset[bin_val] = 1
                                unique_children.append((bin_val, vec, card))
                            else:
                                duplicates_found += 1
                        
                        # Progresso
                        if len(results) > 10 and should_check_iteration(i, 5, offset=0) and i > 0:
                            print(f"\r  Dedup incrementale bitset: {i}/{len(results)} ({calculate_percentage(i, len(results)):.1f}%), {len(unique_children)} unici, {duplicates_found} dup...", end="", flush=True)
                    
                    print(f"\r  Deduplicazione incrementale bitset completata: {len(unique_children)} unici, {duplicates_found} duplicati", end="")
                    
                elif dedup_mode == "sorted" or dedup_mode == "distributed":
                    # Per sorted/distributed, raccoglie in lista temporanea poi applica strategia
                    all_children = []
                    
                    for i, (children_tuples, mhs_local, _, worker_cpu) in enumerate(results):
                        # Check interruzione e timeout
                        if check_interruption(stop_event):
                            print(f"\nInterruzione durante raccolta risultati ({i+1}/{len(results)} batch)")
                            update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                            raise KeyboardInterrupt
                        
                        remaining_time = timeout - (time.time() - start_time) if timeout else float('inf')
                        if timeout and remaining_time <= 1.0:
                            print(f"\nTimeout durante raccolta risultati ({i+1}/{len(results)} batch, {remaining_time:.1f}s)")
                            update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                            raise TimeoutError((found_mhs, level, stats_per_level, mhs_per_level))
                        
                        # MAIN:10 - APPEND(Delta, h) per MHS trovati
                        for cols, card in mhs_local:
                            fs = frozenset(cols)
                            if fs not in found_mhs_dict:
                                found_mhs.append((cols, card))  # APPEND(Delta, h) (MAIN:10)
                                found_mhs_dict[fs] = True
                                new_mhs_count += 1
                        
                        # Accumula figli per deduplicazione globale
                        all_children.extend(children_tuples)
                        
                        # Progresso
                        if len(results) > 10 and should_check_iteration(i, 5, offset=0) and i > 0:
                            print(f"\r  Raccolta batch: {i}/{len(results)} ({calculate_percentage(i, len(results)):.1f}%), {len(all_children)} ipotesi...", end="", flush=True)
                    
                    print(f"\r  Raccolta completata: {len(all_children)} ipotesi da {len(results)} batch", end="")
                    
                    update_emergency_data(get_emergency_data(), found_mhs, level, stats_per_level, mhs_per_level, all_worker_cpu_times)
                    
                    print(f"\n  Deduplicazione {dedup_mode} di {len(all_children)} ipotesi...")
                    dedup_start_time = time.time()
                    
                    try:
                        if dedup_mode == "sorted":
                            unique_children, duplicates_found = dedup_sorted(
                                all_children, timeout, start_time, dedup_start_time, memory_limit_percent
                            )
                        else:  # distributed
                            unique_children, duplicates_found = dedup_distributed(
                                all_children, num_processes, timeout, start_time, dedup_start_time, memory_limit_percent
                            )
                        
                        dedup_time = time.time() - dedup_start_time
                        rate = calculate_rate(len(all_children), dedup_time)
                        print(f"  {format_completion_message(f'Deduplicazione {dedup_mode}', dedup_time, count=len(all_children), rate=rate)}")
                        print(f"  Risultato: {len(unique_children)} unici, {duplicates_found} duplicati rimossi")
                        
                    except (KeyboardInterrupt, TimeoutError, MemoryError) as e:
                        print(f"\nErrore durante deduplicazione {dedup_mode}: {type(e).__name__}")
                        raise
                
                # Ordinamento canonico post-deduplicazione
                # MERGE implicitamente mantiene ordine decrescente (MAIN:19)
                print(f"  Ordinamento di {len(unique_children)} ipotesi per ordine canonico succL...")
                sort_start = time.time()
                unique_children.sort(key=lambda x: x[0], reverse=True)  # Ordine decrescente bin values
                sort_time = time.time() - sort_start
                print(f"  {format_completion_message('Ordinamento', sort_time, count=len(unique_children))}")
                
                # MAIN:20 - current ← next
                current_level = unique_children  # current ← next (MAIN:20)
                stats_per_level[level] = len(current_level)
            
            level_elapsed = time.time() - batch_start_time
            mhs_this_level = mhs_per_level.get(level, 0)
            total_mhs = sum(mhs_per_level.values())
            print(f"\n  Livello {level} completato in {level_elapsed:.1f}s: {len(current_level)} ipotesi generate - MHS trovati in questo livello: {mhs_this_level} - MHS totali trovati: {total_mhs}")
            
            max_level_reached = level
            
            # MAIN:21 - until current = <>
            # Se nessun figlio è stato generato, l'esplorazione è completa
            if not current_level:
                print("Nessun figlio generato per il livello successivo, terminazione.")
                break  # until current = <> (MAIN:21)

    finally:
        # NON chiamare close/join se il pool è già stato terminato (evita blocco su Windows)
        if not pool_already_terminated:
            try:
                timeout_reached, _ = check_timeout(timeout, start_time, margin=2.0)
                if timeout_reached:
                    safe_pool_terminate(pool, "timeout finale")
                else:
                    pool.close()
                    pool.join()
            except Exception:
                safe_pool_terminate(pool, "exception cleanup")
        else:
            # Pool già terminato, niente da fare
            pass
    # MAIN:22 - return Delta
    # Prima di restituire i risultati, gli MHS trovati vengono ordinati per
    # cardinalità (numero di colonne) per fornire un output leggibile e
    # coerente con la versione seriale. Non vengono alterati i contenuti degli
    # insiemi, solo l'ordine di presentazione nei file di output.
    found_mhs.sort(key=lambda x: x[1])  # Ordina per cardinalità crescente
    return found_mhs, stats_per_level, mhs_per_level, max_level_reached, all_worker_cpu_times  # return Delta (MAIN:22)

# -------------------------
def main(argv):
    utility.stop_requested = False
    stop_event = multiprocessing.Event()
    listener = threading.Thread(target=input_listener, args=(stop_event,), daemon=True)
    listener.start()
    signal.signal(signal.SIGINT, handle_sigint)

    if len(argv) < 2:
        print(__doc__)
        sys.exit(1)

    in_path = argv[1]
    timeout = None
    reduction = True
    outdir = None
    num_processes = None
    batch_size = None
    memory_limit = 100
    skip_global_dedup = False
    origine = None
    densita = None
    categoria = None

    for arg in argv[2:]:
        if arg.startswith("--timeout="):
            timeout = int(arg.split("=")[1])
        elif arg == "--no-reduction":
            reduction = False
        elif arg.startswith("--outdir="):
            outdir = arg.split("=")[1]
        elif arg.startswith("--processes="):
            num_processes = int(arg.split("=")[1])
        elif arg.startswith("--batch-size="):
            batch_size = int(arg.split("=")[1])
        elif arg == "--memory-monitoring":
            if memory_limit >= 100:
                memory_limit = 95
        elif arg.startswith("--memory-threshold="):
            memory_limit = int(arg.split("=")[1])
        elif arg == "--skip-global-dedup":
            skip_global_dedup = True
        elif arg.startswith("--origine="):
            origine = arg.split("=")[1]
        elif arg.startswith("--densita="):
            densita = float(arg.split("=")[1])
        elif arg.startswith("--categoria="):
            categoria = arg.split("=")[1]
    
    if memory_limit < 100:
        tracemalloc.start()
        print("[ATTENZIONE] tracemalloc attivo: overhead prestazioni previsto per monitoraggio memoria")
    
    start_wall = time.time()
    start_cpu = time.process_time()

    rows, _ = parse_matrix_file(in_path)
    num_rows = len(rows)
    num_cols_original = len(rows[0]) if rows else 0

    if reduction:
        col_vectors, col_map = get_column_vectors(rows)
    else:
        col_vectors, col_map = [], []
        for j in range(num_cols_original):
            v = 0
            for i in range(num_rows):
                if rows[i][j] == 1:
                    v |= (1 << i)
            col_vectors.append(v)
            col_map.append(j)

    num_cols_reduced = len(col_vectors)
    removed_cols = sorted(set(range(num_cols_original)) - set(col_map))
    
    num_cols_non_empty = sum(1 for v in col_vectors if v != 0)

    max_level = max(num_rows, num_cols_non_empty)

    print(format_parsing_message(num_rows, num_cols_original, num_cols_reduced, 
                                 removed_empty=len(removed_cols)))
    if timeout:
        print(f"Timeout impostato: {timeout} secondi")
    if memory_limit < 100:
        print(f"Monitoraggio memoria abilitato: soglia {memory_limit}% (protezione da esaurimento)")

    start_datetime = datetime.now()
    print(f"\nInizio calcolo: {start_datetime.strftime('%H:%M:%S')}")
    if timeout:
        end_datetime = start_datetime + timedelta(seconds=timeout)
        print(f"Fine prevista: {end_datetime.strftime('%H:%M:%S')}")

    found_mhs, stats_per_level, mhs_per_level = [], {}, {}
    completed = False
    algorithm_start = time.time()
    level_interrupted = -1
    error_type = None
    max_level_reached = 0
    worker_cpu_times = []  # Lista tempi CPU dei worker
    
    # Estende il dizionario di emergenza con i dati di input mutabili + immutabili (costanti durante l'esecuzione)
    emergency_data_obj = get_emergency_data()
    initialize_emergency_input_data(emergency_data_obj, num_cols_original, num_rows, 
                                     col_vectors, removed_cols, origine, densita, categoria)
    
    try:
        found_mhs, stats_per_level, mhs_per_level, max_level_reached, worker_cpu_times = mhs_solver_parallel(
            col_vectors, col_map, num_rows, max_level, algorithm_start,
            timeout, batch_size, num_processes, memory_limit, skip_global_dedup
        )
        completed = True
    except KeyboardInterrupt:
        print("\nCalcolo interrotto dall'utente.")
        
        emergency_data_obj = get_emergency_data()
        if 'found_mhs' in emergency_data_obj and emergency_data_obj['found_mhs']:
            found_mhs = emergency_data_obj['found_mhs']
            print(f"Recuperati {len(found_mhs)} MHS parziali dall'interruzione.")
        
        if 'stats' in emergency_data_obj and emergency_data_obj['stats']:
            stats_per_level = emergency_data_obj['stats']
            
        if 'mhs_per_level' in emergency_data_obj and emergency_data_obj['mhs_per_level']:
            mhs_per_level = emergency_data_obj['mhs_per_level']
            
        if 'level' in emergency_data_obj and emergency_data_obj['level'] > 0:
            level_interrupted = emergency_data_obj['level']
            print(f"Recuperato livello di interruzione: {level_interrupted}.")
            
        # Recupera anche i tempi CPU dei worker
        if 'worker_cpu_times' in emergency_data_obj and emergency_data_obj['worker_cpu_times']:
            worker_cpu_times = emergency_data_obj['worker_cpu_times']
            
    except TimeoutError as e:
        print("\nTimeout raggiunto.")
        if len(e.args) > 0 and isinstance(e.args[0], tuple) and len(e.args[0]) >= 1:
            found_mhs = e.args[0][0]
            if len(e.args[0]) >= 2:
                level_passed = e.args[0][1]
                emergency_data_obj = get_emergency_data()
                if 'level' in emergency_data_obj and emergency_data_obj['level'] > level_passed:
                    level_interrupted = emergency_data_obj['level']
                    print(f"Livello corretto: {level_passed} -> {level_interrupted}")
                else:
                    level_interrupted = level_passed
            if len(e.args[0]) >= 3:
                stats_per_level = e.args[0][2]
            if len(e.args[0]) >= 4:
                mhs_per_level = e.args[0][3]
            # Recupera anche i tempi CPU dei worker dai dati di emergenza
            emergency_data_obj = get_emergency_data()
            if 'worker_cpu_times' in emergency_data_obj and emergency_data_obj['worker_cpu_times']:
                worker_cpu_times = emergency_data_obj['worker_cpu_times']
            print(f"Recuperati {len(found_mhs)} MHS parziali dal timeout.")
    except MemoryError:
        print(f"MEMORIA INSUFFICIENTE.")
        print("Recupero risultati parziali dai dati di emergenza...")
        
        error_type = "memory"
        
        emergency_data_obj = get_emergency_data()
        if 'found_mhs' in emergency_data_obj and emergency_data_obj['found_mhs']:
            found_mhs = emergency_data_obj['found_mhs']
            print(f"Recuperati {len(found_mhs)} MHS parziali dall'errore di memoria")
        else:
            print(f"Nessun MHS recuperabile")
        
        if 'stats' in emergency_data_obj and emergency_data_obj['stats']:
            stats_per_level = emergency_data_obj['stats']
            print(f"Recuperate statistiche per {len(stats_per_level)} livelli")
            
        if 'mhs_per_level' in emergency_data_obj and emergency_data_obj['mhs_per_level']:
            mhs_per_level = emergency_data_obj['mhs_per_level']
            print(f"Recuperata distribuzione MHS per livello")
            
        if 'level' in emergency_data_obj and emergency_data_obj['level'] > 0:
            level_interrupted = emergency_data_obj['level']
            print(f"Livello di interruzione: {level_interrupted}")
        
        # Recupera anche i tempi CPU dei worker
        if 'worker_cpu_times' in emergency_data_obj and emergency_data_obj['worker_cpu_times']:
            worker_cpu_times = emergency_data_obj['worker_cpu_times']
        
        print(f"Procedura di recupero completata. Salvataggio in corso...")
        
    emergency_data_obj = get_emergency_data()
    if not completed and 'level' in emergency_data_obj and emergency_data_obj['level'] > 0:
        if level_interrupted == -1:
            level_interrupted = emergency_data_obj['level']
            print(f"Recuperato livello di interruzione: {level_interrupted}")
        elif level_interrupted + 1 == emergency_data_obj['level']:
            print(f"Correzione livello: da {level_interrupted} a {emergency_data_obj['level']}")
            level_interrupted = emergency_data_obj['level']

    input_dir = os.path.dirname(in_path)
    base = os.path.splitext(os.path.basename(in_path))[0]
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        out_path = os.path.join(outdir, base + ".mhs")
    else:
        out_path = os.path.join(input_dir, base + ".mhs") if input_dir else base + ".mhs"

    perf = measure_performance(start_wall, start_cpu, worker_cpu_times)
    write_mhs_output(
        out_path, found_mhs, num_rows, num_cols_original, num_cols_reduced,
        removed_cols, completed, start_wall, level_interrupted=level_interrupted,
        stats_per_level=stats_per_level, perf=perf, mhs_per_level=mhs_per_level,
        origine=origine, densita=densita, categoria=categoria
    )
    print(f"\nOutput scritto in {out_path}")
    elapsed_total = time.time() - algorithm_start
    print(format_summary_message(len(found_mhs), completed, elapsed_total,
                                 level=level_interrupted if not completed else max_level_reached))
    if not completed:
        interruption_reason = "interruzione"
        
        if 'error_type' in locals():
            if error_type == "memory":
                interruption_reason = "memoria insufficiente"
            elif error_type == "timeout":
                interruption_reason = "timeout"
            elif error_type == "user_interrupt":
                interruption_reason = "interruzione utente"
        else:
            if timeout and (time.time() - start_wall >= timeout):
                interruption_reason = "timeout"
            elif utility.stop_requested:
                interruption_reason = "interruzione utente"
            
        print(f"Terminazione avvenuta al livello {level_interrupted} per {interruption_reason}.")

        if 'error_type' in locals() and error_type == "memory":
            print("NOTA: Interruzione per memoria insufficiente\n")
            print("Il programma ha rilevato che l'utilizzo della memoria ha superato la soglia")
            print(f"configurata ({memory_limit}%) e ha terminato l'elaborazione per prevenire")
            print("crash del sistema o esaurimento totale della RAM.")
            print()
            print("I risultati parziali trovati fino al momento dell'interruzione sono stati")
            print(f"salvati correttamente nel file di output: {out_path}")
            print()
            print("Suggerimenti per elaborazioni future:")
            print("  • Aumentare la soglia memoria (--memory-threshold=N, max 99%)")
            print("  • Ridurre il numero di processi paralleli (--processes=N)")
            print("  • Utilizzare una macchina con più RAM disponibile")
            print("  • Suddividere il problema in sottoproblemi più piccoli")
    
    if mhs_per_level and any(count > 0 for count in mhs_per_level.values()):
        print("Distribuzione MHS per livello:")
        for level in sorted(mhs_per_level.keys()):
            if mhs_per_level[level] > 0:
                print(f"  Livello {level}: {mhs_per_level[level]} MHS")

if __name__ == "__main__":
    main(sys.argv)
