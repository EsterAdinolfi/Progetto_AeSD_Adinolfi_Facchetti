import sys
import os
import time
import tracemalloc
import psutil
from typing import List, Tuple, Dict
from collections import deque
import heapq
import platform
import threading
import signal

stop_requested = False
_input_listener_active = False

def input_listener(stop_event=None):
    """
    Thread listener per interruzione manuale dell'utente (tasto 'q' o ESC).
    
    Args:
        stop_event: evento threading per segnalare stop
    """
    global stop_requested, _input_listener_active
    _input_listener_active = True
    print("\n[Premere 'q' o ESC per interrompere l'esecuzione in qualsiasi momento]\n")

    try:
        is_windows = os.name == 'nt'
        
        if is_windows:
            import msvcrt
            while not stop_requested:
                if stop_event is not None and stop_event.is_set():
                    break
                    
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                        if ch == 'q' or ord(ch) == 27:
                            print("\nInterruzione richiesta dall'utente (Q/ESC).")
                            stop_requested = True
                            if stop_event is not None:
                                stop_event.set()
                            break
                    time.sleep(0.02)
                except Exception:
                    break
        else:
            import termios, tty, select
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                while not stop_requested:
                    if stop_event is not None and stop_event.is_set():
                        break
                        
                    try:
                        if select.select([sys.stdin], [], [], 0.02)[0]:
                            ch = sys.stdin.read(1)
                            if ch.lower() == 'q' or ord(ch) == 27:
                                print("\nInterruzione richiesta dall'utente (Q/ESC).")
                                stop_requested = True
                                if stop_event is not None:
                                    stop_event.set()
                                break
                    except Exception:
                        break
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    finally:
        _input_listener_active = False

def handle_sigint(signum, frame):
    """
    Handler per segnale SIGINT (Ctrl+C).
    
    Args:
        signum: numero del segnale
        frame: frame corrente
    """
    global stop_requested
    
    if hasattr(handle_sigint, '_already_called'):
        return
    
    handle_sigint._already_called = True
    stop_requested = True
    
    print("\n[Interruzione richiesta dall'utente (Ctrl+C)]")
    raise KeyboardInterrupt

last_saved_state = None

class Hypothesis:
    """
    Rappresenta un'ipotesi nell'algoritmo MHS.
    
    Attributes:
        bin: rappresentazione binaria (quali colonne sono selezionate)
        vector: bitmask delle righe coperte
        card: cardinalità (numero colonne selezionate)
        num_cols: numero totale colonne
        delta: differenza con padre (opzionale)
    """
    __slots__ = ("bin", "vector", "card", "num_cols", "delta")
    def __init__(self, bin_repr: int, num_cols: int, vector: int = 0):
        self.bin = bin_repr
        self.vector = vector
        self.card = bin_repr.bit_count()
        self.num_cols = num_cols

    def lm1_index(self) -> int:
        """Indice del bit 1 più significativo"""
        if self.bin == 0:
            return -1
        bit_length = self.bin.bit_length()
        return self.num_cols - bit_length

    def __lt__(self, other):
        return self.bin > other.bin

    def __repr__(self):
        return f"Hypothesis(bin={self.bin:0{self.num_cols}b}, card={self.card}, vector={self.vector:b})"


def parse_matrix_file(path: str) -> Tuple[List[List[int]], List[str]]:
    """
    Legge una matrice binaria da file.
    
    Args:
        path: percorso del file .matrix
        
    Returns:
        Tupla (righe, commenti) dove righe è una lista di liste di 0/1
    """
    rows = []
    comments = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith(';;;'):
                    comments.append(line)
                    continue
                
                if line.endswith('-'):
                    line = line[:-1].strip()
                
                clean_line = ''.join(ch for ch in line if ch in '01 \t')
                tokens = clean_line.split()
                if tokens:
                    row = []
                    for t in tokens:
                        if t in ('0', '1'):
                            row.append(int(t))
                    rows.append(row)
    except KeyboardInterrupt:
        print("\n[Interruzione richiesta durante il parsing del file]")
        raise
    
    if rows:
        max_cols = max(len(r) for r in rows)
        for i in range(len(rows)):
            if len(rows[i]) < max_cols:
                rows[i] = rows[i] + [0] * (max_cols - len(rows[i]))
    return rows, comments

def get_column_vectors(rows: List[List[int]]) -> Tuple[List[int], List[int]]:
    """
    Converte la matrice in vettori colonna (bitmask) e rimuove colonne vuote.
    
    Args:
        rows: matrice binaria (lista di liste)
        
    Returns:
        Tupla (col_vectors, col_map) dove:
        - col_vectors: lista di bitmask delle righe per ogni colonna non vuota
        - col_map: mapping da indice ridotto a indice originale
    """
    if not rows:
        return [], []
    num_rows = len(rows)
    num_cols_original = len(rows[0])
    col_vectors = []
    col_map = []
    for j in range(num_cols_original):
        v = 0
        for i in range(num_rows):
            if rows[i][j] == 1:
                v |= (1 << i)
        
        if v != 0:
            col_vectors.append(v)
            col_map.append(j)
    return col_vectors, col_map

def check_is_solution(h: Hypothesis, num_rows: int) -> bool:
    """
    Verifica se un'ipotesi è un Minimal Hitting Set (copre tutte le righe).
    
    Args:
        h: ipotesi da verificare
        num_rows: numero righe matrice
        
    Returns:
        True se h copre tutte le righe
    """
    all_rows_mask = (1 << num_rows) - 1
    return h.vector == all_rows_mask

def set_fields(h: Hypothesis, col_vectors: List[int]):
    """
    Calcola il vector (righe coperte) per un'ipotesi data.
    
    Args:
        h: ipotesi di cui calcolare il vector
        col_vectors: vettori colonne
    """
    v = 0
    for i in range(h.num_cols):
        if (h.bin >> (h.num_cols - 1 - i)) & 1:
            v |= col_vectors[i]
    h.vector = v

def propagate(parent: Hypothesis, child: Hypothesis, added_col_idx: int, col_vectors: List[int]):
    """
    Propaga il vector dal padre al figlio aggiungendo una colonna.
    
    Args:
        parent: ipotesi padre
        child: ipotesi figlio
        added_col_idx: indice della colonna aggiunta
        col_vectors: vettori colonne
    """
    child.vector = parent.vector | col_vectors[added_col_idx]


def get_leftmost_one_position(bin_val, num_cols):
    """Posizione del bit 1 più significativo"""
    if bin_val == 0:
        return -1
    bit_length = bin_val.bit_length()
    return num_cols - bit_length


def generate_succ_left(hypothesis, col_vectors, found_mhs_sets=None, col_map=None, timeout=None, start_time=None):
    """
    Genera successori secondo la strategia succL(h).
    
    IMPORTANTE: succL garantisce che ogni ipotesi sia generata esattamente una volta
    durante l'esplorazione BFS. Questo significa che NON ci possono essere duplicati
    tra i figli generati da questa funzione per la stessa ipotesi padre.
    
    Strategia: per un'ipotesi h con leftmost-1 in posizione k, genera tutti i figli
    ottenuti aggiungendo un bit 1 in posizione j < k (a sinistra del leftmost-1).
    
    Args:
        hypothesis: ipotesi padre
        col_vectors: vettori colonne (bitmask righe)
        found_mhs_sets: set di MHS già trovati (per pruning)
        col_map: mapping indici ridotti -> originali
        timeout: timeout in secondi
        start_time: timestamp inizio
        
    Returns:
        Lista di ipotesi figlie (oggetti Hypothesis)
    """
    children = []
    num_cols = hypothesis.num_cols
    current_vector = hypothesis.vector
    base_bin = hypothesis.bin
    
    leftmost_one_pos = get_leftmost_one_position(base_bin, num_cols)
    
    if leftmost_one_pos == -1:
        for j in range(num_cols):
            if timeout and start_time and (time.time() - start_time) >= timeout:
                raise TimeoutError("Timeout durante generazione singoletti")
            
            new_bin = 1 << (num_cols - 1 - j)
            new_vector = col_vectors[j]
            
            new_h = Hypothesis(new_bin, num_cols)
            new_h.vector = new_vector
            new_h.card = 1
            children.append(new_h)
        return children
    
    for j in range(leftmost_one_pos):
        if timeout and start_time and j % 10 == 0 and (time.time() - start_time) >= timeout:
            raise TimeoutError("Timeout durante generazione successori succL")
        
        if not ((base_bin >> (num_cols - 1 - j)) & 1):
            new_bin = base_bin | (1 << (num_cols - 1 - j))
            
            if found_mhs_sets and col_map:
                child_cols = set()
                for col_idx in range(num_cols):
                    if (new_bin >> (num_cols - 1 - col_idx)) & 1:
                        child_cols.add(col_map[col_idx])
                
                contains_mhs = any(mhs_set.issubset(child_cols) for mhs_set in found_mhs_sets)
                if contains_mhs:
                    continue
            
            new_vector = current_vector | col_vectors[j]
            
            new_h = Hypothesis(new_bin, num_cols)
            new_h.vector = new_vector
            new_h.card = hypothesis.card + 1
            children.append(new_h)
    
    return children

lock = threading.Lock()
def generate_children_threadsafe(h, current, col_vectors, all_hypotheses_keys, timeout=None, start_time=None):
    """
    Genera figli in modo thread-safe (usato in contesti multi-thread).
    
    Args:
        h: ipotesi padre
        current: lista ipotesi correnti del livello
        col_vectors: vettori colonne
        all_hypotheses_keys: set di chiavi già generate (per evitare duplicati)
        timeout: timeout in secondi
        start_time: timestamp inizio
        
    Returns:
        Lista di figli unici (non già presenti in all_hypotheses_keys)
    """
    if timeout is not None and start_time is not None and (time.time() - start_time) >= timeout:
        return []

    current_bins = {c.bin for c in current}
    tmp_children = generate_succ_left(h, current_bins, col_vectors)

    if timeout is not None and start_time is not None and (time.time() - start_time) >= timeout:
        return []

    children = []
    with lock:
        for c in tmp_children:
            if c.bin not in all_hypotheses_keys:
                all_hypotheses_keys.add(c.bin)
                children.append(c)
    return children

def process_batch(args):
    """
    Elabora un batch di ipotesi (usato per parallelizzazione con multiprocessing).
    
    Args:
        args: tupla contenente (batch, current, col_vectors, all_hypotheses_keys_list,
              num_rows, col_map, timeout, start_time, found_mhs_sets)
              
    Returns:
        Tupla (children, mhs_local, new_keys) con figli generati, MHS trovati,
        e nuove chiavi aggiunte
    """
    (batch, current, col_vectors, all_hypotheses_keys_list,
     num_rows, col_map, timeout, start_time, found_mhs_sets) = args

    all_hypotheses_keys = set(all_hypotheses_keys_list)
    current_bins = {c.bin for c in current}
    found_mhs_sets_local = [set(s) for s in found_mhs_sets] if found_mhs_sets else []

    children = []
    mhs_local = []

    for h in batch:
        if timeout is not None and (time.time() - start_time) >= timeout:
            return children, mhs_local, True
        
        h_cols = [col_map[j] for j in range(h.num_cols) if (h.bin >> (h.num_cols - 1 - j)) & 1]
        h_set = set(h_cols)

        if found_mhs_sets_local and any(mhs.issubset(h_set) for mhs in found_mhs_sets_local):
            continue

        if check_is_solution(h, num_rows):
            h_cols.sort()
            mhs_local.append((h_cols, h.card))
            continue

        new_children = generate_succ_left(h, current_bins, col_vectors)

        survivors = []
        for child in new_children:
            if check_is_solution(child, num_rows):
                cols = [col_map[j] for j in range(child.num_cols) if (child.bin >> (child.num_cols - 1 - j)) & 1]
                cols.sort()
                if not any(frozenset(cols) == frozenset(existing) for existing, _ in mhs_local):
                    mhs_local.append((cols, child.card))
            else:
                if child.bin not in all_hypotheses_keys:
                    survivors.append(child)
        children.extend(survivors)

        if timeout is not None and (time.time() - start_time) >= timeout:
            return children, mhs_local, True
    return children, mhs_local, False


def check_timeout(timeout, start_time, margin=0.0):
    """Controlla se timeout raggiunto
    
    Args:
        timeout: timeout in secondi (None = nessun timeout)
        start_time: timestamp di inizio esecuzione
        margin: margine di sicurezza in secondi (default: 0.0)
        
    Returns:
        tuple: (timeout_reached, remaining_time)
            - timeout_reached: True se timeout raggiunto/imminente
            - remaining_time: tempo rimanente in secondi (None se no timeout)
    """
    if timeout is None:
        return False, None
    
    elapsed = time.time() - start_time
    remaining = timeout - elapsed
    
    if remaining <= margin:
        return True, remaining
    
    return False, remaining


def check_interruption(stop_event=None):
    """
    Verifica se è stata richiesta un'interruzione (Ctrl+C o tasto q/ESC).
    
    Args:
        stop_event: evento threading per segnalazione stop
        
    Returns:
        True se interruzione richiesta
    """
    if stop_requested:
        return True
    
    if stop_event is not None and stop_event.is_set():
        return True
    
    return False


def check_memory_with_cleanup(memory_limit_percent=100, force_gc=True):
    """
    Verifica utilizzo memoria e forza garbage collection se necessario.
    
    Args:
        memory_limit_percent: soglia percentuale memoria (default: 100 = disabilitato)
        force_gc: se True, forza garbage collection se soglia superata
        
    Returns:
        Tupla (is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb, cleaned)
    """
    is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb = check_memory_usage()
    cleaned = False
    
    if mem_percent >= memory_limit_percent and force_gc:
        import gc
        gc.collect()
        cleaned = True
        is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb = check_memory_usage()
    
    return is_mem_crit, mem_percent, mem_avail_mb, mem_used_mb, cleaned


def update_progress(iteration, total, last_update_time, update_interval=1.0, 
                    prefix="", extra_info="", timeout=None, start_time=None):
    """
    Aggiorna messaggio di progresso se intervallo superato.
    
    Args:
        iteration: iterazione corrente
        total: totale iterazioni
        last_update_time: timestamp ultimo aggiornamento
        update_interval: intervallo minimo tra aggiornamenti (secondi)
        prefix: prefisso messaggio
        extra_info: informazioni extra da mostrare
        timeout: timeout in secondi
        start_time: timestamp inizio
        
    Returns:
        Tupla (should_print, new_last_update_time, message)
    """
    current_time = time.time()
    
    if current_time - last_update_time < update_interval:
        return False, last_update_time, None
    
    progress_pct = (iteration / total * 100) if total > 0 else 100
    message = f"\r{prefix}{iteration}/{total} ({progress_pct:.1f}%)"
    
    if timeout and start_time:
        elapsed = current_time - start_time
        remaining = timeout - elapsed
        if remaining > 0:
            message += f" - Tempo: {elapsed:.1f}s - Rimanente: {remaining:.1f}s"
        else:
            message += f" - Tempo: {elapsed:.1f}s - TIMEOUT!"
    
    if extra_info:
        message += f" - {extra_info}"
    
    message += "\t\t"
    
    return True, current_time, message


def combined_check(timeout=None, start_time=None, stop_event=None, 
                  memory_limit=100, iteration=0, check_frequency=1,
                  timeout_margin=0.0):
    """
    Controllo combinato di timeout, interruzione e memoria.
    
    Args:
        timeout: timeout in secondi
        start_time: timestamp inizio
        stop_event: evento threading per stop
        memory_limit: soglia percentuale memoria
        iteration: iterazione corrente
        check_frequency: frequenza controlli
        timeout_margin: margine timeout (secondi)
        
    Returns:
        Dict con campi: should_stop, reason, timeout_reached, remaining_time,
        mem_critical, mem_percent, interrupted
    """
    result = {
        'should_stop': False,
        'reason': None,
        'timeout_reached': False,
        'remaining_time': None,
        'mem_critical': False,
        'mem_percent': 0,
        'interrupted': False
    }
    
    if iteration % check_frequency != 0:
        return result
    
    if check_interruption(stop_event):
        result['should_stop'] = True
        result['reason'] = 'interrupt'
        result['interrupted'] = True
        return result
    
    timeout_reached, remaining = check_timeout(timeout, start_time, timeout_margin)
    result['timeout_reached'] = timeout_reached
    result['remaining_time'] = remaining
    
    if timeout_reached:
        result['should_stop'] = True
        result['reason'] = 'timeout'
        return result
    
    if memory_limit < 100 and iteration % (check_frequency * 10) == 0:
        is_crit, mem_pct, _, _, _ = check_memory_with_cleanup(memory_limit, force_gc=False)
        result['mem_critical'] = is_crit
        result['mem_percent'] = mem_pct
        
        if is_crit:
            result['should_stop'] = True
            result['reason'] = 'memory'
            return result
    
    return result


def periodic_check_with_progress(iteration, total, last_update_time, 
                                 timeout=None, start_time=None, stop_event=None,
                                 memory_limit=100, check_frequency=1, 
                                 prefix="", extra_info=""):
    """
    Controllo periodico con aggiornamento progresso (usato nel solver seriale).
    
    Combina controllo di timeout/interruzione/memoria con aggiornamento
    del messaggio di progresso.
    
    Args:
        iteration: iterazione corrente
        total: totale iterazioni
        last_update_time: timestamp ultimo aggiornamento
        timeout: timeout in secondi
        start_time: timestamp inizio
        stop_event: evento threading per stop
        memory_limit: soglia percentuale memoria
        check_frequency: frequenza controlli
        prefix: prefisso messaggio progresso
        extra_info: info extra da mostrare
        
    Returns:
        Tupla (check_result, should_print, new_time, message)
        
    Raises:
        TimeoutError: se timeout raggiunto
        KeyboardInterrupt: se interruzione richiesta
        MemoryError: se memoria critica
    """
    # Esegue controlli di sicurezza
    check_result = combined_check(
        timeout=timeout, 
        start_time=start_time, 
        stop_event=stop_event,
        memory_limit=memory_limit, 
        iteration=iteration, 
        check_frequency=check_frequency
    )
    
    # Se bisogna fermarsi, solleva l'eccezione appropriata
    if check_result['should_stop']:
        if check_result['reason'] == 'timeout':
            raise TimeoutError(f"Timeout raggiunto dopo {time.time() - start_time:.1f}s")
        elif check_result['reason'] == 'interrupt':
            raise KeyboardInterrupt("Interruzione richiesta dall'utente")
        elif check_result['reason'] == 'memory':
            raise MemoryError(f"Memoria critica: {check_result['mem_percent']:.1f}%")
    
    # Aggiorna progresso se necessario
    should_print, new_time, message = update_progress(
        iteration, total, last_update_time, 
        prefix=prefix, extra_info=extra_info,
        timeout=timeout, start_time=start_time
    )
    
    return check_result, should_print, new_time, message


def check_batch_processing_conditions(timeout, start_time, stop_event, 
                                      memory_limit, iteration, 
                                      check_frequency=10, margin=0.05):
    """
    Controlli ottimizzati per batch processing nel solver parallelo.
    
    Controlla timeout, interruzioni e memoria con frequenza configurabile.
    Ottimizzato per ridurre l'overhead nei worker.
    
    Args:
        timeout: timeout in secondi (None = nessun timeout)
        start_time: timestamp di inizio
        stop_event: Event per interruzioni
        memory_limit: soglia percentuale memoria
        iteration: iterazione corrente
        check_frequency: controlla ogni N iterazioni
        margin: margine di sicurezza per timeout (secondi)
        
    Returns:
        tuple: (should_stop, reason, details)
            - should_stop: True se bisogna fermarsi
            - reason: 'timeout', 'interrupt', 'memory', None
            - details: dict con informazioni aggiuntive
    """
    # Controlla solo ogni N iterazioni per ridurre overhead
    if iteration % check_frequency != 0:
        return False, None, {}
    
    if check_interruption(stop_event):
        return True, 'interrupt', {}
    
    if timeout:
        timeout_reached, remaining = check_timeout(timeout, start_time, margin)
        if timeout_reached:
            return True, 'timeout', {'remaining': remaining}
    
    if memory_limit < 100 and iteration % (check_frequency * 10) == 0:
        import gc
        is_crit, mem_pct, mem_avail, _, _ = check_memory_with_cleanup(memory_limit, force_gc=False)
        if is_crit:
            gc.collect()
            is_crit, mem_pct, mem_avail, _, _ = check_memory_with_cleanup(memory_limit, force_gc=False)
            if is_crit:
                return True, 'memory', {'mem_percent': mem_pct, 'mem_avail_mb': mem_avail}
    
    return False, None, {}


def check_all_loop_conditions(timeout, start_time, stop_event, memory_limit_percent,
                               current_time, last_timeout_check, last_memory_check,
                               timeout_check_interval=0.2, memory_check_interval=0.5):
    """
    Controllo completo condizioni loop con gestione intervalli separati.
    
    Ottimizzato per loop intensivi: timeout e memoria controllati con
    intervalli configurabili indipendenti.
    
    Args:
        timeout: timeout in secondi
        start_time: timestamp inizio
        stop_event: evento threading per stop
        memory_limit_percent: soglia percentuale memoria
        current_time: timestamp corrente
        last_timeout_check: timestamp ultimo controllo timeout
        last_memory_check: timestamp ultimo controllo memoria
        timeout_check_interval: intervallo controllo timeout (secondi)
        memory_check_interval: intervallo controllo memoria (secondi)
        
    Returns:
        Dict con campi: should_stop, reason, timeout_reached, remaining_time,
        memory_critical, mem_percent, mem_avail_mb, needs_timeout_check,
        needs_memory_check, last_timeout_check, last_memory_check
    """
    result = {
        'should_stop': False,
        'reason': None,
        'timeout_reached': False,
        'remaining_time': None,
        'memory_critical': False,
        'mem_percent': None,
        'mem_avail_mb': None,
        'needs_timeout_check': False,
        'needs_memory_check': False,
        'last_timeout_check': last_timeout_check,
        'last_memory_check': last_memory_check
    }
    
    if check_interruption(stop_event):
        result['should_stop'] = True
        result['reason'] = 'interrupt'
        return result
    
    if timeout and (current_time - last_timeout_check >= timeout_check_interval):
        result['needs_timeout_check'] = True
        result['last_timeout_check'] = current_time
        
        timeout_reached, remaining = check_timeout(timeout, start_time, margin=1.0)
        result['timeout_reached'] = timeout_reached
        result['remaining_time'] = remaining
        
        if timeout_reached:
            result['should_stop'] = True
            result['reason'] = 'timeout'
            return result
    
    if current_time - last_memory_check >= memory_check_interval:
        result['needs_memory_check'] = True
        result['last_memory_check'] = current_time
        
        is_mem_crit, mem_percent, mem_avail_mb, _, cleaned = check_memory_with_cleanup(
            memory_limit_percent, force_gc=True
        )
        result['mem_percent'] = mem_percent
        result['mem_avail_mb'] = mem_avail_mb
        result['memory_critical'] = mem_percent >= memory_limit_percent
        
        if result['memory_critical']:
            result['should_stop'] = True
            result['reason'] = 'memory'
            return result
    
    return result


def check_intensive_loop_conditions(timeout, start_time, memory_limit_percent,
                                    current_time, last_memory_check,
                                    memory_check_interval=2.0):
    """
    Controllo ottimizzato per loop molto intensivi (es. deduplicazione).
    
    Riduce overhead controllando memoria solo periodicamente mentre
    timeout e interruzione sono controllati sempre.
    
    Args:
        timeout: timeout in secondi
        start_time: timestamp inizio
        memory_limit_percent: soglia percentuale memoria
        current_time: timestamp corrente
        last_memory_check: timestamp ultimo controllo memoria
        memory_check_interval: intervallo controllo memoria (secondi)
        
    Returns:
        Dict con campi: should_stop, reason, remaining_time, mem_percent,
        last_memory_check
    """
    result = {
        'should_stop': False,
        'reason': None,
        'remaining_time': None,
        'mem_percent': None,
        'last_memory_check': last_memory_check
    }
    
    if check_interruption():
        result['should_stop'] = True
        result['reason'] = 'interrupt'
        return result
    
    if timeout:
        timeout_reached, remaining = check_timeout(timeout, start_time, margin=0.5)
        if timeout_reached:
            result['should_stop'] = True
            result['reason'] = 'timeout'
            result['remaining_time'] = remaining
            return result
    
    if current_time - last_memory_check >= memory_check_interval:
        result['last_memory_check'] = current_time
        
        is_mem_crit, mem_percent, mem_avail_mb, _, cleaned = check_memory_with_cleanup(
            memory_limit_percent, force_gc=True
        )
        result['mem_percent'] = mem_percent
        
        if mem_percent >= memory_limit_percent:
            result['should_stop'] = True
            result['reason'] = 'memory'
            return result
    
    return result


def format_dedup_progress(processed, total, start_time, strategy, extra=""):
    if total == 0:
        return f"    Deduplicazione {strategy}: completato"
    
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    remaining = (total - processed) / rate if rate > 0 else 0
    pct = processed / total * 100
    
    msg = f"\r    Deduplicazione {strategy}: {processed}/{total} ({pct:.1f}%) - {rate:.0f} ipotesi/s - {remaining:.1f}s rimanenti"
    
    if extra:
        msg += f" - {extra}"
    
    msg += "\t"
    return msg

def format_timeout_message(context, processed=None, total=None, remaining_time=None, elapsed=None):
    msg = f"\nTimeout"
    
    if remaining_time is not None:
        if remaining_time > 0:
            msg += f" imminente ({remaining_time:.1f}s rimanenti)"
        else:
            msg += " raggiunto"
    
    msg += f" durante {context}"
    
    if processed is not None and total is not None:
        msg += f" (processati {processed}/{total})"
    
    if elapsed is not None:
        msg += f" - Tempo trascorso: {elapsed:.1f}s"
    
    return msg


def format_memory_message(context, mem_percent, mem_avail_mb=None, after_cleanup=False):
    status = "insufficiente dopo pulizia" if after_cleanup else "critica"
    msg = f"\nMemoria {status} durante {context}: {mem_percent:.1f}%"
    
    if mem_avail_mb is not None:
        msg += f" (solo {mem_avail_mb:.0f}MB liberi)"
    
    return msg


def format_interrupt_message(context, processed=None, total=None, elapsed=None):
    msg = f"\nInterruzione richiesta durante {context}"
    
    if processed is not None and total is not None:
        msg += f" ({processed}/{total}"
        if elapsed is not None:
            msg += f" in {elapsed:.2f}s"
        msg += ")"
    
    return msg


def format_level_summary(level, processed, elapsed, mhs_count, total_mhs):
    return (f"\rLivello {level} completato: {processed} ipotesi elaborate "
            f"- Tempo: {elapsed:.1f}s - MHS livello: {mhs_count} - MHS totali: {total_mhs}")


def format_filter_progress(current, total, elapsed, eliminated, timeout_remaining=None):
    pct = (current / total * 100) if total > 0 else 100
    rate = current / elapsed if elapsed > 0 else 0
    remaining_est = (total - current) / rate if rate > 0 else 0
    
    msg = (f"\r    Filtro dominazione: {current}/{total} ({pct:.1f}%). "
           f"Tempo filtro: {elapsed:.1f}s - stima completamento: {remaining_est:.1f}s")
    
    if timeout_remaining is not None and timeout_remaining < 60:
        msg += f", Timeout globale: {timeout_remaining:.1f}s"
    
    return msg + "   "


def format_batch_creation_progress(current, total, elapsed, timeout_remaining=None):
    """
    Formatta progresso della creazione batch.
    
    Args:
        current: batch corrente
        total: totale batch
        elapsed: tempo trascorso
        timeout_remaining: tempo rimanente (opzionale)
        
    Returns:
        str: messaggio formattato
    """
    pct = (current / total * 100) if total > 0 else 100
    rate = current / elapsed if elapsed > 0 else 0
    remaining_est = (total - current) / rate if rate > 0 else 0
    
    msg = f"\r    Creazione batch: {current}/{total} ({pct:.1f}%) - Tempo: {elapsed:.1f}s, stima: {remaining_est:.1f}s"
    
    if timeout_remaining is not None and timeout_remaining < 60:
        msg += f" | Timeout: {timeout_remaining:.1f}s"
    
    return msg


# =============================================================================
# FUNZIONI PER SALVATAGGIO STATO DI EMERGENZA
# =============================================================================
# Centralizzano la logica di salvataggio/recupero stato in caso di interruzioni
# =============================================================================

def create_emergency_state(found_mhs, level, stats_per_level, mhs_per_level):
    """
    Crea una tupla di stato di emergenza standardizzata.
    
    Args:
        found_mhs: lista di MHS trovati
        level: livello corrente
        stats_per_level: statistiche per livello
        mhs_per_level: MHS per livello
        
    Returns:
        tuple: (found_mhs, level, stats_per_level, mhs_per_level)
    """
    return (found_mhs, level, stats_per_level, mhs_per_level)


def update_emergency_data(emergency_data_dict, found_mhs, level, stats_per_level, mhs_per_level, worker_cpu_times=None):
    """
    Aggiorna il dizionario di emergency data con lo stato corrente.
    
    Args:
        emergency_data_dict: dizionario da aggiornare
        found_mhs: lista di MHS trovati
        level: livello corrente
        stats_per_level: statistiche per livello
        mhs_per_level: MHS per livello
        worker_cpu_times: lista di tempi CPU dei worker (opzionale)
    """
    emergency_data_dict['found_mhs'] = found_mhs[:]  # Copia lista
    emergency_data_dict['level'] = level
    emergency_data_dict['stats'] = stats_per_level.copy()
    emergency_data_dict['mhs_per_level'] = mhs_per_level.copy()
    if worker_cpu_times is not None:
        # Copia profonda della struttura (lista di liste)
        emergency_data_dict['worker_cpu_times'] = [level_times[:] for level_times in worker_cpu_times]


def initialize_emergency_input_data(emergency_data_dict, num_cols_original, num_rows, 
                                    col_vectors, removed_cols, origine=None, densita=None, categoria=None):
    """
    Inizializza i dati di input nel dizionario emergency (immutabili durante l'esecuzione).
    
    Args:
        emergency_data_dict: dizionario da inizializzare
        num_cols_original: numero colonne originali
        num_rows: numero righe
        col_vectors: vettori colonne
        removed_cols: colonne vuote rimosse
        origine: origine matrice (opzionale)
        densita: densità matrice (opzionale)
        categoria: categoria matrice (opzionale)
    """
    emergency_data_dict['num_cols_original'] = num_cols_original
    emergency_data_dict['num_rows'] = num_rows
    emergency_data_dict['col_vectors'] = col_vectors
    emergency_data_dict['removed_cols'] = removed_cols
    emergency_data_dict['origine'] = origine
    emergency_data_dict['densita'] = densita
    emergency_data_dict['categoria'] = categoria


def extract_state_from_exception(exception, default_state):
    """
    Estrae lo stato salvato da un'eccezione (se presente).
    
    Args:
        exception: eccezione da cui estrarre lo stato
        default_state: stato di default se non trovato nell'eccezione
        
    Returns:
        Tupla (found_mhs, level, stats_per_level, mhs_per_level) se presente,
        altrimenti default_state
    """
    if hasattr(exception, 'args') and len(exception.args) > 0:
        if isinstance(exception.args[0], tuple) and len(exception.args[0]) >= 4:
            return exception.args[0]
    
    return default_state


def create_state_exception(exception_type, found_mhs, level, stats_per_level, mhs_per_level):
    """
    Crea un'eccezione che contiene lo stato corrente dell'algoritmo.
    
    Usato per propagare lo stato quando timeout/interruzione/memoria critica
    causano l'interruzione dell'algoritmo.
    
    Args:
        exception_type: tipo di eccezione (TimeoutError, KeyboardInterrupt, MemoryError)
        found_mhs: MHS trovati
        level: livello corrente
        stats_per_level: statistiche per livello
        mhs_per_level: MHS per livello
        
    Returns:
        Eccezione con lo stato embedded negli args
    """
    state = create_emergency_state(found_mhs, level, stats_per_level, mhs_per_level)
    exc = exception_type(state)
    exc.args = (state,)
    return exc


def signal_stop_and_save(stop_event, emergency_data_obj, found_mhs, level, 
                         stats_per_level, mhs_per_level):
    """
    Segnala stop e salva stato corrente.
    
    Args:
        stop_event: evento threading da settare
        emergency_data_obj: dizionario emergency da aggiornare
        found_mhs: MHS trovati
        level: livello corrente
        stats_per_level: statistiche per livello
        mhs_per_level: MHS per livello
    """
    if stop_event:
        stop_event.set()
    update_emergency_data(emergency_data_obj, found_mhs, level, stats_per_level, mhs_per_level)

def safe_pool_terminate(pool, context="operazione"):
    """
    Termina un pool di worker in modo sicuro gestendo eccezioni di Windows/Python 3.12.
    
    Args:
        pool: multiprocessing.Pool da terminare
        context: descrizione contesto per log (opzionale)
        
    Returns:
        True se terminato con successo, False se errore
    """
    try:
        pool.terminate()
        pool.close()
        return True
    except (ValueError, OSError):
        # Su Windows con Python 3.12, pool.terminate() può sollevare
        # "ValueError: concurrent send_bytes() calls are not supported"
        # È un bug noto del multiprocessing - ignoriamo silenziosamente
        return False
    except Exception:
        # Altri errori imprevisti - messaggio minimo senza traceback
        # L'utente non deve vedere dettagli interni del multiprocessing
        return False


def should_check_iteration(iteration, frequency, offset=0):
    """
    Verifica se è il momento di eseguire un controllo periodico.
    
    Maschera il pattern ripetitivo `if (i - offset) % frequency == 0:` 
    rendendo il codice più leggibile e dichiarativo.
    
    Args:
        iteration (int): Numero di iterazione corrente
        frequency (int): Frequenza del controllo (ogni quante iterazioni)
        offset (int): Offset da sottrarre all'iterazione (default: 0)
        
    Returns:
        bool: True se è il momento di controllare, False altrimenti
        
    Examples:
        >>> # Controllo ogni 100 iterazioni
        >>> if should_check_iteration(i, 100):
        >>>     check_memory()
        
        >>> # Controllo ogni 2 iterazioni con offset
        >>> if should_check_iteration(i, 2, offset=idx):
        >>>     check_timeout()
    """
    return (iteration - offset) % frequency == 0


def save_and_raise(exception_type, context, found_mhs, level, stats_per_level, mhs_per_level, 
                   message=None, last_saved_state_ref=None):
    """
    Salva lo stato e solleva un'eccezione con messaggio formattato.
    
    Funzione di convenienza che unifica salvataggio stato + formattazione messaggio + raise.
    
    Args:
        exception_type: tipo di eccezione (TimeoutError, KeyboardInterrupt, MemoryError)
        context: contesto dell'operazione
        found_mhs: lista di MHS trovati
        level: livello corrente
        stats_per_level: statistiche per livello
        mhs_per_level: MHS per livello
        message: messaggio personalizzato (opzionale, usa template se None)
        last_saved_state_ref: riferimento a variabile globale last_saved_state (opzionale)
        
    Raises:
        exception_type: con stato embedded
    """
    state = create_emergency_state(found_mhs, level, stats_per_level, mhs_per_level)
    
    # Aggiorna last_saved_state se fornito
    if last_saved_state_ref is not None:
        globals()['last_saved_state'] = state
    
    # Usa messaggio personalizzato o template
    if message is None:
        if exception_type == TimeoutError:
            message = f"Timeout durante {context}"
        elif exception_type == KeyboardInterrupt:
            message = f"Interruzione durante {context}"
        elif exception_type == MemoryError:
            message = f"Memoria insufficiente durante {context}"
        else:
            message = f"Errore durante {context}"
    
    # Crea e solleva eccezione
    exc = exception_type(state)
    exc.args = (state,)
    raise exc


# =============================================================================
# OUTPUT E MISURAZIONE PERFORMANCE
# =============================================================================
def write_mhs_output(out_path: str, found_mhs: List[Tuple[List[int], int]],
                     n_rows: int, m_original: int, m_reduced: int,
                     removed_cols: List[int], completed: bool,
                     start_time: float, level_interrupted: int=-1, timeout: int=None,
                     stats_per_level: dict=None, perf: dict=None, mhs_per_level: dict=None,
                     solver_name: str="mhs_solver_parallel.py",
                     origine: str=None, densita: float=None, categoria: str=None):
    """
    Scrive i risultati MHS in un file di output con formato standardizzato.
    
    Il file contiene:
    - commenti con metadati (;;;)
    - statistiche dell'esecuzione
    - MHS trovati come vettori binari (uno per riga)
    
    Args:
        out_path: percorso del file di output
        found_mhs: lista di tuple (indici_colonne, cardinalità)
        n_rows, m_original, m_reduced: dimensioni della matrice
        removed_cols: indici delle colonne vuote rimosse
        completed: True se l'algoritmo è terminato completamente
        start_time: timestamp di inizio esecuzione
        level_interrupted: livello di interruzione (-1 se completato)
        timeout: timeout impostato (opzionale)
        stats_per_level: statistiche per livello
        perf: dizionario con metriche di performance
        mhs_per_level: numero di MHS trovati per livello
        solver_name: nome del solver utilizzato (default: mhs_solver_parallel.py)
        origine: origine del file (benchmarks1/benchmarks2, opzionale)
        densita: densità della matrice (opzionale)
        categoria: categoria della matrice (trivial/tiny/small/medium/large/xlarge, opzionale)
    """
    elapsed = time.time() - start_time
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write(f";;; MHS generati dal solver {solver_name}\n;;;\n")
        out.write(";;; Matrice di input:\n")
        out.write(f";;; |N| (righe) = {n_rows}\n")
        out.write(f";;; |M| (colonne) = {m_original}\n")
        if origine:
            out.write(f";;; Origine: {origine}\n")
        if densita is not None:
            out.write(f";;; Densita: {densita:.4f}\n")
        if categoria:
            out.write(f";;; Categoria: {categoria}\n")
        out.write(f";;; Matrice ridotta |M'| (colonne non vuote) = {m_reduced}\n")
        if removed_cols:
            out.write(f";;; Indici delle colonne vuote rimosse (a partire da 0): {removed_cols}\n")
        out.write(f";;; Livello massimo di esplorazione = {max(n_rows, m_reduced)}\n;;;\n")
        out.write(f";;; Numero di MHS trovati = {len(found_mhs)}\n")
        if found_mhs:
            sizes = [len(cols) for cols, _ in found_mhs]
            out.write(f";;; Cardinalita' minima = {min(sizes)}, Cardinalita' massima = {max(sizes)}\n")
        else:
            out.write(";;; Cardinalita' minima = -, Cardinalita' massima = -\n")
        out.write(f";;; Completato? {completed}\n")
        if not completed and level_interrupted >= -1:
            out.write(f";;; Interruzione al livello = {level_interrupted}\n")
        if timeout:
            out.write(f";;; Timeout imposto = {timeout} secondi\n")
        out.write(";;;\n")

        if stats_per_level:
            out.write(";;; Numero ipotesi generate per livello:\n")
            for lvl, count in stats_per_level.items():
                out.write(f";;;   Livello {lvl}: {count}\n")

        if mhs_per_level:
            out.write(";;;\n;;; MHS trovati per livello:\n")
            total_mhs = 0
            for lvl in sorted(mhs_per_level.keys()):
                count = mhs_per_level[lvl]
                total_mhs += count
                out.write(f";;;   Livello {lvl}: {count} MHS\n")
            out.write(f";;;   Totale: {total_mhs} MHS\n")

        if perf:
            out.write(";;;\n;;; Prestazioni:\n")
            out.write(f";;;   Tempo reale = {perf['tempo_reale']:.4f} s\n")
            out.write(f";;;   CPU time totale = {perf['tempo_cpu']:.4f} s\n")
            
            # Dettagli tempi CPU per solver parallelo
            if perf.get('num_worker', 0) > 0:  # Scrivi se ci sono worker (anche se tempo è 0)
                out.write(f";;;     CPU time master = {perf['tempo_cpu_master']:.4f} s\n")
                out.write(f";;;     CPU time worker (totale) = {perf['tempo_cpu_worker_totale']:.4f} s\n")
                out.write(f";;;     CPU time worker (max) = {perf['tempo_cpu_worker_max']:.4f} s\n")
                out.write(f";;;     CPU time worker (media) = {perf['tempo_cpu_worker_media']:.4f} s\n")
                out.write(f";;;     Numero worker = {perf['num_worker']}\n")
                
                # Formatta worker times per livello
                worker_times_per_level = perf['worker_cpu_times_list']
                if worker_times_per_level and isinstance(worker_times_per_level, list):
                    # Check se è lista di liste (per livello) o lista piatta (vecchio formato)
                    if worker_times_per_level and isinstance(worker_times_per_level[0], list):
                        # Nuovo formato: lista di liste per livello
                        out.write(f";;;     CPU time singoli worker per livello:\n")
                        for level_idx, level_times in enumerate(worker_times_per_level):
                            if level_times:  # Solo se ci sono worker in questo livello
                                times_str = ", ".join(f"{t:.4f}" for t in level_times)
                                out.write(f";;;       Livello {level_idx}: [{times_str}] s\n")
                    else:
                        # Vecchio formato: lista piatta (backward compatibility)
                        out.write(f";;;     CPU time singoli worker = [\n")
                        for i in range(0, len(worker_times_per_level), 10):
                            chunk = worker_times_per_level[i:i+10]
                            chunk_str = ", ".join(f"{t:.4f}" for t in chunk)
                            out.write(f";;;       {chunk_str}")
                            if i + 10 < len(worker_times_per_level):
                                out.write(",\n")
                            else:
                                out.write("\n")
                        out.write(f";;;     ] s\n")
            else:
                # Versione seriale: nessun worker
                out.write(f";;;     CPU time singoli worker: non presente (esecuzione seriale)\n")
            
            out.write(f";;;   Memoria RSS = {perf['mem_rss']/1024:.1f} KB\n")
            if perf['mem_picco'] is not None:
                out.write(f";;;   Picco memoria = {perf['mem_picco']/1024:.1f} KB\n")
            else:
                out.write(f";;;   Picco memoria = non rilevato (richiede --memory-monitoring)\n")

        out.write(";;;\n")
        for cols, _card in found_mhs:
            vect = ['0'] * m_original
            for orig_idx in cols:
                vect[orig_idx] = '1'
            out.write(' '.join(vect) + "\n")

def measure_performance(start_wall, start_cpu, worker_cpu_times=None):
    """
    Misura le metriche di performance dell'esecuzione.
    
    Args:
        start_wall: timestamp iniziale (time.time())
        start_cpu: CPU time iniziale (time.process_time()) - tempo CPU del processo master
        worker_cpu_times: lista opzionale di tempi CPU dei worker (per solver parallelo)
        
    Returns:
        Dizionario con tempo_reale, tempo_cpu, tempo_cpu_master, tempo_cpu_worker_totale,
        tempo_cpu_worker_max, mem_rss, mem_picco (in byte)
    """
    elapsed_wall = time.time() - start_wall
    elapsed_cpu_master = time.process_time() - start_cpu
    process = psutil.Process(os.getpid())
    mem_rss = process.memory_info().rss
    
    # tracemalloc potrebbe non essere attivo (se --memory-monitoring non è stato specificato)
    # In quel caso non riportiamo il picco di memoria
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
    else:
        # Senza tracemalloc, non abbiamo un picco affidabile
        peak = None
    
    # Calcolo tempi CPU per solver parallelo
    if worker_cpu_times and len(worker_cpu_times) > 0:
        # worker_cpu_times è una lista di liste (una per livello)
        # Appiattisci la struttura per ottenere tutti i tempi worker
        if isinstance(worker_cpu_times[0], list):
            # Nuova struttura: lista di liste per livello
            all_worker_times = [t for level_times in worker_cpu_times for t in level_times]
        else:
            # Vecchia struttura: lista piatta (retrocompatibilità)
            all_worker_times = worker_cpu_times
        
        if all_worker_times:
            worker_cpu_total = sum(all_worker_times)
            worker_cpu_max = max(all_worker_times)
            worker_cpu_media = sum(all_worker_times) / len(all_worker_times)
            num_worker = len(all_worker_times)
        else:
            worker_cpu_total = 0
            worker_cpu_max = 0
            worker_cpu_media = 0
            num_worker = 0
        
        # CPU time effettivo = master + totale worker
        # (rappresenta il tempo CPU totale utilizzato da tutti i processi)
        tempo_cpu_effettivo = elapsed_cpu_master + worker_cpu_total
    else:
        # Solver seriale: solo il tempo CPU del master
        worker_cpu_total = 0
        worker_cpu_max = 0
        worker_cpu_media = 0
        num_worker = 0
        tempo_cpu_effettivo = elapsed_cpu_master
    
    return {
        "tempo_reale": elapsed_wall,
        "tempo_cpu": tempo_cpu_effettivo,  # Tempo CPU totale (master + worker)
        "tempo_cpu_master": elapsed_cpu_master,  # Solo master
        "tempo_cpu_worker_totale": worker_cpu_total,  # Somma di tutti i worker
        "tempo_cpu_worker_max": worker_cpu_max,  # Max tra i worker
        "tempo_cpu_worker_media": worker_cpu_media,  # Media dei worker
        "num_worker": num_worker,  # Numero di worker
        "worker_cpu_times_list": worker_cpu_times if worker_cpu_times else [],  # LISTA COMPLETA dei tempi
        "mem_rss": mem_rss,
        "mem_picco": peak
    }

def check_memory_usage(threshold_percent=90):
    """
    Controlla l'utilizzo della memoria di sistema.
    
    Args:
        threshold_percent: soglia percentuale per considerare la memoria critica
        
    Returns:
        Tupla (is_critical, percent_used, available_mb, used_mb)
    """
    try:
        memory = psutil.virtual_memory()
        percent_used = memory.percent
        available_mb = memory.available / (1024 * 1024)
        used_mb = memory.used / (1024 * 1024)
        is_critical = percent_used > threshold_percent
        return is_critical, percent_used, available_mb, used_mb
    except Exception:
        print(f"Errore nel controllo memoria.")
        return False, 0, 0, 0

def is_small_matrix(rows):
    """
    Determina se una matrice è "piccola" e quindi adatta al solver seriale.
    
    Usa gli stessi criteri di categorizzazione definiti in matrices_selection.py
    per garantire coerenza con il sistema di classificazione delle matrici.
    
    Una matrice viene considerata piccola (e quindi eseguita in modalità seriale) se:
    - È trivial: N≤1 o M'≤1 (soluzioni banali, istantanee)
    - È tiny: N≤3 e M'≤15 (< 1 secondo)
    - È small: N≤5 e M'≤30 (< 10 secondi)
    
    Per matrici più grandi (medium, large, xlarge) viene usato il solver parallelo.
    
    Args:
        rows: matrice binaria (lista di liste)
        
    Returns:
        True se la matrice è piccola (seriale), False se è grande (parallelo)
    """
    if not rows:
        return True
    
    num_rows = len(rows)
    num_cols_original = len(rows[0]) if rows else 0
    
    # Conta colonne non vuote (M_ridotto)
    non_empty_cols = 0
    for j in range(num_cols_original):
        if any(rows[i][j] == 1 for i in range(num_rows)):
            non_empty_cols += 1
    
    # Criteri allineati con matrices_selection.py:
    
    # 1. Trivial: banali (1 riga o 1 colonna ridotta)
    if num_rows <= 1 or non_empty_cols <= 1:
        return True
    
    # 2. Tiny: molto piccole (completabili in < 1 secondo)
    if num_rows <= 3 and non_empty_cols <= 15:
        return True
    
    # 3. Small: piccole (completabili in < 10 secondi)
    if num_rows <= 5 and non_empty_cols <= 30:
        return True
    
    # Tutte le altre (medium, large, xlarge) usano parallelo
    return False


# =============================================================================
# FUNZIONI DI CALCOLO 
# =============================================================================

def calculate_percentage(current, total):
    """
    Calcola e formatta la percentuale con 1 decimale.
    
    Args:
        current: valore corrente
        total: valore totale
        
    Returns:
        float: percentuale (0.0-100.0)
    """
    if total <= 0:
        return 100.0
    return round((current / total) * 100.0, 1)


def calculate_rate(items, elapsed_time):
    """
    Calcola il rate di processamento (items/secondo).
    
    Args:
        items: numero di elementi processati
        elapsed_time: tempo trascorso in secondi
        
    Returns:
        float: rate (items/sec), 0.0 se elapsed_time <= 0
    """
    if elapsed_time <= 0:
        return 0.0
    return items / elapsed_time


def calculate_eta(remaining_items, rate):
    """
    Calcola il tempo stimato rimanente (ETA).
    
    Args:
        remaining_items: elementi rimanenti da processare
        rate: rate di processamento (items/sec)
        
    Returns:
        float: tempo stimato in secondi, None se rate <= 0
    """
    if rate <= 0:
        return None
    return remaining_items / rate


def format_time(seconds):
    """
    Formatta i secondi in formato leggibile.
    
    Args:
        seconds: tempo in secondi
        
    Returns:
        str: tempo formattato (es. "1h 23m", "45m 30s", "12.5s")
    """
    if seconds is None:
        return "N/A"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds / 3600)
        mins = int((seconds % 3600) / 60)
        return f"{hours}h {mins}m"


# =============================================================================
# FUNZIONI MESSAGGI INFORMATIVI 
# =============================================================================

def format_parsing_message(num_rows, num_cols_original, num_cols_reduced, 
                           removed_empty=0, removed_duplicates=0):
    """
    Formatta messaggio di parsing input.
    
    Args:
        num_rows: numero di righe (N)
        num_cols_original: numero di colonne originali (M)
        num_cols_reduced: numero di colonne dopo riduzione (M')
        removed_empty: colonne vuote rimosse
        removed_duplicates: colonne duplicate rimosse
        
    Returns:
        str: messaggio formattato
    """
    msg = f"Parsing input: |N|={num_rows}, |M|={num_cols_original}"
    
    if num_cols_reduced < num_cols_original:
        msg += f" → |M'|={num_cols_reduced}"
        if removed_empty > 0:
            msg += f" (vuote: {removed_empty}"
        if removed_duplicates > 0:
            if removed_empty > 0:
                msg += f", duplicate: {removed_duplicates})"
            else:
                msg += f" (duplicate: {removed_duplicates})"
        elif removed_empty > 0:
            msg += ")"
    
    return msg


def format_level_start_message(level, count, detail="ipotesi da analizzare"):
    """
    Formatta messaggio di inizio livello.
    
    Args:
        level: numero del livello
        count: numero di elementi
        detail: descrizione (default: "ipotesi da analizzare")
        
    Returns:
        str: messaggio formattato
    """
    return f"Livello {level}: {count} {detail}"


def format_completion_message(operation, elapsed_time, count=None, rate=None):
    """
    Formatta messaggio di completamento operazione.
    
    Args:
        operation: nome operazione (es. "Ordinamento", "Deduplicazione")
        elapsed_time: tempo trascorso in secondi
        count: numero elementi processati (opzionale)
        rate: rate di processamento (opzionale)
        
    Returns:
        str: messaggio formattato
    """
    msg = f"{operation} completato in {elapsed_time:.1f}s"
    
    if count is not None:
        msg += f" ({count} elementi"
        if rate is not None and rate > 0:
            msg += f", {rate:.0f}/s"
        msg += ")"
    
    return msg


def format_dedup_progress_message(phase, current, total, elapsed_time, 
                                  duplicates_found=None):
    """
    Formatta messaggio di progresso deduplicazione.
    
    Args:
        phase: fase deduplicazione (es. "bitset", "sorted", "distributed")
        current: elemento corrente
        total: totale elementi
        elapsed_time: tempo trascorso
        duplicates_found: duplicati trovati finora (opzionale)
        
    Returns:
        str: messaggio formattato
    """
    pct = calculate_percentage(current, total)
    rate = calculate_rate(current, elapsed_time) if elapsed_time > 0 else 0
    
    msg = f"Deduplicazione {phase}: {current}/{total} ({pct:.1f}%)"
    
    if rate > 0:
        msg += f" - {rate:.0f}/s"
    
    if duplicates_found is not None:
        msg += f" - Duplicati: {duplicates_found}"
    
    eta = calculate_eta(total - current, rate) if rate > 0 else None
    if eta is not None and eta > 1:
        msg += f" - ETA: {format_time(eta)}"
    
    return msg


def format_batch_creation_message(batch_num, total_batches, batch_size, 
                                  total_items, elapsed_time=None):
    """
    Formatta messaggio di creazione batch.
    
    Args:
        batch_num: numero batch corrente
        total_batches: totale batch
        batch_size: dimensione del batch
        total_items: totale elementi
        elapsed_time: tempo trascorso (opzionale)
        
    Returns:
        str: messaggio formattato
    """
    pct = calculate_percentage(batch_num, total_batches)
    msg = f"Batch {batch_num}/{total_batches} ({pct:.1f}%) - size: {batch_size}"
    
    if elapsed_time is not None:
        rate = calculate_rate(batch_num, elapsed_time)
        if rate > 0:
            msg += f" - {rate:.1f} batch/s"
            eta = calculate_eta(total_batches - batch_num, rate)
            if eta is not None and eta > 1:
                msg += f" - ETA: {format_time(eta)}"
    
    return msg


def format_worker_progress_message(worker_id, processed, total, elapsed_time):
    """
    Formatta messaggio di progresso worker.
    
    Args:
        worker_id: ID del worker
        processed: elementi processati
        total: totale elementi
        elapsed_time: tempo trascorso
        
    Returns:
        str: messaggio formattato
    """
    pct = calculate_percentage(processed, total)
    rate = calculate_rate(processed, elapsed_time) if elapsed_time > 0 else 0
    
    msg = f"[Worker {worker_id}] {processed}/{total} ({pct:.1f}%)"
    
    if rate > 0:
        msg += f" - {rate:.0f}/s"
        eta = calculate_eta(total - processed, rate)
        if eta is not None and eta > 1:
            msg += f" - ETA: {format_time(eta)}"
    
    return msg


def format_summary_message(found_mhs, completed, elapsed_time, 
                          level=None, total_explored=None):
    """
    Formatta messaggio di riepilogo finale o parziale.
    
    Args:
        found_mhs: numero MHS trovati
        completed: True se completato, False se parziale
        elapsed_time: tempo totale trascorso
        level: livello raggiunto (opzionale)
        total_explored: totale ipotesi esplorate (opzionale)
        
    Returns:
        str: messaggio formattato
    """
    status = "Completato" if completed else "Parziale"
    msg = f"Trovati {found_mhs} MHS. {status}"
    
    if level is not None:
        msg += f" - Livello: {level}"
    
    if total_explored is not None:
        msg += f" - Esplorati: {total_explored}"
    
    msg += f" - Tempo: {format_time(elapsed_time)}"
    
    return msg


# =============================================================================
# GESTIONE ECCEZIONI E SALVATAGGIO EMERGENZA 
# =============================================================================

def handle_solver_exception(exception, emergency_data, out_path, 
                            start_wall, start_cpu, solver_name,
                            save_partial=True):
    # Estrae lo stato dall'eccezione se disponibile
    state = extract_state_from_exception(exception)
    
    # Determina il tipo di eccezione
    exc_type = type(exception).__name__
    
    # Prepara i dati per il salvataggio
    found_mhs = emergency_data.get('found_mhs', [])
    level_reached = emergency_data.get('level', 0)
    total_explored = emergency_data.get('total_explored', 0)
    level_stats = emergency_data.get('level_stats', [])
    
    # Salva risultati parziali se richiesto
    if save_partial and out_path:
        try:
            print(f"\n[{exc_type}] Salvataggio risultati parziali...")
            write_mhs_output(
                out_path,
                found_mhs,
                level_reached,
                total_explored,
                0,  # current_level_count
                level_stats,
                completed=False,
                start_wall=start_wall,
                perf=measure_performance(start_wall, start_cpu),
                solver_name=solver_name
            )
            print(f"[OK] Salvati {len(found_mhs)} MHS parziali in {out_path}")
        except Exception:
            print(f"[ERRORE] Errore nel salvataggio.")
    
    # Stampa messaggio di errore appropriato
    if isinstance(exception, TimeoutError):
        elapsed = time.time() - start_wall
        print(f"\n[TIMEOUT] Raggiunto dopo {elapsed:.1f}s")
        print(f"   MHS trovati: {len(found_mhs)}, Livello: {level_reached}")
    elif isinstance(exception, MemoryError):
        print(f"\n[MEMORIA] Memoria esaurita")
        print(f"   MHS trovati: {len(found_mhs)}, Livello: {level_reached}")
    elif isinstance(exception, KeyboardInterrupt):
        print(f"\n[INTERRUPT] Interruzione utente")
        print(f"   MHS trovati: {len(found_mhs)}, Livello: {level_reached}")
    else:
        print(f"\n[ERRORE]")
    
    # Rilancia l'eccezione originale
    raise exception


def safe_write_output(out_path, found_mhs, level, total_explored, 
                      current_level_count, level_stats, completed,
                      start_wall, start_cpu, solver_name, 
                      max_retries=3, retry_delay=0.5):
    for attempt in range(max_retries):
        try:
            write_mhs_output(
                out_path,
                found_mhs,
                level,
                total_explored,
                current_level_count,
                level_stats,
                completed,
                start_wall,
                perf=measure_performance(start_wall, start_cpu),
                solver_name=solver_name
            )
            return True
        except Exception:
            if attempt < max_retries - 1:
                print(f"[WARNING] Errore salvataggio (tentativo {attempt + 1}/{max_retries}).")
                time.sleep(retry_delay)
            else:
                print(f"[ERRORE] Salvataggio fallito dopo {max_retries} tentativi.")
                return False
    
    return False



