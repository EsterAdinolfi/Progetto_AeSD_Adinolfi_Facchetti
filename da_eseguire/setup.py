# =============================================================================
# setup.py - Script di orchestrazione per l'esecuzione del solver MHS
# =============================================================================
"""
Script di setup automatico per l'esecuzione completa del solver MHS.

Lo script gestisce l'intero workflow:
1. Selezione automatica delle matrici di test
2. Esecuzione del solver (seriale/parallelo/auto)
3. Collezione delle metriche di performance
4. Unificazione e analisi dei risultati

Utilizzo:
    python setup.py [opzioni]

Opzioni:
    --parallel         Forza l'utilizzo del solver parallelo per tutte le matrici
    --serial           Forza l'utilizzo del solver seriale per tutte le matrici
    --memory-monitoring Abilita monitoraggio continuo RAM con protezione (solo parallelo)
    --memory-threshold=N Soglia percentuale per monitoraggio memoria (default: 95, range: 50-99)
    --timeout=N        Timeout in secondi per matrice (default: nessuno)
    --no-reduction     Non rimuove le colonne vuote dalla matrice (default: rimuove)
    --processes=N      Numero di processi paralleli (default: auto, solo con parallelo)
    --batch-size=N     Dimensione minima batch (default: 1000, solo con parallelo)
    --skip-global-dedup Disabilita deduplicazione globale (solo ordinamento, solo con parallelo)
    --selected-dir=DIR Directory delle matrici selezionate (default: selezionate)
    --results-dir=DIR  Directory per salvare i risultati (default: basato su modalità)

Selezione automatica (default):
    - Matrici trivial/tiny/small (N≤5, M'≤30): solver seriale
    - Matrici medium/large/xlarge: solver parallelo
    (criteri allineati con matrices_selection.py)
    
Output:
    - risultati_[serial/parallel/auto]/: file .mhs con soluzioni
    - results.json: metriche di performance
    - statistiche_prestazioni.txt: report analisi
"""

import os
import subprocess
import sys
import json
import signal
import threading
import time
from datetime import datetime, timedelta
from collector_performance import statistiche
from utility import input_listener, handle_sigint, stop_requested
from cleanup_misplaced import cleanup_misplaced_mhs
from reprocess_missing import reprocess_missing_matrices

# =============================================================================
# FUNZIONI DI UTILITÀ
# =============================================================================
def ensure_dir(path):
    """
    Crea una cartella se non esiste già.
    
    Args:
        path: percorso della cartella da creare
    """
    print("\n\n" + "~"*80)
    print("VERIFICA E CREAZIONE CARTELLE NECESSARIE")
    print(f"Cartella: {path}")
    print("~"*80)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"È stata creata la cartella \"{path}\"")
    else:
        print(f"La cartella \"{path}\" è già esistente")

def is_empty_dir(path, ext_filter=None):
    """
    Verifica se una cartella è vuota (o non contiene file con estensione specifica).
    
    Args:
        path: percorso della cartella da verificare
        ext_filter: estensione file da cercare (es. '.matrix', '.mhs')
                    Se None, viene cercato qualsiasi file
    
    Returns:
        True se la cartella è vuota (o non contiene file dell'estensione), False altrimenti
    """
    if not os.path.exists(path):
        return True
    for root, dirs, files in os.walk(path):
        for f in files:
            if not ext_filter or f.endswith(ext_filter):
                return False
    return True

# =============================================================================
# ESECUZIONE DEL SOLVER
# =============================================================================
def run_solver_on_selected(selected_dir="selezionate", timeout=None, solver_mode="auto", results_dir="risultati_auto", memory_monitoring=False, memory_threshold=95, num_processes=None, batch_size=None, no_reduction=False, skip_global_dedup=False):
    """
    Esegue il solver su tutte le matrici nella cartella selezionata.
    
    Per ogni matrice:
    1. Viene determinato quale solver utilizzare (seriale/parallelo/auto)
    2. Viene costruito il comando con le opzioni appropriate
    3. Viene eseguito il solver con timeout (se specificato)
    4. Viene verificato che il file .mhs sia stato generato
    
    Args:
        selected_dir: cartella contenente le matrici da elaborare
        timeout: timeout in secondi per ogni matrice (None = nessun timeout)
        solver_mode: "auto", "serial" o "parallel"
        results_dir: cartella di output
        memory_monitoring: abilita monitoraggio continuo RAM con protezione (solo parallelo)
        memory_threshold: soglia percentuale per monitoraggio memoria (default: 95)
        num_processes: numero di processi paralleli (None = auto, solo parallelo)
        batch_size: dimensione batch per parallelizzazione (None = auto, solo parallelo)
        no_reduction: non rimuove le colonne vuote dalla matrice
    """    
    # Importa la funzione run_solver dal modulo run
    from run import run_solver
    
    global stop_requested
    
    # Verifichiamo che la cartella selezionata esista
    if not os.path.exists(selected_dir):
        print(f"Errore: la cartella '{selected_dir}' non esiste!")
        return
    
    # Carica selection.json per ottenere i metadata delle matrici
    selection_dict = {}
    if os.path.exists("selection.json"):
        try:
            with open("selection.json", "r", encoding="utf-8") as f:
                selection_data = json.load(f)
                # Crea dizionario: nome_file -> metadata
                for item in selection_data:
                    selection_dict[item["file"]] = item
                print(f"Caricati metadata per {len(selection_dict)} matrici da selection.json")
        except Exception:
            print(f"Attenzione: impossibile caricare selection.json.")
            print("I file .mhs verranno generati senza metadata aggiuntivi")
    else:
        print("Attenzione: selection.json non trovato. I file .mhs verranno generati senza metadata aggiuntivi")

    # Contatori per statistiche finali
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # Verifichiamo se esistono le cartelle benchmarks1 e benchmarks2
    benchmarks_dirs = []
    if os.path.exists(os.path.join(selected_dir, "benchmarks1")):
        benchmarks_dirs.append("benchmarks1")
    if os.path.exists(os.path.join(selected_dir, "benchmarks2")):
        benchmarks_dirs.append("benchmarks2")

    for bench in benchmarks_dirs:
        src_dir = os.path.join(selected_dir, bench) if bench else selected_dir
        out_dir = os.path.join(results_dir, bench) if bench else results_dir
        
        # Assicuriamoci che la cartella di output esista
        os.makedirs(out_dir, exist_ok=True)
        
        print(f"\nCerco matrici in {src_dir}...")
        
        # PULIZIA PRELIMINARE: verifica se ci sono file .mhs nella cartella sorgente
        # (potrebbero essere stati creati lì per errore in esecuzioni precedenti)
        print(f"Verifico presenza di file .mhs nella cartella sorgente...")
        misplaced_mhs = [f for f in os.listdir(src_dir) if f.endswith(".mhs")]
        if misplaced_mhs:
            print(f"Attenzione: trovati {len(misplaced_mhs)} file .mhs nella cartella sorgente (posizione errata)")
            import shutil
            for mhs_file in misplaced_mhs:
                src_path = os.path.join(src_dir, mhs_file)
                dest_path = os.path.join(out_dir, mhs_file)
                print(f"  Sposto {mhs_file} -> {dest_path}")
                try:
                    shutil.move(src_path, dest_path)
                    print(f"  Spostato correttamente")
                except Exception:
                    print(f"  Errore durante lo spostamento.")
        else:
            print(f"Nessun file .mhs trovato nella cartella sorgente (OK)")
        
        # Verifica che la directory esista e contenga file
        if not os.path.exists(src_dir):
            print(f"Errore: la directory {src_dir} non esiste!")
            continue
            
        # Trova tutti i file .matrix, escludendo solo i duplicati veri (_dup)
        # I file _v2 hanno contenuto diverso e devono essere elaborati
        all_matrix_files = [f for f in os.listdir(src_dir) if f.endswith(".matrix")]
        matrix_files = [f for f in all_matrix_files if "_dup" not in f]
        
        excluded_count = len(all_matrix_files) - len(matrix_files)
        if excluded_count > 0:
            print(f"Esclusi {excluded_count} file duplicati (rinominati con _dup)")
        
        if not matrix_files:
            print(f"Attenzione: nessun file .matrix trovato in {src_dir}")
            continue
            
        print(f"Trovati {len(matrix_files)} file .matrix in {src_dir} (da elaborare)")
        
        for fname in matrix_files:
            # Verifica se è stata richiesta un'interruzione
            if stop_requested:
                print("\nInterruzione richiesta dall'utente. Interrompo l'elaborazione delle matrici.")
                return
                
            in_path = os.path.join(src_dir, fname)
            
            # Parametri per il solver
            force_serial = solver_mode == "serial"
            force_parallel = solver_mode == "parallel"
            
            # Utilizziamo direttamente la funzione run_solver importata da run.py
            print(f"\nProcesso {in_path}...")
            
            # Comando di base - utilizziamo il nuovo script run.py che sceglie automaticamente il solver
            cmd = [
                sys.executable, 
                "run.py",
                in_path,
                f"--outdir={out_dir}"  # Specifico direttamente la directory di output
            ]
            
            # Aggiungi timeout solo se specificato
            if timeout is not None:
                cmd.append(f"--timeout={timeout}")
            
            # Aggiungiamo opzioni in base alla modalità
            if force_serial:
                cmd.append("--serial")
            elif force_parallel:
                cmd.append("--parallel")
                
            # Aggiungiamo l'opzione memory-monitoring se specificata
            if memory_monitoring and not force_serial:
                cmd.append("--memory-monitoring")
                cmd.append(f"--memory-threshold={memory_threshold}")
            
            # Aggiungiamo parametri di parallelizzazione se specificati (solo per parallelo)
            if not force_serial:
                if num_processes is not None:
                    cmd.append(f"--processes={num_processes}")
                if batch_size is not None:
                    cmd.append(f"--batch-size={batch_size}")
            
            # Aggiungiamo l'opzione no-reduction se specificata
            if no_reduction:
                cmd.append("--no-reduction")
            
            # Aggiungiamo l'opzione skip-global-dedup se specificata (solo per parallelo)
            if skip_global_dedup and not force_serial:
                cmd.append("--skip-global-dedup")
            
            # Aggiungiamo i metadata se disponibili da selection.json
            if fname in selection_dict:
                metadata = selection_dict[fname]
                if "origine" in metadata:
                    cmd.append(f"--origine={metadata['origine']}")
                if "densita" in metadata:
                    cmd.append(f"--densita={metadata['densita']}")
                if "categoria" in metadata:
                    cmd.append(f"--categoria={metadata['categoria']}")
                
            # Esegui il comando
            print(f"Eseguo: {' '.join(cmd)}")
            total_processed += 1
            
            try:
                # Modalità standard: mostra tutto l'output in tempo reale
                print()  # Linea vuota per separare visivamente l'output del solver
                try:
                    # Esegui il comando e cattura eventuali errori
                    result = subprocess.run(cmd, capture_output=False, text=True)
                except KeyboardInterrupt:
                    # Ctrl+C premuto durante l'esecuzione
                    print("\n\nInterruzione richiesta dall'utente (Ctrl+C).")
                    stop_requested = True
                    return
                except Exception:
                    print(f"\n\nERRORE durante l'esecuzione del comando.")
                    print("Continuo con la matrice successiva...")
                    total_failed += 1
                    continue
                
                # Controlla periodicamente se è stata richiesta un'interruzione
                if stop_requested:
                    print("\nInterruzione richiesta dall'utente. Interrompo l'elaborazione delle matrici.")
                    return
                
                # Verifica il codice di uscita del processo
                if result.returncode != 0:
                    print(f"Attenzione: elaborazione terminata con exit code {result.returncode}")
                    
                # Verifichiamo che il file .mhs sia stato creato nella directory corretta
                base_name = os.path.splitext(os.path.basename(in_path))[0]
                mhs_file = os.path.join(out_dir, f"{base_name}.mhs")
                
                # Controlla anche se il file è stato creato nella directory sbagliata (stessa della matrice)
                wrong_location = os.path.join(src_dir, f"{base_name}.mhs")
                
                if os.path.exists(mhs_file):
                    print(f"File di output generato correttamente: {mhs_file}")
                    total_success += 1
                elif os.path.exists(wrong_location):
                    # Il file è stato creato nella directory sbagliata, spostalo
                    print(f"File .mhs trovato in posizione errata: {wrong_location}")
                    print(f"  Lo sposto nella directory corretta: {mhs_file}")
                    try:
                        import shutil
                        shutil.move(wrong_location, mhs_file)
                        print(f"File spostato correttamente")
                        total_success += 1
                    except Exception:
                        print(f"ERRORE nello spostamento del file.")
                        total_failed += 1
                else:
                    print(f"ERRORE: file di output non trovato né in {mhs_file} né in {wrong_location}")
                    print(f"  Questa matrice potrebbe essere stata interrotta o aver causato un crash.")
                    total_failed += 1
                    # Continua comunque con le altre matrici (non interrompiamo il batch)
                    
            except KeyboardInterrupt:
                print("\nInterruzione richiesta dall'utente durante l'elaborazione della matrice.")
                stop_requested = True
                return
            except Exception:
                print(f"ERRORE CRITICO durante l'elaborazione di {fname}.")
                total_failed += 1
                # Continua con le altre matrici anche in caso di errore critico
    
    # Stampa riepilogo finale
    print("\n" + "="*60)
    print("RIEPILOGO ELABORAZIONE BATCH")
    print("="*60)
    print(f"Matrici elaborate:    {total_processed}")
    print(f"Successi:          {total_success}")
    print(f"Fallimenti:        {total_failed}")
    if total_processed > 0:
        success_rate = (total_success / total_processed) * 100
        print(f"Tasso di successo:    {success_rate:.1f}%")
    print("="*60 + "\n")

# ===== FUNZIONE MAIN =====

def main():
    """
    Funzione principale che orchestra il workflow completo:
    1. Selezione delle matrici (se necessario)
    2. Esecuzione del solver su ciascuna matrice
    3. Raccolta delle prestazioni
    4. Unificazione dei risultati in un unico JSON
    5. Analisi e visualizzazione delle statistiche
    """
    global stop_requested
    stop_requested = False
    
    # Inizializza il listener per interruzioni utente
    listener_thread = threading.Thread(target=input_listener, daemon=True)
    listener_thread.start()
    
    signal.signal(signal.SIGINT, handle_sigint)
    
    # Parse degli argomenti da riga di comando
    use_parallel = "--parallel" in sys.argv
    use_serial = "--serial" in sys.argv
    use_memory_monitoring = "--memory-monitoring" in sys.argv
    no_reduction = "--no-reduction" in sys.argv
    
    # Estrae il valore del timeout (default: None = nessun timeout)
    timeout = None
    for arg in sys.argv:
        if arg.startswith("--timeout="):
            try:
                timeout = int(arg.split("=")[1])
                print(f"Timeout impostato a {timeout} secondi dalla linea di comando")
            except ValueError:
                print("Valore non valido per --timeout, ignoro e uso nessun timeout")
    
    # Estrae il valore della soglia memoria (default: 95%)
    memory_threshold = 95
    for arg in sys.argv:
        if arg.startswith("--memory-threshold="):
            try:
                memory_threshold = int(arg.split("=")[1])
                if memory_threshold < 50 or memory_threshold > 99:
                    print(f"Avviso: soglia memoria {memory_threshold}% fuori range consigliato (50-99%). Uso comunque il valore specificato.")
            except ValueError:
                print("Valore non valido per --memory-threshold, uso il default di 95%")
    
    # Estrae il numero di processi paralleli (default: None = auto)
    num_processes = None
    for arg in sys.argv:
        if arg.startswith("--processes="):
            try:
                num_processes = int(arg.split("=")[1])
                if num_processes < 1:
                    print(f"Errore: numero di processi deve essere >= 1")
                    sys.exit(1)
                print(f"Numero di processi paralleli impostato a {num_processes}")
            except ValueError:
                print("Valore non valido per --processes, uso la configurazione automatica")
    
    # Estrae la dimensione del batch (default: None = auto)
    batch_size = None
    for arg in sys.argv:
        if arg.startswith("--batch-size="):
            try:
                batch_size = int(arg.split("=")[1])
                if batch_size < 1:
                    print(f"Errore: dimensione batch deve essere >= 1")
                    sys.exit(1)
                print(f"Dimensione batch impostata a {batch_size}")
            except ValueError:
                print("Valore non valido per --batch-size, uso la configurazione automatica")
    
    # Verifica se disabilitare la deduplicazione globale
    skip_global_dedup = "--skip-global-dedup" in sys.argv
    if skip_global_dedup:
        print("Deduplicazione globale disabilitata (solo ordinamento canonico)")
    
    # Estrae la directory selezionate (default: selezionate)
    selected_dir = "selezionate"
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--selected-dir":
            if i + 1 < len(sys.argv):
                selected_dir = sys.argv[i + 1]
                print(f"Directory selezionate impostata a {selected_dir}")
                i += 1  # Salta il valore
        elif arg.startswith("--selected-dir="):
            selected_dir = arg.split("=", 1)[1]
            print(f"Directory selezionate impostata a {selected_dir}")
        i += 1
    
    # Estrae la directory risultati (default: basato su modalità)
    results_dir_override = None
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--results-dir":
            if i + 1 < len(sys.argv):
                results_dir_override = sys.argv[i + 1]
                print(f"Directory risultati impostata a {results_dir_override}")
                i += 1  # Salta il valore
        elif arg.startswith("--results-dir="):
            results_dir_override = arg.split("=", 1)[1]
            print(f"Directory risultati impostata a {results_dir_override}")
        i += 1
    
    # Determina la modalità del solver
    if use_parallel and use_serial:
        print("Errore: non è possibile specificare sia --parallel che --serial")
        sys.exit(1)
    elif use_parallel:
        solver_mode = "parallel"
        results_dir = "risultati_parallel"
    elif use_serial:
        solver_mode = "serial"
        results_dir = "risultati_serial"
    else:
        solver_mode = "auto"
        results_dir = "risultati_auto"
    
    # Se specificata, sovrascrivi la directory risultati
    if results_dir_override:
        results_dir = results_dir_override
                
    if use_memory_monitoring and use_serial:
        print("Nota: l'opzione --memory-monitoring viene ignorata quando si usa la versione seriale")
    
    print("~"*80)
    print("AVVIO DEL PROGRAMMA DI RICERCA DEI MHS")
    print("Progetto del corso di Algoritmi e Strutture Dati - Adinolfi, Facchetti")
    print("Programma di ricerca dei MHS. \nModalità usata:", solver_mode)
    print("~"*80)

    # 1. Verifica della cartella risultati
    ensure_dir(results_dir)

    # 2. Verifica della cartella selezionate
    ensure_dir(selected_dir)

    # 3. Se la cartella selezionate è vuota (anche le sottocartelle), lancia matrices_selection.py per selezionare le matrici
    print("\n\n" + "~"*80)
    print("SELEZIONE DELLE MATRICI")
    print("~"*80)
    
    # Verifica interruzione
    if stop_requested:
        print("\nInterruzione richiesta dall'utente. Chiusura in corso...")
        return
        
    if selected_dir == "selezionate" and is_empty_dir(selected_dir):
        print(f"Cartella '{selected_dir}' vuota. Avvio di matrices_selection.py.")
        # Usa lo stesso interprete con cui è lanciato questo script per evitare problemi di versioni diverse o di ambiente virtuale
        try:
            result = subprocess.run([sys.executable, "matrices_selection.py"])
            # Verifica se l'utente ha interrotto durante la selezione
            if result.returncode != 0:
                stop_requested = True
                return
        except KeyboardInterrupt:
            print("\n")  # Solo una linea vuota per separazione visiva
            stop_requested = True
            return
    else:
        if selected_dir == "selezionate":
            print(f"Le matrici sono già state selezionate in '{selected_dir}'.")
        else:
            print(f"Utilizzo matrici dalla directory personalizzata '{selected_dir}'.")
    
    # FASE 4: esecuzione del solver su tutte le matrici selezionate
    print("\n\n" + "~"*80)
    print("ESECUZIONE DEL SOLVER")
    print("~"*80)
    
    if stop_requested:
        print("\nInterruzione richiesta dall'utente. Chiusura in corso...")
        return
        
    if is_empty_dir(results_dir, ".mhs"):
        if timeout is not None:
            print(f"Cartella '{results_dir}' vuota. Avvio del solver in modalità {solver_mode} su tutte le matrici selezionate con timeout di {timeout} secondi.")
        else:
            print(f"Cartella '{results_dir}' vuota. Avvio del solver in modalità {solver_mode} su tutte le matrici selezionate senza timeout.")
        try:
            run_solver_on_selected(selected_dir, timeout, solver_mode, results_dir, use_memory_monitoring, memory_threshold, num_processes, batch_size, no_reduction, skip_global_dedup)
        except KeyboardInterrupt:
            print("\nInterruzione richiesta dall'utente durante l'esecuzione del solver.")
            stop_requested = True
            return
    else:
        print(f"I MHS sono già stati calcolati nella cartella '{results_dir}'.")

    # FASE 4.5: Pulizia file mal posizionati e riprocessamento matrici mancanti
    print("\n\n" + "~"*80)
    print("VERIFICA E PULIZIA POST-ELABORAZIONE")
    print("~"*80)
    
    if stop_requested:
        print("\nInterruzione richiesta dall'utente. Chiusura in corso...")
        return
    
    # Pulizia automatica dei file .mhs mal posizionati
    print("\n1. Verifica posizionamento file .mhs...")
    try:
        moved, errors = cleanup_misplaced_mhs(
            selected_dir=selected_dir,
            results_dir=results_dir,
            verbose=True
        )
        if moved > 0:
            print(f"   {moved} file .mhs spostati nella posizione corretta")
        if errors > 0:
            print(f"   {errors} errori durante lo spostamento")
    except KeyboardInterrupt:
        print("\nInterruzione richiesta dall'utente durante la pulizia dei file.")
        stop_requested = True
        return
    except Exception:
        print(f"   Errore durante la pulizia.")
        print("   Proseguo comunque con il riprocessamento...")
    
    # Riprocessamento automatico delle matrici mancanti
    print("\n2. Verifica matrici non processate...")
    try:
        success, failed = reprocess_missing_matrices(
            selected_dir=selected_dir,
            results_dir=results_dir,
            timeout=timeout,
            solver_mode=solver_mode,
            memory_monitoring=use_memory_monitoring,
            memory_threshold=memory_threshold,
            num_processes=num_processes,
            batch_size=batch_size,
            no_reduction=no_reduction,
            skip_global_dedup=skip_global_dedup,
            verbose=True
        )
        if success > 0:
            print(f"   {success} matrici riprocessate con successo")
        if failed > 0:
            print(f"   {failed} matrici fallite durante il riprocessamento")
    except KeyboardInterrupt:
        print("\nInterruzione richiesta dall'utente durante il riprocessamento.")
        stop_requested = True
        return
    except Exception:
        print(f"   Errore durante il riprocessamento.")
        print("   Proseguo comunque con la raccolta delle prestazioni...")

    # FASE 5: raccolta delle prestazioni dai file .mhs generati
    print("\n\n" + "~"*80)
    print("COLLEZIONE DELLE PRESTAZIONI")
    print("~"*80)
    
    if stop_requested:
        print("\nInterruzione richiesta dall'utente. Chiusura in corso...")
        return
        
    out_json = os.path.join(results_dir, "results.json")
    if not os.path.exists(out_json):
        print(f"{out_json} non trovato. Raccolta delle prestazioni tramite collector_performance.py.")
        try:
            subprocess.run([sys.executable, "collector_performance.py", results_dir, out_json])
        except KeyboardInterrupt:
            print("\nInterruzione richiesta dall'utente durante la raccolta delle prestazioni.")
            stop_requested = True
            return
    else:
        print(f"Il file JSON dei risultati è già esistente ({out_json}).")

    # FASE 6: Analisi delle prestazioni e generazione del report testuale
    # Nota: l'unificazione non è più necessaria perché i metadata sono già nei file .mhs
    print("\n\n" + "~"*80)
    print("ANALISI DELLE PRESTAZIONI")
    print("~"*80)
    
    if stop_requested:
        print("\nInterruzione richiesta dall'utente. Chiusura in corso...")
        return
        
    output_txt = os.path.join(results_dir, "statistiche_prestazioni.txt")
    print(f"\nAnalisi delle prestazioni e generazione del report in {output_txt}")
    try:
        statistiche(results_dir, out_json, output_txt)
    except KeyboardInterrupt:
        print("\nInterruzione richiesta dall'utente durante l'analisi delle prestazioni.")
        print("\nProcesso terminato su richiesta dell'utente.")
        return

    print("\nTutto completato!")
if __name__ == "__main__":
    main()
