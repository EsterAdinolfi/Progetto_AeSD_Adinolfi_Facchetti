#!/usr/bin/env python3
# =============================================================================
# menu.py - Interfaccia testuale interattiva per MHS Solver
# =============================================================================
"""
Menu interattivo principale per l'utilizzo del solver MHS.

Offre all'utente la scelta tra:
1. Eseguire una singola matrice con parametri personalizzati
2. Eseguire il programma automatico su tutte le matrici selezionate
3. Visualizzare informazioni e aiuto
4. Uscire

Utilizzo:
    python menu.py
"""

import os
import sys
import subprocess
from pathlib import Path

def clear_screen():
    """Pulisce lo schermo del terminale."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Stampa l'intestazione del menu principale."""
    print("=" * 70)
    print("  Menu interattivo")
    print("  Problema: ricerca dei Minimal Hitting Set")
    print("  Autrici: Adinolfi Ester, Facchetti Nicole")
    print("=" * 70)
    print()

def print_menu():
    """Stampa le opzioni del menu principale."""
    print("Cosa vuoi fare?")
    print()
    print("  [1] Eseguire una singola matrice (run.py)")
    print("  [2] Eseguire il programma automatico su tutte le matrici (setup.py)")
    print("  [3] Informazioni e aiuto")
    print("  [4] Esci")
    print()

def list_available_matrices():
    """Elenca le matrici disponibili nella cartella corrente."""
    matrices = list(Path('.').glob('*.matrix'))
    
    if not matrices:
        print("ATTENZIONE:  Nessuna matrice trovata nella cartella corrente.")
        return []
    
    print("\nMatrici disponibili:")
    for i, matrix in enumerate(matrices, 1):
        size = os.path.getsize(matrix)
        print(f"  [{i}] {matrix.name} ({size} bytes)")
    
    return matrices

def get_matrix_path():
    """Richiede all'utente la selezione o l'inserimento del percorso della matrice."""
    print("\n" + "-" * 70)
    print("SELEZIONE MATRICE")
    print("-" * 70)
    
    matrices = list_available_matrices()
    
    print("\nOpzioni:")
    print("  • Inserisci il numero della matrice dall'elenco")
    print("  • Inserisci il percorso completo di un'altra matrice")
    print("  • Premi INVIO per tornare al menu principale")
    print()
    
    choice = input("Scelta: ").strip()
    
    if not choice:
        return None
    
    # Se è un numero, seleziona dalla lista
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(matrices):
            return str(matrices[idx])
        else:
            print(f"ERRORE: Numero non valido. Scegli tra 1 e {len(matrices)}.")
            input("\nPremi INVIO per continuare...")
            return None
    
    # Altrimenti, viene considerato come percorso
    if os.path.exists(choice):
        return choice
    else:
        print(f"ERRORE: file non trovato: {choice}")
        input("\nPremere INVIO per continuare...")
        return None

def get_run_options():
    """Richiede all'utente le opzioni di configurazione per run.py."""
    print("\n" + "-" * 70)
    print("CONFIGURAZIONE PARAMETRI")
    print("-" * 70)
    print()
    
    # Timeout
    print("Timeout in secondi:")
    print("  • Per matrici piccole: 10-30s")
    print("  • Per matrici medie: 60-120s")
    print("  • Per matrici grandi: 300-600s")
    print("  • Lasciare vuoto per nessun timeout (esegue fino al completamento)")
    timeout_input = input("Timeout [nessuno]: ").strip()
    timeout = int(timeout_input) if timeout_input.isdigit() else None
    
    # Output directory (opzionale)
    print("\nDirectory di output per file .mhs (opzionale):")
    print("  • Lasciare vuoto per salvare nella stessa directory della matrice")
    output_input = input("Output directory [auto]: ").strip()
    output = output_input if output_input else None
    

    # Opzioni avanzate
    print("\n" + "-" * 70)
    print("OPZIONI AVANZATE (opzionali)")
    print("-" * 70)
    print()
    
    while True:
        show_advanced = input("Configurare opzioni avanzate? [S/N]: ").strip().lower()
        if show_advanced == '' or show_advanced == 'n':
            show_advanced = 'n'
            break
        elif show_advanced == 's':
            break
        else:
            print("Errore: inserire 's' per configurare, 'n' per saltare, o INVIO per default")
    
    force_serial = None
    memory_limit = True
    memory_threshold = 95  # default
    no_reduction = False  # default: riduci colonne vuote
    
    if show_advanced == 's':
        # Riduzione colonne vuote
        print("\nRiduzione colonne vuote (la riduzione migliora le prestazioni senza alterare i risultati):")
        print("  • s: Rimuovi colonne vuote prima dell'elaborazione (default, consigliato)")
        print("  • n: Mantieni tutte le colonne (anche quelle vuote)")
        while True:
            reduction_input = input("Ridurre colonne vuote? [s]: ").strip().lower()
            if reduction_input == '' or reduction_input == 's':
                no_reduction = False  # Riduci (default)
                break
            elif reduction_input == 'n':
                no_reduction = True  # NON ridurre
                break
            else:
                print("Errore: inserire 's' per ridurre, 'n' per mantenere, o INVIO per default")
        
        # Forza modalità
        print("\nForzare modalità di esecuzione:")
        print("  • serial: Forza modalità seriale (1 solo processo)")
        print("  • parallel: Forza modalità parallela (anche per matrici piccole)")
        print("  • auto: Scelta automatica basata sulla dimensione (consigliato)")
        while True:
            force_input = input("Modalità [auto]: ").strip().lower()
            if force_input == '' or force_input == 'auto':
                force_serial = None  # Auto
                break
            elif force_input == 'serial':
                force_serial = True
                break
            elif force_input == 'parallel':
                force_serial = False
                break
            else:
                print("Errore: inserire 'serial', 'parallel', 'auto', o INVIO per default")
        
        # Monitoraggio memoria (solo se NON è forzata modalità seriale)
        if force_serial != True:
            print("\nMonitoraggio memoria:")
            print("  • s: Abilita (monitora RAM e interrompe se supera soglia)")
            print("  • n: Disabilita (nessun controllo, rischio esaurimento memoria)")
            while True:
                memory_input = input("Abilita monitoraggio memoria? [s]: ").strip().lower()
                if memory_input == '' or memory_input == 's':
                    memory_limit = True
                    break
                elif memory_input == 'n':
                    memory_limit = False
                    memory_threshold = 100  # Disabilitato
                    break
                else:
                    print("Errore: inserire 's' per abilitare, 'n' per disabilitare, o INVIO per default")
        else:
            # Modalità seriale: monitoraggio memoria non supportato
            memory_limit = False
            memory_threshold = 100
        
        if memory_limit:
            # Chiedi soglia personalizzata
            while True:
                threshold_input = input("Soglia percentuale memoria (50-99) [95]: ").strip()
                if not threshold_input:
                    memory_threshold = 95
                    break
                try:
                    memory_threshold = int(threshold_input)
                    if 50 <= memory_threshold <= 99:
                        break
                    else:
                        print("Errore: inserire un valore tra 50 e 99")
                except ValueError:
                    print("Errore: inserire un numero intero valido")
    
    # Parametri di parallelizzazione (solo per modalità parallela/auto)
    num_processes = None
    batch_size = None
    skip_global_dedup = False
    
    if show_advanced == 's' and force_serial != True:
        print("\nParametri di parallelizzazione (solo per modalità parallela):")
        print("  • Lasciare vuoto per configurazione automatica (consigliato)")
        
        # Numero di processi
        print("\nNumero di processi paralleli:")
        print("  • Default: CPU count - 1")
        print("  • Range consigliato: 2-16 (dipende dalla CPU)")
        processes_input = input("Numero processi [auto]: ").strip()
        if processes_input.isdigit() and int(processes_input) >= 1:
            num_processes = int(processes_input)
        
        # Dimensione batch (soglia minima)
        print("\nSoglia minima dimensione batch:")
        print("  • Default: 1000")
        print("  • La dimensione effettiva è sempre calcolata adattativamente")
        print("  • Questo valore garantisce che i batch non siano troppo piccoli")
        batch_input = input("Soglia minima batch [auto]: ").strip()
        if batch_input.isdigit() and int(batch_input) >= 1:
            batch_size = int(batch_input)
        
        # Deduplicazione globale
        print("\nDeduplicazione globale risultati:")
        print("  • n: Attiva (default, consigliato per robustezza contro edge-cases)")
        print("  • s: Disabilitata (teoricamente più corretta, più efficiente)")
        print()
        print("  Nota: L'algoritmo succL garantisce teoricamente che ogni ipotesi sia")
        print("  generata esattamente una volta. La deduplicazione offre robustezza extra")
        print("  in caso di bug o condizioni impreviste, ma non dovrebbe essere necessaria.")
        while True:
            dedup_input = input("\nVuoi disabilitare la deduplicazione globale? [n]: ").strip().lower()
            if dedup_input == '' or dedup_input == 'n':
                skip_global_dedup = False
                break
            elif dedup_input == 's':
                skip_global_dedup = True
                break
            else:
                print("Errore: inserire 's' per disabilitare, 'n' per mantenere attiva, o INVIO per default")
    
    return timeout, output, force_serial, memory_limit, memory_threshold, num_processes, batch_size, no_reduction, skip_global_dedup

def run_single_matrix():
    """Esegue run.py per una singola matrice."""
    clear_screen()
    print_header()
    
    # Selezione matrice
    matrix_path = get_matrix_path()
    if not matrix_path:
        return
    
    # Configurazione parametri
    timeout, output, force_serial, memory_limit, memory_threshold, num_processes, batch_size, no_reduction, skip_global_dedup = get_run_options()
    
    # Conferma
    print("\n" + "=" * 70)
    print("RIEPILOGO ESECUZIONE")
    print("=" * 70)
    print(f"  Matrice:       {matrix_path}")
    print(f"  Timeout:       {timeout}s" if timeout else "  Timeout:       nessuno (fino a completamento)")
    if output:
        print(f"  Output dir:    {os.path.dirname(output) if os.path.dirname(output) else '.'}")
    
    # Mostra opzioni avanzate se specificate
    if force_serial is not None or memory_limit is not None or num_processes is not None or batch_size is not None or no_reduction or skip_global_dedup:
        print("\n  Opzioni avanzate:")
        if no_reduction:
            print(f"    • Riduzione colonne:    Disabilitata (mantiene colonne vuote)")
        if force_serial is not None:
            if force_serial:
                print(f"    • Modalità:             Seriale (forzata)")
            else:
                print(f"    • Modalità:             Parallela (forzata)")
        if memory_limit:
            print(f"    • Monitoraggio RAM:     Abilitato (soglia {memory_threshold}%)")
        elif memory_limit is False:
            print(f"    • Monitoraggio RAM:     Disabilitato")
        if num_processes is not None:
            print(f"    • Processi paralleli:   {num_processes}")
        if batch_size is not None:
            print(f"    • Soglia minima batch:  {batch_size}")
        if skip_global_dedup:
            print(f"    • Deduplicazione:       Disabilitata (solo ordinamento)")
    
    print("=" * 70)
    print()
    
    while True:
        confirm = input("Vuoi procedere? [s/n]: ").strip().lower()
        if confirm == '' or confirm == 's':
            break
        elif confirm == 'n':
            print("Operazione annullata.")
            input("\nPremi INVIO per tornare al menu...")
            return
        else:
            print("Errore: inserire 's' per procedere, 'n' per annullare, o INVIO per conferma")
    
    # Costruisci comando
    # run.py accetta il file come argomento posizionale
    cmd = [sys.executable, "run.py", matrix_path]
    
    # Aggiungi parametri solo se specificati
    if timeout is not None:
        cmd.append(f"--timeout={timeout}")
    
    if output:
        cmd.append(f"--outdir={os.path.dirname(output) if os.path.dirname(output) else '.'}")
    
    # Opzioni avanzate
    if no_reduction:
        cmd.append("--no-reduction")
    
    if force_serial is True:
        cmd.append("--serial")
    elif force_serial is False:
        cmd.append("--parallel")
    
    if memory_limit:
        cmd.append("--memory-monitoring")
        cmd.append(f"--memory-threshold={memory_threshold}")
    
    # Parametri di parallelizzazione (solo se non è seriale)
    if force_serial != True:
        if num_processes is not None:
            cmd.append(f"--processes={num_processes}")
        if batch_size is not None:
            cmd.append(f"--batch-size={batch_size}")
        if skip_global_dedup:
            cmd.append("--skip-global-dedup")
    

    print("\n" + "=" * 70)
    print("ESECUZIONE IN CORSO...")
    print("=" * 70)
    print()
    
    # Esegui
    try:
        result = subprocess.run(cmd, check=False)
        print("\n" + "=" * 70)
        if result.returncode == 0:
            print(" Esecuzione completata con successo!")
        else:
            print(f"ATTENZIONE:  Esecuzione terminata con codice {result.returncode}")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("ATTENZIONE:  Esecuzione interrotta dall'utente (Ctrl+C).")
        print("=" * 70)
    except Exception:
        print(f"\nERRORE:  Errore durante l'esecuzione.")
    
    input("\nPremi INVIO per tornare al menu...")

def run_setup():
    """Esegue setup.py per l'elaborazione automatica."""
    clear_screen()
    print_header()
    
    print("SETUP AUTOMATICO")
    print("-" * 70)
    print()
    print("Questo comando eseguirà setup.py, che:")
    print("  • Legge il catalogo delle matrici selezionate")
    print("  • Calcola automaticamente i parametri ottimali")
    print("  • Esegue il solver su tutte le matrici")
    print()
    
    # Opzione timeout override
    print("Timeout in secondi per singola matrice (opzionale):")
    print("  • Lascia vuoto per nessun timeout (esegue fino al completamento)")
    print("  • Inserisci un numero di secondi per forzare lo stesso timeout per tutte")
    timeout_input = input("Timeout in secondi [nessuno]: ").strip()
    
    # Directory selezionate (opzionale)
    print("\nDirectory delle matrici selezionate (opzionale):")
    print("  • Lascia vuoto per usare 'selezionate' (default)")
    selected_dir_input = input("Directory selezionate [selezionate]: ").strip()
    selected_dir = selected_dir_input if selected_dir_input else "selezionate"
    
    # Directory risultati (opzionale)
    print("\nDirectory per salvare i risultati (opzionale):")
    print("  • Lascia vuoto per usare directory automatica basata sulla modalità")
    results_dir_input = input("Directory risultati [auto]: ").strip()
    results_dir = results_dir_input if results_dir_input else None
    

    # Opzioni avanzate
    print("\n" + "-" * 70)
    print("OPZIONI AVANZATE (opzionali)")
    print("-" * 70)
    print()
    
    while True:
        show_advanced = input("Vuoi configurare opzioni avanzate? [S/N]: ").strip().lower()
        if show_advanced == '' or show_advanced == 'n':
            show_advanced = 'n'
            break
        elif show_advanced == 's':
            break
        else:
            print("Errore: inserire 's' per configurare, 'n' per saltare, o INVIO per default")
    
    force_mode = None
    memory_monitoring = True
    memory_threshold = 95  # default
    num_processes = None
    batch_size = None
    no_reduction = False  # default: riduci colonne vuote
    
    if show_advanced == 's':
        # Riduzione colonne vuote
        print("\nRiduzione colonne vuote (la riduzione migliora le prestazioni senza alterare i risultati):")
        print("  • s: Rimuovi colonne vuote prima dell'elaborazione (default, consigliato)")
        print("  • n: Mantieni tutte le colonne (anche quelle vuote)")
        while True:
            reduction_input = input("Ridurre colonne vuote? [s]: ").strip().lower()
            if reduction_input == '' or reduction_input == 's':
                no_reduction = False  # Riduci (default)
                break
            elif reduction_input == 'n':
                no_reduction = True  # NON ridurre
                break
            else:
                print("Errore: inserire 's' per ridurre, 'n' per mantenere, o INVIO per default")
        
        # Forza modalità
        print("\nForza modalità di esecuzione per tutte le matrici:")
        print("  • serial: Forza modalità seriale")
        print("  • parallel: Forza modalità parallela")
        print("  • auto: Scelta automatica (default)")
        while True:
            force_input = input("Modalità [auto]: ").strip().lower()
            if force_input == '' or force_input == 'auto':
                force_mode = None
                break
            elif force_input == 'serial':
                force_mode = 'serial'
                break
            elif force_input == 'parallel':
                force_mode = 'parallel'
                break
            else:
                print("Errore: inserire 'serial', 'parallel', 'auto', o INVIO per default")
        
        # Monitoraggio memoria (solo se NON è forzata modalità seriale)
        if force_mode != 'serial':
            print("\nMonitoraggio memoria:")
            print("  • s: Abilita (monitora RAM, interrompe se supera soglia)")
            print("  • n: Disabilita (nessun controllo, rischio esaurimento memoria)")
            while True:
                memory_input = input("Abilita monitoraggio memoria? [s]: ").strip().lower()
                if memory_input == '' or memory_input == 's':
                    memory_monitoring = True
                    break
                elif memory_input == 'n':
                    memory_monitoring = False
                    memory_threshold = 100  # Disabilitato
                    break
                else:
                    print("Errore: inserire 's' per abilitare, 'n' per disabilitare, o INVIO per default")
        else:
            # Modalità seriale: monitoraggio memoria non supportato
            memory_monitoring = False
        
        if memory_monitoring:
            while True:
                threshold_input = input("Soglia percentuale memoria (50-99) [95]: ").strip()
                if not threshold_input:
                    memory_threshold = 95
                    break
                try:
                    memory_threshold = int(threshold_input)
                    if 50 <= memory_threshold <= 99:
                        break
                    else:
                        print("Errore: inserire un valore tra 50 e 99")
                except ValueError:
                    print("Errore: inserire un numero intero valido")
        else:
            memory_threshold = 100  # Disabilitato
        
        # Parametri di parallelizzazione (solo per modalità parallela/auto)
        if force_mode != 'serial':
            print("\nParametri di parallelizzazione (per modalità parallela/auto):")
            print("  • Lasciare vuoto per configurazione automatica (consigliato)")
            
            # Numero di processi
            print("\nNumero di processi paralleli:")
            print("  • Applica lo stesso valore a tutte le matrici")
            print("  • Auto: CPU count - 1")
            processes_input = input("Numero processi [auto]: ").strip()
            if processes_input.isdigit() and int(processes_input) >= 1:
                num_processes = int(processes_input)
            
            # Dimensione batch (soglia minima)
            print("\nSoglia minima dimensione batch:")
            print("  • Auto: 1000")
            print("  • La dimensione effettiva è sempre calcolata adattativamente")
            print("  • Questo valore garantisce che i batch non siano troppo piccoli")
            batch_input = input("Soglia minima batch [auto]: ").strip()
            if batch_input.isdigit() and int(batch_input) >= 1:
                batch_size = int(batch_input)
            
            # Deduplicazione globale
            print("\nDeduplicazione globale risultati:")
            print("  • n: Attiva (default, consigliato per robustezza contro edge-cases)")
            print("  • s: Disabilitata (teoricamente più corretta, più efficiente)")
            print()
            print("  Nota: L'algoritmo succL garantisce teoricamente che ogni ipotesi sia")
            print("  generata esattamente una volta. La deduplicazione offre robustezza extra")
            print("  in caso di bug o condizioni impreviste, ma non dovrebbe essere necessaria.")
            skip_global_dedup = False
            while True:
                dedup_input = input("\nVuoi disabilitare la deduplicazione globale? [n]: ").strip().lower()
                if dedup_input == '' or dedup_input == 'n':
                    skip_global_dedup = False
                    break
                elif dedup_input == 's':
                    skip_global_dedup = True
                    break
                else:
                    print("Errore: inserire 's' per disabilitare, 'n' per mantenere attiva, o INVIO per default")
        else:
            skip_global_dedup = False
    else:
        skip_global_dedup = False
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("RIEPILOGO SETUP")
    print("=" * 70)
    print(f"  Timeout:          {timeout_input}s" if timeout_input.isdigit() else "  Timeout:          nessuno (fino a completamento)")
    print(f"  Directory selezionate: {selected_dir}")
    if results_dir:
        print(f"  Directory risultati:    {results_dir}")
    else:
        print(f"  Directory risultati:    auto (basata su modalità)")
    if show_advanced == 's':
        print("\n  Opzioni avanzate:")
        if no_reduction:
            print(f"    • Riduzione colonne:    Disabilitata (mantiene colonne vuote)")
        if force_mode:
            print(f"    • Modalità:             {force_mode.capitalize()} (forzata)")
        else:
            print(f"    • Modalità:             Auto")
        if memory_monitoring:
            print(f"    • Monitoraggio RAM:     Abilitato (soglia {memory_threshold}%)")
        else:
            print(f"    • Monitoraggio RAM:     Disabilitato")
        if num_processes is not None:
            print(f"    • Processi paralleli:   {num_processes}")
        if batch_size is not None:
            print(f"    • Soglia minima batch:  {batch_size}")
        if skip_global_dedup:
            print(f"    • Deduplicazione:       Disabilitata (solo ordinamento)")
    print("=" * 70)
    print()
    
    while True:
        confirm = input("Vuoi procedere? [S/N]: ").strip().lower()
        if confirm == '' or confirm == 's':
            break
        elif confirm == 'n':
            print("Operazione annullata.")
            input("\nPremi INVIO per tornare al menu...")
            return
        else:
            print("Errore: inserire 's' per procedere, 'n' per annullare, o INVIO per conferma")
    
    # Costruisci comando
    cmd = [sys.executable, "setup.py"]
    if timeout_input.isdigit():
        cmd.append(f"--timeout={timeout_input}")
    
    # Directory selezionate e risultati
    if selected_dir != "selezionate":
        cmd.append(f"--selected-dir={selected_dir}")
    if results_dir:
        cmd.append(f"--results-dir={results_dir}")
    
    # Aggiungi opzioni avanzate
    if no_reduction:
        cmd.append("--no-reduction")
    if force_mode == 'serial':
        cmd.append("--serial")
    elif force_mode == 'parallel':
        cmd.append("--parallel")
    if memory_monitoring:
        cmd.append("--memory-monitoring")
        cmd.append(f"--memory-threshold={memory_threshold}")
    
    # Parametri di parallelizzazione
    if num_processes is not None:
        cmd.append(f"--processes={num_processes}")
    if batch_size is not None:
        cmd.append(f"--batch-size={batch_size}")
    if skip_global_dedup:
        cmd.append("--skip-global-dedup")
    
    print("\n" + "=" * 70)
    print("ESECUZIONE SETUP IN CORSO...")
    print("=" * 70)
    print()
    
    # Esegui
    try:
        result = subprocess.run(cmd, check=False)
        print("\n" + "=" * 70)
        if result.returncode == 0:
            print(" Setup completato con successo!")
        else:
            print(f"ATTENZIONE:  Setup terminato con codice {result.returncode}")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("ATTENZIONE:  Esecuzione interrotta dall'utente (Ctrl+C).")
        print("=" * 70)
    except Exception:
        print(f"\nERRORE!")
    
    input("\nPremi INVIO per tornare al menu...")

def show_help():
    """Mostra informazioni e aiuto."""
    clear_screen()
    print_header()
    
    print("INFORMAZIONI E AIUTO")
    print("-" * 70)
    print()
    print("DESCRIZIONE:")
    print("   Implementazione di algoritmi per la ricerca dei Minimal Hitting Set (MHS)")
    print("   su matrici binarie. Il sistema può utilizzare tecniche di parallelizzazione,")
    print("   ottimizzazione della memoria e strategie di deduplicazione avanzate per")
    print("   gestire matrici di grandi dimensioni in modo efficiente.")
    print()
    print("MODALITA' DI ESECUZIONE:")
    print()
    print("   1. SINGOLA MATRICE (run.py):")
    print("      Consente l'esecuzione su una matrice specifica con configurazione")
    print("      personalizzata dei parametri. Il sistema seleziona automaticamente")
    print("      tra modalità seriale e parallela in base alle dimensioni della matrice.")
    print()
    print("      Parametri configurabili:")
    print("      - Timeout (secondi): tempo massimo di esecuzione. Opzionale: se non specificato,")
    print("        l'algoritmo esegue fino a completamento senza limiti temporali.")
    print("      - Output directory: directory per il salvataggio del file .mhs. Se non specificata,")
    print("        viene utilizzata la stessa directory del file di input.")
    print("      - Opzioni avanzate: riduzione colonne, modalità forzata (seriale/parallela), monitoraggio RAM")
    print("        e parametri di parallelizzazione (processi, dimensione batch, deduplicazione globale).")
    print()
    print("   2. ESECUZIONE AUTOMATICA (setup.py):")
    print("      Avvia la selezione delle matrici e crea le cartelle necessarie.")
    print("      Esegue il solver su tutte le matrici presenti nel catalogo, creando file .mhs.")
    print("      Esegue l'analisi delle performance e genera report statistici.")
    print("      Genera i file results.json con metriche dettagliate e statistiche_prestazioni.txt")
    print("      con report di analisi. Gestisce automaticamente la pulizia di file .mhs spostati")
    print("      e la ri-elaborazione di matrici mancanti.")
    print()
    print("      Parametri configurabili:")
    print("      - Timeout (secondi): tempo massimo di esecuzione. Opzionale: se non specificato,")
    print("        l'algoritmo esegue fino a completamento senza limiti temporali.")
    print("      - Directory selezionate: cartella contenente le matrici da elaborare. Se non specificata,")
    print("        viene utilizzata la directory 'selezionate'.")
    print("      - Output directory: directory per il salvataggio del file .mhs. Se non specificata,")
    print("        viene utilizzata la directory creata automaticamente in base alla modalità di elaborazione selezionata.")
    print("      - Opzioni avanzate: riduzione colonne, modalità forzata (seriale/parallela), monitoraggio RAM")
    print("        e parametri di parallelizzazione (processi, dimensione batch, deduplicazione globale).")
    print()
    print("GESTIONE INTERRUZIONI:")
    print()
    print("   Durante l'esecuzione è possibile interrompere il processo in due modalità:")
    print()
    print("   - Interruzione controllata (tasto 'q' o ESC):")
    print("     Arresta l'elaborazione in modo controllato, salvando tutti i risultati")
    print("     parziali ottenuti fino al momento dell'interruzione.")
    print()
    print("   - Interruzione forzata (Ctrl+C):")
    print("     Termina immediatamente il processo. I risultati parziali vengono")
    print("     salvati, ma potrebbero essere incompleti.")
    print()
    print("FILE DI OUTPUT:")
    print()
    print("   Per esecuzione singola (run.py), file .mhs contenente:")
    print("     • Informazioni sul solver utilizzato")
    print("     • Informazioni sulla matrice di input")    
    print("     • Informazioni sugli MHS trovati (numero, cardinalità minima e massima)")
    print("     • Informazioni su timeout o interruzioni")
    print("     • Per ogni livello, informazioni sul numero di ipotesi generate e MHS trovati")
    print("     • Metadati dell'elaborazione: tempo, memoria, parametri utilizzati)")
    print("         - Tempo totale di esecuzione")
    print("         - Se calcolo in parallelo, tempo CPU totale, del master, dei worker (singoli, totale, massimo e medio)")
    print("         - Memoria RSS utilizzata")
    print("         - Picco di memoria RAM utilizzata")
    print("     • Lista dei Minimal Hitting Set trovati")
    print()
    print("   Per esecuzione automatica (setup.py):")
    print("   - Directory risultati_[modalità]/ con file .mhs per ogni matrice elaborata")
    print("   - results.json: metriche di performance dettagliate per tutte le matrici")
    print("   - statistiche_prestazioni.txt: report completo di analisi delle prestazioni")
    print()
    print("-" * 70)
    
    input("\nPremere INVIO per tornare al menu...")

def main():
    """Loop principale del menu."""
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("Scegli un'opzione [1-4]: ").strip()
        
        if choice == '1':
            run_single_matrix()
        elif choice == '2':
            run_setup()
        elif choice == '3':
            show_help()
        elif choice == '4':
            clear_screen()
            print("\nGrazie per aver usato MHS Solver.")
            print("   Arrivederci!\n")
            sys.exit(0)
        else:
            print(f"\nERRORE: opzione non valida: '{choice}'")
            print("   Scegli un numero tra 1 e 4.")
            input("\nPremi INVIO per continuare...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print("\n\nATTENZIONE: programma interrotto dall'utente.")
        print(" Arrivederci!\n")
        sys.exit(0)
