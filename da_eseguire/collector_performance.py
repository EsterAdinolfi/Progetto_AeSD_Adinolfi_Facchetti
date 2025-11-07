#!/usr/bin/env python3
"""
collector_performance.py

Script per la raccolta e l'analisi delle prestazioni dai file di output (.mhs) generati dai solver MHS.

FUNZIONALITÀ PRINCIPALI:
1. Scandisce ricorsivamente una directory alla ricerca di file .mhs
2. Estrae metriche di performance da ciascun file:
   - Tempi di esecuzione (reale, CPU master, CPU workers per livello)
   - Consumo di memoria (RSS, picco di memoria)
   - Numero di MHS trovati e stato di completamento
   - Dimensioni matrice (N, M, M')
   - Densità e categoria della matrice
3. Calcola statistiche aggregate (media, mediana, moda) per:
   - Gruppi di benchmark (benchmarks1, benchmarks2)
   - Categorie di matrice (trivial, tiny, small, medium, large, xlarge)
4. Analizza correlazioni densità-prestazioni con doppio approccio:
   - Solo matrici completate (elimina bias da timeout)
   - Tutte le matrici (include casi limite)
5. Genera report in formato JSON (strutturato) e TXT (leggibile)

UTILIZZO:
    python collector_performance.py [dir_input] [output.json] [output_txt]

    dir_input   - Directory contenente i file .mhs da analizzare (default: "risultati")
    output.json - File JSON di output con le prestazioni aggregate (default: "dir_input/results.json")
    output_txt  - File testuale opzionale per report leggibile (default: nessuno)

ESEMPI:
    # Analizza risultati in modalità automatica
    python collector_performance.py risultati_auto results.json statistiche.txt
    
    # Analizza risultati seriali
    python collector_performance.py risultati_serial results.json
    
    # Uso programmatico da altri script
    from collector_performance import statistiche
    statistiche('risultati_auto', 'results.json', 'statistiche.txt')

FORMATO FILE .MHS:
    I file .mhs contengono commenti iniziali con ";;;" seguiti dalla lista degli MHS.
    Le metriche vengono estratte da righe come:
    - ";;; Numero di MHS trovati = 42"
    - ";;; Tempo reale = 12.34 s"
    - ";;; CPU time totale = 23.45 s"
    - ";;; Memoria RSS = 1234.56 KB"
    - ";;; Categoria: small"

AUTORI:
    Progetto AeSD - Algoritmi e Strutture Dati
"""

import os
import sys
import json
import glob
import statistics
import math
from collections import defaultdict, Counter


# ===== UTILITY: CALCOLO METRICHE =====

def get_cpu_time(result, cpu_metric):
    """
    Estrae il tempo CPU dal risultato usando la metrica specificata.
    
    Supporta diverse metriche per aggregare i tempi dei worker paralleli:
    - 'cpu_master': solo tempo del processo master (coordinatore)
    - 'cpu_total': somma dei tempi di tutti i worker
    - 'cpu_max': massimo tempo tra i worker (approssima tempo wall-clock)
    - 'cpu_avg': media dei tempi dei worker
    
    Args:
        result (dict): dizionario con i risultati di un singolo test, contenente:
            - 'tempo_cpu': tempo CPU del master (float, opzionale)
            - 'cpu_worker_times': lista o lista di liste dei tempi worker (opzionale)
        cpu_metric (str): metrica da usare per il calcolo
    
    Returns:
        float or None: tempo CPU calcolato secondo la metrica, o None se non disponibile
        
    Note:
        - Per esecuzioni seriali, cpu_worker_times è None o lista vuota
        - Per esecuzioni parallele, cpu_worker_times può essere:
          * Lista piatta: [t1, t2, t3, ...] (formato vecchio)
          * Lista di liste: [[t1_L0], [t2_L1, t3_L1], ...] (formato nuovo, per livello)
    """
    if cpu_metric == 'cpu_master':
        return result.get('tempo_cpu')

    worker_times = result.get('cpu_worker_times')
    if not worker_times or not isinstance(worker_times, list):
        return None

    if cpu_metric == 'cpu_total':
        return sum(worker_times)
    elif cpu_metric == 'cpu_max':
        return max(worker_times) if worker_times else None
    elif cpu_metric == 'cpu_avg':
        return sum(worker_times) / len(worker_times) if worker_times else None

    return None


def calculate_statistics(values):
    """
    Calcola le statistiche descrittive di base di una lista di valori numerici.
    
    Utile per analizzare distribuzioni di metriche di performance (tempi, memoria, ecc.).
    Filtra automaticamente valori nulli (None) prima del calcolo.
    
    Args:
        values (list): lista di valori numerici (int o float), può contenere None
        
    Returns:
        dict: dizionario con tre chiavi:
            - 'media' (float or None): media aritmetica
            - 'mediana' (float or None): valore mediano (50° percentile)
            - 'moda' (float or None): valore più frequente, o None se non esiste moda unica
            
    Examples:
        >>> calculate_statistics([1, 2, 3, 4, 5])
        {'media': 3.0, 'mediana': 3, 'moda': None}
        
        >>> calculate_statistics([1, 2, 2, 3, None])
        {'media': 2.0, 'mediana': 2, 'moda': 2}
        
    Note:
        - Restituisce {'media': None, 'mediana': None, 'moda': None} se la lista è vuota
          o contiene solo valori None
        - La moda può essere None anche con dati validi se non esiste un valore predominante
    """
    if not values:
        return {'media': None, 'mediana': None, 'moda': None}
    
    # Filtra valori nulli
    valid_values = [v for v in values if v is not None]
    
    if not valid_values:
        return {'media': None, 'mediana': None, 'moda': None}
    
    media = sum(valid_values) / len(valid_values)
    mediana = statistics.median(valid_values)
    
    # Gestisce eccezione quando non esiste una moda unica
    try:
        moda = statistics.mode(valid_values)
    except statistics.StatisticsError:
        moda = None
    
    return {
        'media': media,
        'mediana': mediana,
        'moda': moda
    }


def calculate_correlation(x_values, y_values):
    """
    Calcola il coefficiente di correlazione di Pearson tra due serie di valori.
    
    La correlazione di Pearson misura la relazione lineare tra due variabili:
    - r = +1: correlazione positiva perfetta (all'aumentare di x, aumenta y)
    - r = -1: correlazione negativa perfetta (all'aumentare di x, diminuisce y)
    - r = 0: nessuna correlazione lineare
    
    Interpretazione pratica:
    - |r| < 0.3: correlazione debole o trascurabile
    - 0.3 ≤ |r| < 0.7: correlazione moderata
    - |r| ≥ 0.7: correlazione forte
    
    Args:
        x_values (list): lista di valori per la variabile indipendente (es: densità)
        y_values (list): lista di valori per la variabile dipendente (es: tempo)
    
    Returns:
        float or None: coefficiente di correlazione nell'intervallo [-1, +1],
                      o None se il calcolo non è possibile (dati insufficienti/invalidi)
    
    Note:
        - Richiede almeno 2 coppie valide (x, y) con entrambi i valori non None
        - Filtra automaticamente coppie con valori None
        - Usa statistics.correlation() se disponibile (Python 3.10+),
          altrimenti implementazione manuale tramite formula standard
        - Restituisce None se denominatore è zero (varianza nulla)
    
    Examples:
        >>> calculate_correlation([1, 2, 3], [2, 4, 6])  # Perfetta correlazione positiva
        1.0
        
        >>> calculate_correlation([1, 2, 3], [3, 2, 1])  # Perfetta correlazione negativa
        -1.0
    """
    if not x_values or not y_values or len(x_values) != len(y_values):
        return None

    # Filtra coppie valide (entrambe non None)
    valid_pairs = [(x, y) for x, y in zip(x_values, y_values) if x is not None and y is not None]

    if len(valid_pairs) < 2:
        return None

    x_vals, y_vals = zip(*valid_pairs)

    try:
        # Usa statistics.correlation() se disponibile (Python 3.10+)
        if hasattr(statistics, 'correlation'):
            return statistics.correlation(x_vals, y_vals)
        else:
            # Fallback per versioni precedenti: implementazione manuale
            n = len(x_vals)
            sum_x = sum(x_vals)
            sum_y = sum(y_vals)
            sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
            sum_x2 = sum(x * x for x in x_vals)
            sum_y2 = sum(y * y for y in y_vals)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))

            if denominator == 0:
                return None

            return numerator / denominator
    except:
        return None


# ===== PARSING E ESTRAZIONE DATI =====

def parse_mhs_file(filename):
    """
    Estrae le informazioni di prestazione da un file .mhs generato dal solver MHS.
    
    I file .mhs contengono due sezioni principali:
    1. Header con commenti ";;;" contenenti metriche di performance e metadati
    2. Corpo con la lista degli MHS trovati (non elaborato da questa funzione)
    
    METRICHE ESTRATTE:
    - Performance:
      * tempo_reale: tempo wall-clock di esecuzione (secondi)
      * tempo_cpu: tempo CPU del processo master (secondi)
      * cpu_worker_times: tempi CPU dei worker (lista o lista di liste per livello)
      * mem_rss_kb: memoria RSS al termine (KB)
      * mem_picco_kb: picco di memoria durante l'esecuzione (KB) o "non_rilevato"
    
    - Risultati:
      * MHS_trovati: numero di Minimal Hitting Sets identificati
      * completato: True se terminato con successo, False se timeout/errore
    
    - Caratteristiche matrice:
      * N: numero di righe della matrice originale
      * M: numero di colonne della matrice originale
      * M_ridotto: numero di colonne dopo rimozione colonne vuote
      * densita: percentuale di celle con valore 1 (0.0-1.0)
      * categoria: classificazione dimensionale (trivial, tiny, small, medium, large, xlarge)
      * origine: cartella sorgente (benchmarks1 o benchmarks2)
    
    FORMATO CPU WORKER TIMES:
    - None: esecuzione seriale (nessun worker)
    - Lista piatta [t1, t2, ...]: formato vecchio, tempi aggregati
    - Lista di liste [[t1_L0], [t2_L1, t3_L1], ...]: formato nuovo, tempi per livello
    
    Args:
        filename (str): percorso assoluto del file .mhs da analizzare
        
    Returns:
        dict: dizionario con tutte le informazioni estratte. Chiavi:
            - file (str): nome del file
            - path (str): percorso completo
            - MHS_trovati (int): numero MHS trovati
            - completato (bool): stato di completamento
            - tempo_reale (float or None): tempo wall-clock in secondi
            - tempo_cpu (float or None): tempo CPU master in secondi
            - cpu_worker_times (list or None): tempi worker
            - mem_rss_kb (float or None): memoria RSS in KB
            - mem_picco_kb (float, str, or None): picco memoria in KB o "non_rilevato"
            - N (int or None): numero righe
            - M (int or None): numero colonne originali
            - M_ridotto (int or None): numero colonne dopo riduzione
            - origine (str or None): cartella benchmark sorgente
            - densita (float or None): densità matrice (0.0-1.0)
            - categoria (str or None): categoria dimensionale
    
    Note:
        - Se un campo non è trovato nel file, il valore resta None (o default appropriato)
        - Gestisce sia formato vecchio (CPU times piatti) che nuovo (per livello)
        - I valori di memoria possono essere "non_rilevato" se il monitoraggio era disabilitato
    
    Raises:
        Non solleva eccezioni: errori di parsing sono silenziati e il campo resta None
    """
    info = {
        'file': os.path.basename(filename),
        'path': filename,
        'MHS_trovati': 0,
        'completato': False,
        'tempo_reale': None,
        'tempo_cpu': None,
        'cpu_worker_times': None,
        'mem_rss_kb': None,
        'mem_picco_kb': None,
        'N': None,
        'M': None,
        'M_ridotto': None,
        'origine': None,
        'densita': None,
        'categoria': None
    }
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Estrae le informazioni dai commenti in testa al file (righe che iniziano con ";;;")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            
            if not line.startswith(';;;'):
                break
                
            if 'Numero di MHS trovati' in line:
                try:
                    info['MHS_trovati'] = int(line.split('=')[1].strip())
                except (IndexError, ValueError):
                    pass
                    
            elif 'Completato?' in line:
                val = line.split("?")[1].strip().lower()
                if val in ("true", "false"):
                    info['completato'] = (val == "true")
                
            elif 'Tempo trascorso' in line:
                try:
                    info['tempo_reale'] = float(line.split('=')[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
                    
            elif 'Tempo reale' in line:
                try:
                    info['tempo_reale'] = float(line.split('=')[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
                    
            elif 'CPU time totale' in line:
                try:
                    info['tempo_cpu'] = float(line.split('=')[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
            
            elif 'Categoria:' in line:
                try:
                    info['categoria'] = line.split(':')[1].strip()
                except (IndexError, ValueError):
                    pass
                    
            elif 'Memoria RSS' in line:
                try:
                    info['mem_rss_kb'] = float(line.split('=')[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
                    
            elif 'Picco memoria' in line:
                try:
                    # Controlla se è "non rilevato"
                    value_part = line.split('=')[1].strip()
                    if 'non rilevato' in value_part.lower():
                        info['mem_picco_kb'] = "non_rilevato"
                    else:
                        info['mem_picco_kb'] = float(value_part.split()[0])
                except (IndexError, ValueError):
                    pass

            elif '|N| (righe) =' in line:
                try:
                    value_part = line.split('=')[1].strip()
                    info['N'] = int(value_part)
                except (IndexError, ValueError):
                    pass

            elif '|M| (colonne) =' in line:
                try:
                    value_part = line.split('=')[1].strip()
                    info['M'] = int(value_part)
                except (IndexError, ValueError):
                    pass
            
            
            elif 'Matrice ridotta |M\'|' in line:
                try:
                    # Estrae M_ridotto dalla riga: "Matrice ridotta |M'| (colonne non vuote) = 33"
                    value_part = line.split('=')[1].strip()
                    info['M_ridotto'] = int(value_part)
                except (IndexError, ValueError):
                    pass
            
            elif 'Origine:' in line:
                try:
                    info['origine'] = line.split(':')[1].strip()
                except (IndexError, ValueError):
                    pass
            
            elif 'Densita:' in line:
                try:
                    info['densita'] = float(line.split(':')[1].strip())
                except (IndexError, ValueError):
                    pass
            
            elif 'CPU time singoli worker' in line:
                try:
                    # Check per nuovo formato (per livello) o vecchio formato (lista piatta)
                    if 'per livello:' in line:
                        # Nuovo formato: worker_times per livello
                        worker_times_per_level = []
                        
                        # Leggi le righe successive con formato "Livello X: [...]"
                        while i < len(lines):
                            next_line = lines[i].strip()
                            i += 1
                            
                            if not next_line.startswith(';;;'):
                                i -= 1  # Torna indietro di una riga
                                break
                            
                            content = next_line[3:].strip()
                            
                            # Check se è una riga di livello
                            if 'Livello' in content and ':' in content:
                                # Estrai numero livello e tempi
                                level_part, times_part = content.split(':', 1)
                                level_num = int(level_part.split()[1])
                                
                                # Estrai i tempi dalla parte [...] s
                                times_str = times_part.replace('s', '').strip().strip('[]')
                                if times_str:
                                    level_times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
                                    # Assicurati che la lista sia abbastanza grande
                                    while len(worker_times_per_level) <= level_num:
                                        worker_times_per_level.append([])
                                    worker_times_per_level[level_num] = level_times
                            else:
                                # Fine della sezione worker
                                i -= 1
                                break
                        
                        info['cpu_worker_times'] = worker_times_per_level if worker_times_per_level else None
                    
                    elif 'non presente' in line or 'esecuzione seriale' in line:
                        # Versione seriale: nessun worker
                        info['cpu_worker_times'] = None
                    
                    else:
                        # Vecchio formato: lista piatta multi-linea
                        worker_times = []
                        value_part = line.split('=')[1].strip()
                        
                        # Se la lista è già completa su una riga
                        if value_part.startswith('[') and ']' in value_part:
                            times_str = value_part.replace('s', '').strip().strip('[]')
                            if times_str:
                                worker_times = [float(t.strip()) for t in times_str.split(',') if t.strip()]
                        else:
                            # Lista su più righe: continua a leggere
                            # Rimuovi '[' iniziale se presente
                            if value_part.startswith('['):
                                value_part = value_part[1:]
                            
                            # Accumula i valori dalla prima riga
                            current_values = value_part.replace('s', '').strip()
                            if current_values and current_values not in ['', ',']:
                                worker_times.extend([float(t.strip()) for t in current_values.split(',') if t.strip()])
                            
                            # Continua a leggere le righe successive
                            while i < len(lines):
                                next_line = lines[i].strip()
                                i += 1
                                
                                if not next_line.startswith(';;;'):
                                    i -= 1  # Torna indietro di una riga
                                    break
                                
                                # Rimuovi ;;; e spazi
                                content = next_line[3:].strip()
                                
                                # Se contiene ']', è l'ultima riga
                                if ']' in content:
                                    content = content.replace(']', '').replace('s', '').strip()
                                    if content and content not in ['', ',']:
                                        worker_times.extend([float(t.strip()) for t in content.split(',') if t.strip()])
                                    break
                                else:
                                    # Riga intermedia
                                    content = content.replace('s', '').strip()
                                    if content and content not in ['', ',']:
                                        worker_times.extend([float(t.strip()) for t in content.split(',') if t.strip()])
                        
                        info['cpu_worker_times'] = worker_times if worker_times else []
                except (IndexError, ValueError) as e:
                    pass
                    
    except Exception:
        print(f"Errore nella lettura del file {filename}.")

    return info

# ===== RACCOLTA E AGGREGAZIONE PERFORMANCE =====

def collect_performance(input_dir, output_json):
    """
    Raccoglie le prestazioni da tutti i file .mhs in una directory e le salva in JSON.
    
    FUNZIONAMENTO:
    1. Verifica esistenza della directory di input
    2. Se output_json esiste già, lo ricarica (evita rielaborazione)
    3. Scandisce ricorsivamente input_dir cercando file con estensione .mhs
    4. Elabora ogni file trovato tramite parse_mhs_file()
    5. Ordina i risultati alfabeticamente per nome file
    6. Salva i risultati aggregati in formato JSON
    
    OTTIMIZZAZIONE:
    Se il file JSON di output esiste già, lo ricarica direttamente invece di 
    rielaborare tutti i file .mhs. Questo permette chiamate ripetute senza overhead.
    Per forzare la rielaborazione, eliminare manualmente il file JSON.
    
    Args:
        input_dir (str): percorso directory contenente i file .mhs da analizzare
                        (la ricerca è ricorsiva nelle sottodirectory)
        output_json (str): percorso del file JSON per salvare i risultati aggregati
                          (directory viene creata automaticamente se non esiste)
        
    Returns:
        list or None: lista di dizionari, uno per ogni file .mhs elaborato,
                     oppure None in caso di errore critico
                     
    Formato output:
        Ogni elemento della lista è un dizionario con struttura:
        {
            'file': 'nome_file.mhs',
            'path': '/percorso/completo/nome_file.mhs',
            'MHS_trovati': 42,
            'completato': True,
            'tempo_reale': 12.34,
            'tempo_cpu': 23.45,
            ... (vedi parse_mhs_file per campi completi)
        }
    
    Effetti collaterali:
        - Crea directory per output_json se non esiste
        - Scrive file JSON su disco
        - Stampa messaggi di stato su stdout
    
    Note:
        - Gestisce gracefully directory vuote (restituisce lista vuota [])
        - Errori di parsing singoli file non bloccano l'elaborazione totale
        - Il file JSON viene scritto con indentazione (indent=4) per leggibilità
    
    Example:
        >>> results = collect_performance('risultati_auto', 'risultati_auto/results.json')
        Raccolte le prestazioni di 43 file .mhs in risultati_auto/results.json
        >>> len(results)
        43
    """
    results = []
    
    if not os.path.exists(input_dir):
        print(f"Errore: la directory {input_dir} non esiste!")
        return None
    
    # Se il file di output esiste già, lo ricarica invece di rielaborare
    if os.path.exists(output_json):
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            print(f"Errore nella lettura del file {output_json}.")

    # Crea la directory per il file di output se necessaria
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            print(f"Errore nella creazione della directory {output_dir}.")
            return None
    
    # Scansione ricorsiva per trovare tutti i file .mhs
    mhs_files = glob.glob(os.path.join(input_dir, "**", "*.mhs"), recursive=True)
    
    if not mhs_files:
        print(f"Attenzione: nessun file .mhs trovato in {input_dir}")
        return []
        
    # Elabora ogni file .mhs trovato
    for file_path in mhs_files:
        try:
            info = parse_mhs_file(file_path)
            results.append(info)
        except Exception:
            print(f"Errore nell'analisi di {file_path}.")
    
    results.sort(key=lambda x: x['file'])
    
    # Salva i risultati in JSON
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"\nRaccolte le prestazioni di {len(results)} file .mhs in {output_json}")
    except Exception:
        print(f"\nErrore nella scrittura del file {output_json}.")
        return None
    
    return results
    
# ===== GENERAZIONE REPORT STATISTICI =====

def print_statistics(results, input_dir=None, output_txt=None):
    """
    Genera e visualizza un report statistico completo dei risultati di performance.
    
    SEZIONI DEL REPORT:
    1. Statistiche globali aggregate:
       - Tasso di completamento complessivo
       - Tempo medio di esecuzione
       - Totale MHS trovati su tutte le matrici
    
    2. Statistiche per gruppo di benchmark (benchmarks1, benchmarks2):
       - Metriche descrittive (media, mediana, moda) per:
         * Tempo reale
         * Tempo CPU
         * Memoria RSS
         * Picco di memoria
       - Conteggi per gruppo
    
    3. Statistiche per categoria (trivial, tiny, small, medium, large, xlarge):
       - Stesse metriche descrittive dei gruppi benchmark
       - Tasso di completamento per categoria
       - Totale MHS per categoria
    
    4. Analisi correlazione densità-prestazioni:
       - Approccio 1: solo matrici completate (elimina bias da timeout)
       - Approccio 2: tutte le matrici (include casi limite)
       - Correlazioni calcolate per: tempo, memoria RSS, picco memoria, MHS
       - Confronto tra i due approcci
    
    5. Statistiche densità per categoria:
       - Range (min-max)
       - Media e mediana
       - Correlazioni specifiche per categoria
    
    6. Statistiche sintetiche per origine (benchmarks1/benchmarks2):
       - Conteggi, completamenti, MHS totali, tempo medio
    
    GESTIONE CATEGORIE:
    Le categorie vengono caricate in ordine di priorità:
    1. Da 'risultati_unificati.json' (se disponibile in input_dir)
    2. Dal campo 'categoria' nei file .mhs individuali
    3. Fallback: categoria 'xlarge' per file senza categoria
    
    Args:
        results (list): lista di dizionari con i risultati da parse_mhs_file()
        input_dir (str, optional): directory di input, usata per cercare
                                  risultati_unificati.json con categorie esatte
        output_txt (str, optional): percorso file di testo per salvare il report.
                                   Se None, stampa solo su stdout
        
    Returns:
        list: lista di stringhe, una per ogni riga del report generato
              (utile per ulteriori elaborazioni o test)
    
    Effetti collaterali:
        - Stampa il report completo su stdout
        - Se output_txt specificato, scrive report su file (crea directory se necessaria)
    
    Note:
        - Gestisce gracefully risultati vuoti (stampa messaggio e restituisce [])
        - I valori "non_rilevato" per picco memoria sono automaticamente filtrati
        - Le categorie vengono processate nell'ordine standard (trivial→xlarge)
        - Calcoli statistici utilizzano solo valori validi (non None)
    
    Examples:
        >>> results = collect_performance('risultati_auto', 'results.json')
        >>> lines = print_statistics(results, 'risultati_auto', 'statistiche.txt')
        >>> print(f"Report di {len(lines)} righe generato")
    """
    if not results:
        print("Nessun risultato da visualizzare.")
        return []
    
    output_lines = []
    
    def add_line(line=""):
        output_lines.append(line)
        print(line)
    
    # Calcola statistiche aggregate globali
    completed = sum(1 for r in results if r['completato'])
    avg_time = sum(r['tempo_reale'] for r in results if r['tempo_reale'] is not None) / len(results)
    total_mhs = sum(r['MHS_trovati'] for r in results)
    
    add_line(f"\nStatistiche Complessive:")
    add_line(f"Completati: {completed}/{len(results)} ({completed/len(results)*100:.1f}%)")
    add_line(f"Tempo medio: {avg_time:.2f}s")
    add_line(f"Totale MHS trovati: {total_mhs}")
    
    # Raggruppa i risultati per tipo di benchmark (basandosi sul percorso del file)
    benchmark_groups = {
        'benchmarks1': [],
        'benchmarks2': [],
        'all': results
    }
    
    for r in results:
        path_parts = r['path'].lower().split(os.sep)
        if 'benchmarks1' in path_parts:
            benchmark_groups['benchmarks1'].append(r)
        elif 'benchmarks2' in path_parts:
            benchmark_groups['benchmarks2'].append(r)
    
    # Raggruppamento per categoria (trivial, tiny, small, medium, large, xlarge)
    category_names = ["trivial", "tiny", "small", "medium", "large", "xlarge"]
    categories = {cat: [] for cat in category_names}
    
    # Tenta di caricare risultati_unificati.json per ottenere le categorie esatte
    unified_results_file = os.path.join(input_dir, 'risultati_unificati.json') if input_dir else None
    if unified_results_file and os.path.exists(unified_results_file):
        try:
            with open(unified_results_file, 'r', encoding='utf-8') as f:
                unified_results = json.load(f)
                
            # Mappa: nome file -> categoria
            file_to_category = {}
            for ur in unified_results:
                file_to_category[os.path.basename(ur['file'])] = ur.get('categoria', '')
                
            # Assegna ciascun risultato alla categoria corrispondente
            for r in results:
                filename = os.path.basename(r['file'])
                categoria = file_to_category.get(filename, '')
                if categoria in categories:
                    categories[categoria].append(r)
                    
            add_line(f"Categorie caricate dal file {unified_results_file}")
        except Exception:
            add_line(f"Attenzione: errore nel caricamento delle categorie da {unified_results_file}.")
            add_line("Uso le categorie dai file .mhs")
    
    # Se non abbiamo categorie da risultati_unificati, le leggiamo direttamente dai results
    # (che ora contengono 'categoria' letta dai file .mhs)
    if not any(categories.values()):
        add_line("Categorie lette direttamente dai file .mhs")
        for r in results:
            categoria = r.get('categoria', 'sconosciuta')
            if categoria in categories:
                categories[categoria].append(r)
            elif categoria == 'sconosciuta':
                # Fallback per file vecchi senza categoria
                categories['xlarge'].append(r)
    
    # Statistiche dettagliate per i gruppi di benchmark
    add_line("\n" + "="*80)
    add_line("STATISTICHE DETTAGLIATE PER GRUPPI DI BENCHMARK")
    add_line("="*80)
    
    for group_name, group_results in benchmark_groups.items():
        if not group_results:
            continue
            
        add_line(f"\n--- Gruppo: {group_name} ({len(group_results)} file) ---")
        
        # Conteggio dei completati
        completed = sum(1 for r in group_results if r['completato'])
        total_mhs_group = sum(r['MHS_trovati'] for r in group_results)
        add_line(f"Completati: {completed}/{len(group_results)} ({completed/len(group_results)*100:.1f}%)")
        add_line(f"Totale MHS trovati: {total_mhs_group}")
        
        # Raccogli i valori per le statistiche
        tempo_reale_values = [r['tempo_reale'] for r in group_results if r['tempo_reale'] is not None]
        tempo_cpu_values = [r['tempo_cpu'] for r in group_results if r['tempo_cpu'] is not None]
        mem_rss_values = [r['mem_rss_kb'] for r in group_results if r['mem_rss_kb'] is not None]
        mem_picco_values = [r['mem_picco_kb'] for r in group_results if r['mem_picco_kb'] is not None and r['mem_picco_kb'] != "non_rilevato"]
        
        # Calcola le statistiche
        stats_tempo_reale = calculate_statistics(tempo_reale_values)
        stats_tempo_cpu = calculate_statistics(tempo_cpu_values)
        stats_mem_rss = calculate_statistics(mem_rss_values)
        stats_mem_picco = calculate_statistics(mem_picco_values)
        
        # Stampa le statistiche
        add_line("\nTempo Reale (secondi):")
        add_line(f"  Media: {stats_tempo_reale['media']:.2f}" if stats_tempo_reale['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_tempo_reale['mediana']:.2f}" if stats_tempo_reale['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_tempo_reale['moda']:.2f}" if stats_tempo_reale['moda'] is not None else "  Moda: N/A")
        
        add_line("\nTempo CPU (secondi):")
        add_line(f"  Media: {stats_tempo_cpu['media']:.2f}" if stats_tempo_cpu['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_tempo_cpu['mediana']:.2f}" if stats_tempo_cpu['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_tempo_cpu['moda']:.2f}" if stats_tempo_cpu['moda'] is not None else "  Moda: N/A")
        
        add_line("\nMemoria RSS (KB):")
        add_line(f"  Media: {stats_mem_rss['media']:.2f}" if stats_mem_rss['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_mem_rss['mediana']:.2f}" if stats_mem_rss['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_mem_rss['moda']:.2f}" if stats_mem_rss['moda'] is not None else "  Moda: N/A")
        
        add_line("\nPicco Memoria (KB):")
        add_line(f"  Media: {stats_mem_picco['media']:.2f}" if stats_mem_picco['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_mem_picco['mediana']:.2f}" if stats_mem_picco['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_mem_picco['moda']:.2f}" if stats_mem_picco['moda'] is not None else "  Moda: N/A")
    
    # Statistiche dettagliate per categoria
    add_line("\n" + "="*80)
    add_line("STATISTICHE DETTAGLIATE PER CATEGORIA")
    add_line("="*80)
    
    for cat_name in category_names:  # Utilizzo le categorie predefinite nell'ordine corretto
        cat_results = categories[cat_name]
        if not cat_results:
            continue
            
        add_line(f"\n--- Categoria: {cat_name} ({len(cat_results)} file) ---")
        
        # Conteggio dei completati
        completed = sum(1 for r in cat_results if r['completato'])
        total_mhs_cat = sum(r['MHS_trovati'] for r in cat_results)
        if len(cat_results) > 0:
            compl_pct = completed / len(cat_results) * 100
        else:
            compl_pct = 0
        add_line(f"Completati: {completed}/{len(cat_results)} ({compl_pct:.1f}%)")
        add_line(f"Totale MHS trovati: {total_mhs_cat}")
        
        # Raccogli i valori per le statistiche
        tempo_reale_values = [r['tempo_reale'] for r in cat_results if r['tempo_reale'] is not None]
        tempo_cpu_values = [r['tempo_cpu'] for r in cat_results if r['tempo_cpu'] is not None]
        mem_rss_values = [r['mem_rss_kb'] for r in cat_results if r['mem_rss_kb'] is not None]
        mem_picco_values = [r['mem_picco_kb'] for r in cat_results if r['mem_picco_kb'] is not None and r['mem_picco_kb'] != "non_rilevato"]
        
        # Calcola le statistiche
        stats_tempo_reale = calculate_statistics(tempo_reale_values)
        stats_tempo_cpu = calculate_statistics(tempo_cpu_values)
        stats_mem_rss = calculate_statistics(mem_rss_values)
        stats_mem_picco = calculate_statistics(mem_picco_values)
        
        # Stampa le statistiche
        add_line("\nTempo Reale (secondi):")
        add_line(f"  Media: {stats_tempo_reale['media']:.2f}" if stats_tempo_reale['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_tempo_reale['mediana']:.2f}" if stats_tempo_reale['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_tempo_reale['moda']:.2f}" if stats_tempo_reale['moda'] is not None else "  Moda: N/A")
        
        add_line("\nTempo CPU (secondi):")
        add_line(f"  Media: {stats_tempo_cpu['media']:.2f}" if stats_tempo_cpu['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_tempo_cpu['mediana']:.2f}" if stats_tempo_cpu['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_tempo_cpu['moda']:.2f}" if stats_tempo_cpu['moda'] is not None else "  Moda: N/A")
        
        add_line("\nMemoria RSS (KB):")
        add_line(f"  Media: {stats_mem_rss['media']:.2f}" if stats_mem_rss['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_mem_rss['mediana']:.2f}" if stats_mem_rss['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_mem_rss['moda']:.2f}" if stats_mem_rss['moda'] is not None else "  Moda: N/A")
        
        add_line("\nPicco Memoria (KB):")
        add_line(f"  Media: {stats_mem_picco['media']:.2f}" if stats_mem_picco['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_mem_picco['mediana']:.2f}" if stats_mem_picco['mediana'] is not None else "  Mediana: N/A")
        add_line(f"  Moda: {stats_mem_picco['moda']:.2f}" if stats_mem_picco['moda'] is not None else "  Moda: N/A")
    
    # Analisi correlazione densità-prestazioni
    add_line("\n" + "="*80)
    add_line("ANALISI CORRELAZIONE DENSITÀ-PRESTAZIONI")
    add_line("="*80)
    
    # Statistiche densità globali
    densita_values = [r['densita'] for r in results if r['densita'] is not None]
    if densita_values:
        stats_densita = calculate_statistics(densita_values)
        add_line("\nStatistiche Densità Globali:")
        add_line(f"  Range: {min(densita_values):.4f} - {max(densita_values):.4f}")
        add_line(f"  Media: {stats_densita['media']:.4f}" if stats_densita['media'] is not None else "  Media: N/A")
        add_line(f"  Mediana: {stats_densita['mediana']:.4f}" if stats_densita['mediana'] is not None else "  Mediana: N/A")
    
    # Correlazioni densità-prestazioni: doppio approccio
    add_line("\n" + "-"*60)
    add_line("CORRELAZIONI DENSITÀ-PRESTAZIONI")
    add_line("-"*60)
    
    # Approccio 1: Solo matrici completate
    completed_results = [r for r in results if r['completato']]
    densita_completed = [r['densita'] for r in completed_results if r['densita'] is not None]
    
    add_line(f"\nApproccio 1: Solo matrici completate ({len(completed_results)} istanze)")
    
    if densita_completed:
        # Tempo Reale - filtra solo istanze con entrambe le metriche valide
        tempo_completed = [r['tempo_reale'] for r in completed_results if r['densita'] is not None and r['tempo_reale'] is not None]
        densita_tempo_compl = [r['densita'] for r in completed_results if r['densita'] is not None and r['tempo_reale'] is not None]
        corr_tempo_compl = calculate_correlation(densita_tempo_compl, tempo_completed)
        add_line(f"  Tempo Reale:     {corr_tempo_compl:.3f}" if corr_tempo_compl is not None else "  Tempo Reale:     N/A")
        
        # Memoria RSS - filtra solo istanze con entrambe le metriche valide
        mem_rss_completed = [r['mem_rss_kb'] for r in completed_results if r['densita'] is not None and r['mem_rss_kb'] is not None]
        densita_rss_compl = [r['densita'] for r in completed_results if r['densita'] is not None and r['mem_rss_kb'] is not None]
        corr_rss_compl = calculate_correlation(densita_rss_compl, mem_rss_completed)
        add_line(f"  Memoria RSS:     {corr_rss_compl:.3f}" if corr_rss_compl is not None else "  Memoria RSS:     N/A")
        
        # Picco Memoria - filtra solo istanze con entrambe le metriche valide
        mem_picco_completed = [r['mem_picco_kb'] for r in completed_results if r['densita'] is not None and r['mem_picco_kb'] is not None and r['mem_picco_kb'] != "non_rilevato"]
        densita_picco_compl = [r['densita'] for r in completed_results if r['densita'] is not None and r['mem_picco_kb'] is not None and r['mem_picco_kb'] != "non_rilevato"]
        corr_picco_compl = calculate_correlation(densita_picco_compl, mem_picco_completed)
        add_line(f"  Picco Memoria:   {corr_picco_compl:.3f}" if corr_picco_compl is not None else "  Picco Memoria:   N/A")
        
        # MHS Trovati - filtra solo istanze con entrambe le metriche valide
        mhs_completed = [r['MHS_trovati'] for r in completed_results if r['densita'] is not None]
        densita_mhs_compl = [r['densita'] for r in completed_results if r['densita'] is not None]
        corr_mhs_compl = calculate_correlation(densita_mhs_compl, mhs_completed)
        add_line(f"  MHS Trovati:     {corr_mhs_compl:.3f}" if corr_mhs_compl is not None else "  MHS Trovati:     N/A")
    
    # Approccio 2: Tutte le matrici
    tempo_values = [r['tempo_reale'] for r in results if r['densita'] is not None and r['tempo_reale'] is not None]
    densita_tempo_all = [r['densita'] for r in results if r['densita'] is not None and r['tempo_reale'] is not None]
    
    add_line(f"\nApproccio 2: Tutte le matrici ({len(results)} istanze)")
    
    corr_tempo = calculate_correlation(densita_tempo_all, tempo_values)
    add_line(f"  Tempo Reale:     {corr_tempo:.3f}" if corr_tempo is not None else "  Tempo Reale:     N/A")
    
    mem_rss_values = [r['mem_rss_kb'] for r in results if r['densita'] is not None and r['mem_rss_kb'] is not None]
    densita_rss_all = [r['densita'] for r in results if r['densita'] is not None and r['mem_rss_kb'] is not None]
    corr_mem_rss = calculate_correlation(densita_rss_all, mem_rss_values)
    add_line(f"  Memoria RSS:     {corr_mem_rss:.3f}" if corr_mem_rss is not None else "  Memoria RSS:     N/A")
    
    mem_picco_values = [r['mem_picco_kb'] for r in results if r['densita'] is not None and r['mem_picco_kb'] is not None and r['mem_picco_kb'] != "non_rilevato"]
    densita_picco_all = [r['densita'] for r in results if r['densita'] is not None and r['mem_picco_kb'] is not None and r['mem_picco_kb'] != "non_rilevato"]
    corr_mem_picco = calculate_correlation(densita_picco_all, mem_picco_values)
    add_line(f"  Picco Memoria:   {corr_mem_picco:.3f}" if corr_mem_picco is not None else "  Picco Memoria:   N/A")
    
    mhs_values = [r['MHS_trovati'] for r in results if r['densita'] is not None]
    densita_mhs_all = [r['densita'] for r in results if r['densita'] is not None]
    corr_mhs = calculate_correlation(densita_mhs_all, mhs_values)
    add_line(f"  MHS Trovati:     {corr_mhs:.3f}" if corr_mhs is not None else "  MHS Trovati:     N/A")
    
    # Confronto tra i due approcci
    if densita_completed and corr_tempo_compl is not None and corr_tempo is not None:
        add_line(f"\nConfronto tra approcci:")
        add_line(f"  Diff. Tempo:     {abs(corr_tempo_compl - corr_tempo):.3f}")
        add_line(f"  Diff. RSS:       {abs(corr_rss_compl - corr_mem_rss):.3f}" if corr_rss_compl is not None and corr_mem_rss is not None else "  Diff. RSS:       N/A")
        add_line(f"  Diff. Picco:     {abs(corr_picco_compl - corr_mem_picco):.3f}" if corr_picco_compl is not None and corr_mem_picco is not None else "  Diff. Picco:     N/A")
        add_line(f"  Diff. MHS:       {abs(corr_mhs_compl - corr_mhs):.3f}" if corr_mhs_compl is not None and corr_mhs is not None else "  Diff. MHS:       N/A")
    
    # Analisi densità per categoria
    add_line("\n" + "-"*60)
    add_line("STATISTICHE DENSITÀ PER CATEGORIA")
    add_line("-"*60)
    
    for cat_name in category_names:
        cat_results = categories[cat_name]
        if not cat_results:
            continue
            
        cat_densita_values = [r['densita'] for r in cat_results if r['densita'] is not None]
        if cat_densita_values:
            stats_cat_densita = calculate_statistics(cat_densita_values)
            add_line(f"\nCategoria {cat_name}:")
            add_line(f"  Range: {min(cat_densita_values):.4f} - {max(cat_densita_values):.4f}")
            add_line(f"  Media: {stats_cat_densita['media']:.4f}" if stats_cat_densita['media'] is not None else "  Media: N/A")
            add_line(f"  Mediana: {stats_cat_densita['mediana']:.4f}" if stats_cat_densita['mediana'] is not None else "  Mediana: N/A")
            
            # Correlazioni specifiche per categoria
            cat_tempo_values = [r['tempo_reale'] for r in cat_results]
            cat_corr_tempo = calculate_correlation(cat_densita_values, cat_tempo_values)
            if cat_corr_tempo is not None:
                add_line(f"  Corr. Tempo: {cat_corr_tempo:.3f}")
    
    # Statistiche aggregate per origine (benchmarks1 vs benchmarks2)
    add_line("\n" + "="*80)
    add_line("STATISTICHE SINTETICHE PER ORIGINE")
    add_line("="*80)
    origin_stats = defaultdict(lambda: {'count': 0, 'completed': 0, 'mhs': 0, 'time': 0.0})
    
    for r in results:
        # Usa il campo 'origine' se disponibile, altrimenti estrai dalla path
        origin = r.get('origine', None)
        if not origin:
            # Fallback: cerca 'benchmarks1' o 'benchmarks2' nel percorso
            path_lower = r['path'].lower()
            if 'benchmarks1' in path_lower:
                origin = 'benchmarks1'
            elif 'benchmarks2' in path_lower:
                origin = 'benchmarks2'
            else:
                origin = 'sconosciuta'
            
        origin_stats[origin]['count'] += 1
        if r['completato']:
            origin_stats[origin]['completed'] += 1
        origin_stats[origin]['mhs'] += r['MHS_trovati']
        if r['tempo_reale'] is not None:
            origin_stats[origin]['time'] += r['tempo_reale']
    
    # Ordina per nome origine per output consistente
    for origin in sorted(origin_stats.keys()):
        stats = origin_stats[origin]
        if stats['count'] > 0:
            avg_time = stats['time'] / stats['count']
            compl_pct = stats['completed'] / stats['count'] * 100
            add_line(f"- {origin}: {stats['count']} file, {compl_pct:.1f}% completati, "
                f"{stats['mhs']} MHS trovati, tempo medio {avg_time:.2f}s")
    
    # Salva le statistiche in un file di testo se richiesto
    if output_txt:
        try:
            output_dir = os.path.dirname(output_txt)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_txt, 'w', encoding='utf-8') as f:
                for line in output_lines:
                    f.write(line + '\n')
            
            add_line(f"\nStatistiche salvate in {output_txt}")
        except Exception:
            add_line(f"\nErrore nel salvataggio delle statistiche in {output_txt}.")

    return output_lines


# ===== INTERFACCE PUBBLICHE =====

def statistiche(input_dir, output_json, output_txt=None):
    """
    Funzione wrapper ad alto livello per raccogliere e visualizzare statistiche complete.
    
    Questa è la funzione principale per uso programmatico dello script.
    Combina le operazioni di:
    1. Raccolta dati dai file .mhs (collect_performance)
    2. Generazione report statistico (print_statistics)
    
    È equivalente a chiamare lo script da riga di comando ma utilizzabile
    da altri moduli Python.
    
    Args:
        input_dir (str): directory contenente i file .mhs da analizzare
        output_json (str): percorso file JSON per salvare i dati strutturati
        output_txt (str, optional): percorso file TXT per salvare il report leggibile
                                   Se None, il report viene solo stampato su stdout
        
    Returns:
        None
    
    Effetti collaterali:
        - Legge tutti i file .mhs in input_dir (ricorsivamente)
        - Scrive output_json con i dati strutturati
        - Scrive output_txt con il report (se specificato)
        - Stampa report su stdout
        - Stampa messaggi di errore in caso di problemi
    
    Gestione errori:
        In caso di eccezioni durante l'elaborazione, stampa un messaggio
        di errore generico e continua (non solleva eccezioni).
    
    Examples:
        >>> # Uso base: JSON e report testuale
        >>> statistiche('risultati_auto', 'results.json', 'statistiche.txt')
        
        >>> # Solo JSON, report su stdout
        >>> statistiche('risultati_serial', 'results.json')
        
        >>> # Da altri script
        >>> from collector_performance import statistiche
        >>> statistiche('risultati_parallel', 'performance.json', 'report.txt')
    
    Note:
        - Se output_json esiste già, i dati vengono ricaricati senza rielaborazione
        - Per forzare rielaborazione, eliminare manualmente output_json
        - Il report testuale contiene analisi molto più dettagliate rispetto al JSON
    """
    try:
        results = collect_performance(input_dir, output_json)
        if results:
            print_statistics(results, input_dir, output_txt)
    except Exception:
        print(f"Errore durante la raccolta o l'analisi delle prestazioni.")


# ===== ENTRY POINT =====

def main():
    """
    Funzione main per esecuzione da riga di comando.
    
    SINTASSI:
        python collector_performance.py [dir_input] [output.json]
    
    PARAMETRI:
        dir_input (opzionale): directory contenente i file .mhs
                              Default: 'risultati'
        
        output.json (opzionale): percorso file JSON di output
                                Default: 'dir_input/results.json'
    
    COMPORTAMENTO:
        Esegue SOLO la raccolta dei dati (collect_performance), senza generare
        il report statistico dettagliato. Per ottenere anche il report, usare:
        
        >>> from collector_performance import statistiche
        >>> statistiche('input_dir', 'output.json', 'report.txt')
    
    ESEMPI:
        # Usa default (risultati/results.json)
        python collector_performance.py
        
        # Directory custom, JSON custom
        python collector_performance.py risultati_auto performance.json
        
        # Solo directory custom (JSON: risultati_auto/results.json)
        python collector_performance.py risultati_auto
    
    Note:
        - Questa funzione viene chiamata solo se lo script è eseguito direttamente
        - Per uso programmatico da altri moduli, preferire statistiche()
        - Errori critici vengono stampati su stdout ma non sollevano eccezioni
    """
    # Parsing degli argomenti da riga di comando
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = 'risultati'
        
    if len(sys.argv) > 2:
        output_json = sys.argv[2]
    else:
        output_json = os.path.join(input_dir, 'results.json')
    
    # Esegue solo la raccolta dei dati, senza stampare statistiche
    try:
        collect_performance(input_dir, output_json)
    except Exception:
        print(f"Errore durante la raccolta delle prestazioni.")

if __name__ == "__main__":
    main()