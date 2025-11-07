# Progetto algoritmi e strutture dati

Progetto per il corso di Algoritmi e Strutture Dati - Implementazione di algoritmi per il calcolo del Minimal Hitting Set (MHS).

## Autrici

- Ester Adinolfi, matricola 723378
- Nicole Facchetti, matricola 731029

## Descrizione

Questo progetto implementa algoritmi per il calcolo del Minimal Hitting Set su matrici binarie, con particolare attenzione alle prestazioni e all'ottimizzazione. Sono state sviluppate due versioni dell'algoritmo (seriale e parallela) e condotte analisi sperimentali approfondite sui risultati ottenuti.

## Struttura del progetto

### `con_risultati/`

Contiene il codice sorgente completo e tutti i risultati sperimentali ottenuti durante l'analisi:

- **Codice sorgente**: implementazioni degli algoritmi MHS in diverse varianti
- **`benchmarks/`**: matrici di test utilizzate per la sperimentazione
- **`risultati_serial/`**: risultati dell'esecuzione in modalità seriale
- **`risultati_parallel/`**: risultati dell'esecuzione in modalità parallela
- **`risultati_auto/`**: risultati dell'esecuzione in modalità automatica
- **`selezionate/`**: subset di matrici selezionate per l'analisi

Ogni cartella di risultati contiene:
- `results.json`: dati completi delle esecuzioni
- `statistiche_prestazioni.txt`: statistiche aggregate
- Sottocartelle con i risultati dettagliati per ogni benchmark (i file .mhs)

### `da_eseguire/`

Contiene esclusivamente il codice sorgente, senza risultati pre-generati. Questa cartella è destinata all'esecuzione e testing del progetto:

- Stessa struttura di codice della cartella `con_risultati/`
- Include le matrici di benchmark necessarie per i test
- Non contiene file di risultati pre-esistenti

### `relazione/`

Contiene la documentazione completa del progetto e i file sorgente:

- **`relazione.tex`**: file principale della relazione in LaTeX
- **`Capitoli/`**: capitoli della relazione (introduzione, algoritmo, interfaccia, sperimentazione, conclusioni)
- **`Immagini/`**: grafici e figure generati per l'analisi
- **`scripts/`**: script Python per la generazione automatica dei grafici (`generate_plots.py`)

## Requisiti

- Python 3.12.4
- Librerie Python necessarie (installabili tramite `pip`):
  - numpy
  - pandas
  - matplotlib
  - seaborn

## Installazione

1. Clonare il repository
2. Installare le dipendenze:

```bash
pip install numpy pandas matplotlib seaborn
```

## Esecuzione

### Interfaccia testuale (consigliato)

Il progetto include un'interfaccia testuale interattiva che permette di accedere a tutte le funzionalità in modo guidato.

Per avviare l'interfaccia, posizionarsi nella cartella `da_eseguire/` ed eseguire:

```bash
python menu.py
```

L'interfaccia permette di:
- Eseguire l'algoritmo MHS in diverse modalità
- Selezionare matrici di benchmark
- Visualizzare e analizzare i risultati
- Configurare i parametri di esecuzione

L'interfaccia include anche un help, raggiungibile tramite il menu principale.

### Esecuzione diretta

Per maggiori dettagli sui comandi disponibili e le opzioni di esecuzione avanzate, è possibile consultare la relazione nella cartella `relazione/`.

## Generazione dei grafici

Per rigenerare i grafici utilizzati nella relazione, posizionarsi nella cartella `relazione/scripts/` ed eseguire:

```bash
# Genera grafici usati nella relazione 
python generate_plots.py used --log_scale
```

## Note

- I risultati sperimentali completi sono disponibili nella cartella `con_risultati/`
- Per testing e nuove esecuzioni, utilizzare la cartella `da_eseguire/`
- La relazione completa in formato PDF è già presente nella cartella `relazione/`, ma può essere rigenerata compilando i file LaTeX.

