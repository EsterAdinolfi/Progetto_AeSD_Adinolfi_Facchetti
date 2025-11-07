#!/usr/bin/env python3
# =============================================================================
# matrices_selection.py - Selezione automatica delle matrici di test
# =============================================================================
"""
Script per la selezione bilanciata di matrici di test dal dataset di benchmark.

Lo script:
1. Crea un catalogo completo di tutte le matrici in benchmarks1 e benchmarks2
2. Seleziona 43 matrici uniche (per nome E contenuto) bilanciate per categoria
3. Copia le matrici selezionate nella cartella 'selezionate'

Output:
    - catalog.json: tutte le matrici analizzate con statistiche
    - selection.json: matrici selezionate per i test
    - selezionate/: cartella con le matrici copiate

Utilizzo:
    python matrices_selection.py
"""

import os, json, random, shutil, hashlib, sys

random.seed(42)  # Seme fisso per riproducibilità

SRC_DIRS = ["benchmarks/benchmarks1", "benchmarks/benchmarks2"]
DEST_DIR = "selezionate"
os.makedirs(DEST_DIR, exist_ok=True)

def read_matrix(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith(";;;"):
                continue
            if "-" in s:
                s = s.split("-")[0].strip()
            toks = s.split()
            if toks:
                rows.append([int(t) for t in toks])
    return rows

def compute_stats(path):
    rows = read_matrix(path)
    if not rows:
        return None
    N = len(rows)
    M = len(rows[0])
    col_sums = [sum(r[j] for r in rows) for j in range(M)]
    M_ridotto = sum(1 for c in col_sums if c > 0)
    ones = sum(col_sums)
    densita = ones / (N*M) if N*M else 0
    return N, M, M_ridotto, densita

def categorize(N, M, M_ridotto, densita):
    if N <= 1 or M_ridotto <= 1:
        return "trivial"
    if N <= 3 and M_ridotto <= 15:
        return "tiny"
    if N <= 5 and M_ridotto <= 30:
        return "small"
    if N <= 6 and M_ridotto <= 50:
        return "medium"
    if N <= 8 and M_ridotto <= 90:
        return "large"
    return "xlarge"

def select_balanced_from_category(items, target, selected, selected_hashes, selected_names):
    """
    Seleziona un numero target di matrici da una categoria, bilanciando tra cartelle.
    
    Args:
        items (list): Matrici disponibili per questa categoria
        target (int): Numero di matrici da selezionare
        selected (list): Matrici già selezionate (aggiornato in-place)
        selected_hashes (set): Hash già selezionati (aggiornato in-place)
        selected_names (set): Nomi già selezionati (aggiornato in-place)
    
    Returns:
        list: Matrici selezionate per questa categoria
    """
    # Filtra matrici già selezionate (stesso hash o nome)
    available = [item for item in items 
                if item["hash"] not in selected_hashes 
                and item["file"] not in selected_names]
    
    if not available:
        return []
    
    # Bilancia selezione tra cartelle
    b1_available = [item for item in available if item["origine"] == "benchmarks1"]
    b2_available = [item for item in available if item["origine"] == "benchmarks2"]
    
    b1_selected_count = sum(1 for s in selected if s["origine"] == "benchmarks1")
    b2_selected_count = sum(1 for s in selected if s["origine"] == "benchmarks2")
    
    # Preferisci cartella meno rappresentata
    if b1_selected_count <= b2_selected_count and b1_available:
        # Prendi prima da benchmarks1
        num_from_b1 = min(len(b1_available), target)
        chosen_from_b1 = random.sample(b1_available, num_from_b1)
        remaining = target - num_from_b1
        chosen_from_b2 = random.sample(b2_available, min(len(b2_available), remaining)) if remaining > 0 else []
    else:
        # Prendi prima da benchmarks2
        num_from_b2 = min(len(b2_available), target)
        chosen_from_b2 = random.sample(b2_available, num_from_b2)
        remaining = target - num_from_b2
        chosen_from_b1 = random.sample(b1_available, min(len(b1_available), remaining)) if remaining > 0 else []
    
    chosen = chosen_from_b1 + chosen_from_b2
    
    # Aggiorna set selezionati
    for item in chosen:
        selected_hashes.add(item["hash"])
        selected_names.add(item["file"])
    
    return chosen

if __name__ == "__main__":
    try:
        # --- 1. Creazione catalogo completo ---
        print("Creazione catalogo completo...")
        catalog = []

        for src in SRC_DIRS:
            print(f"  Analizzo {src}...")
            files = [f for f in os.listdir(src) if f.endswith(".matrix")]
            for idx, fname in enumerate(files, 1):
                path = os.path.join(src, fname)
                print(f"\r    [{idx}/{len(files)}] {fname:<40}", end="", flush=True)
                try:
                    with open(path, "rb") as f:
                        filehash = hashlib.md5(f.read()).hexdigest()
                    stats = compute_stats(path)
                    if not stats:
                        continue
                    N, M, M_ridotto, densita = stats
                    cat = categorize(N, M, M_ridotto, densita)
                    catalog.append({
                        "file": fname,
                        "path": path,
                        "origine": os.path.basename(src),
                        "N": N, "M": M, "M_ridotto": M_ridotto, "densita": densita,
                        "categoria": cat,
                        "hash": filehash
                    })
                except Exception:
                    print(f"\n    ERRORE!")
            print()  # Nuova linea dopo il progress

        print(f"  Catalogo creato: {len(catalog)} matrici")

        # --- 2. Selezione 43 matrici uniche ---
        print("\nSelezione matrici uniche...")

        categories = ["trivial", "tiny", "small", "medium", "large", "xlarge"]
        target_per_category = {
            "trivial": 3, "tiny": 10, "small": 10, "medium": 10, "large": 5, "xlarge": 5
        }

        selected = []
        selected_hashes = set()
        selected_names = set()

        for cat in categories:
            items = [c for c in catalog if c["categoria"] == cat]
            target = target_per_category[cat]
            
            chosen = select_balanced_from_category(items, target, selected, selected_hashes, selected_names)
            
            selected.extend(chosen)
            
            # Statistiche
            m_list = [c["M_ridotto"] for c in chosen]
            avg_m = sum(m_list) / len(m_list) if m_list else 0
            print(f"  {cat}: {len(chosen)} matrici (M_ridotto avg: {avg_m:.1f})")

        # --- 3. Copia file selezionati ---
        print(f"\nCopia {len(selected)} matrici in {DEST_DIR}...")
        for item in selected:
            dest_sub = os.path.join(DEST_DIR, item["origine"])
            os.makedirs(dest_sub, exist_ok=True)
            shutil.copy(item["path"], os.path.join(dest_sub, item["file"]))

        # --- 4. Salva output ---
        with open("catalog.json", "w") as f:
            json.dump(catalog, f, indent=2)
        with open("selection.json", "w") as f:
            json.dump(selected, f, indent=2)

        # --- Riepilogo ---
        print(f"\n{'='*50}")
        print("RIASSUNTO SELEZIONE")
        print(f"{'='*50}")
        print(f"Totale matrici selezionate: {len(selected)}")
        print(f"Hash unici: {len(selected_hashes)}")
        print(f"Nomi unici: {len(selected_names)}")

        origin_count = {}
        for item in selected:
            origin_count[item["origine"]] = origin_count.get(item["origine"], 0) + 1

        print("\nDistribuzione per origine:")
        for origin in sorted(origin_count):
            print(f"  {origin}: {origin_count[origin]} matrici")

        print(f"\nSelezione completata! File in {DEST_DIR}/")
        print(f"{'='*50}")
    except KeyboardInterrupt:
        sys.exit(1)
