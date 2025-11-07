import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

# Set seaborn style for better aesthetics
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# Paths
ROOT = Path(__file__).resolve().parents[2]

# Default paths (can be overridden by command line args)
DEFAULT_RESULTS_DIR = ROOT / 'NEW' / 'risultati_auto'
DEFAULT_JSON = DEFAULT_RESULTS_DIR / 'results.json'
DEFAULT_TXT = DEFAULT_RESULTS_DIR / 'statistiche_prestazioni.txt'

# Output directory for plots
OUT_DIR = ROOT / 'Relazione' / 'Immagini'
OUT_DIR.mkdir(exist_ok=True)

# Category order for consistent plotting
CATEGORY_ORDER = ['trivial', 'tiny', 'small', 'medium', 'large', 'xlarge']

def load_and_prepare_data(json_path, cpu_metric='total'):
    """Load and prepare data from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    # Add computed CPU worker metrics
    if 'cpu_worker_times' in df.columns:
        # Funzione helper per flattare lista di liste (nuovo formato) o usare lista piatta (vecchio formato)
        def flatten_worker_times(x):
            if x is None:
                return []
            if isinstance(x, list):
                if x and isinstance(x[0], list):
                    # Nuovo formato: lista di liste per livello -> flatten
                    return [time for level in x for time in level]
                else:
                    # Vecchio formato: lista piatta
                    return x
            return []
        
        df['cpu_worker_sum'] = df['cpu_worker_times'].apply(lambda x: sum(flatten_worker_times(x)))
        df['cpu_worker_mean'] = df['cpu_worker_times'].apply(lambda x: sum(flatten_worker_times(x))/len(flatten_worker_times(x)) if flatten_worker_times(x) else 0)
        df['cpu_worker_max'] = df['cpu_worker_times'].apply(lambda x: max(flatten_worker_times(x)) if flatten_worker_times(x) else 0)
    
    # Add selected CPU time column
    if cpu_metric == 'total':
        df['cpu_time_selected'] = df['tempo_cpu']
    elif cpu_metric == 'sum':
        df['cpu_time_selected'] = df['cpu_worker_sum']
    elif cpu_metric == 'mean':
        df['cpu_time_selected'] = df['cpu_worker_mean']
    elif cpu_metric == 'max':
        df['cpu_time_selected'] = df['cpu_worker_max']
    else:
        df['cpu_time_selected'] = df['tempo_cpu']  # fallback
    
    return df

def generate_plots_for_mode(results_dir_name, cpu_metric='total'):
    """Generate plots for a single results directory"""
    results_json = ROOT / 'NEW' / results_dir_name / 'results.json'
    stats_txt = ROOT / 'NEW' / results_dir_name / 'statistiche_prestazioni.txt'
    
    if not results_json.exists():
        print(f"Errore: file {results_json} non trovato!")
        return
    
    df = load_and_prepare_data(results_json, cpu_metric)
    print(f"Caricato {len(df)} risultati da {results_dir_name} (metrica CPU: {cpu_metric})")
    
    # Determine prefix
    if "auto" in results_dir_name.lower():
        prefix = f"auto_{cpu_metric}_"
    elif "serial" in results_dir_name.lower():
        prefix = f"serial_{cpu_metric}_"
    else:
        prefix = f"{cpu_metric}_"
    
    # Generate standard plots
    generate_standard_plots(df, prefix, cpu_metric)
    
    # Generate density plots if available
    if 'densita' in df.columns:
        generate_density_plots(df, prefix, cpu_metric)
    
    print(f"Grafici per {results_dir_name} (metrica CPU: {cpu_metric}) generati in: {OUT_DIR}")

def generate_comparison_plots(serial_dir_name, parallel_dir_name, cpu_metric='total'):
    """Generate comparison plots between serial and parallel results"""
    serial_json = ROOT / 'NEW' / serial_dir_name / 'results.json'
    parallel_json = ROOT / 'NEW' / parallel_dir_name / 'results.json'
    
    if not serial_json.exists():
        print(f"Errore: file {serial_json} non trovato!")
        return
    if not parallel_json.exists():
        print(f"Errore: file {parallel_json} non trovato!")
        return
    
    df_serial = load_and_prepare_data(serial_json, cpu_metric)
    df_parallel = load_and_prepare_data(parallel_json, cpu_metric)
    print(f"Confronto caricato: {len(df_serial)} risultati seriali, {len(df_parallel)} risultati paralleli (metrica CPU: {cpu_metric})")
    
    prefix = f"confr_{cpu_metric}_"
    
    # Generate comparison plots
    generate_comparison_plots_code(df_serial, df_parallel, prefix, cpu_metric)
    
    print(f"Grafici di confronto (metrica CPU: {cpu_metric}) generati in: {OUT_DIR}")

def generate_standard_plots(df, prefix, cpu_metric='total'):
    """Generate standard plots for a single dataset"""
    # Standard figure size for all plots
    FIG_SIZE = (16, 10)
    
    # Completion by category
    completion = df.groupby('categoria', observed=False).apply(lambda g: pd.Series({
        'count': len(g),
        'completed': int(g['completato'].sum())
    }), include_groups=False).reset_index()
    completion['pct'] = completion['completed'] / completion['count'] * 100
    completion = completion.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
    # Fill NaN with 0 for categories with no data
    completion['pct'] = completion['pct'].fillna(0.0)

    plt.figure(figsize=FIG_SIZE)
    bars = plt.bar(completion['categoria'], completion['pct'], color=sns.color_palette('pastel')[9], edgecolor='black', linewidth=1)
    plt.ylabel('Percentuale completamento (%)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Percentuale di completamento per categoria', fontsize=22, pad=16)
    plt.xticks(ha='right', fontsize=20)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars, completion['pct']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}completion_by_category.png', dpi=300)
    plt.close()

    # Time by category
    time_stats = df.groupby('categoria', observed=False)['tempo_reale'].agg(['mean','median','count']).reset_index()
    time_stats = time_stats.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
    # Fill NaN with 0 for categories with no data
    time_stats['mean'] = time_stats['mean'].fillna(0.0)
    time_stats['median'] = time_stats['median'].fillna(0.0)
    
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(time_stats))
    bars1 = plt.bar(x - 0.2, time_stats['mean'], width=0.4, label='Mean', color=sns.color_palette('pastel')[1], edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + 0.2, time_stats['median'], width=0.4, label='Median', color=sns.color_palette('pastel')[2], edgecolor='black', linewidth=1)
    plt.xticks(x, time_stats['categoria'],   ha='right', fontsize=20)
    plt.ylabel('Tempo reale (s)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Tempo reale medio e mediano per categoria', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars1, time_stats['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    for bar, value in zip(bars2, time_stats['median']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}time_by_category.png', dpi=300)
    plt.close()

    # CPU Time by category (using selected metric)
    if 'cpu_time_selected' in df.columns:
        cpu_stats = df.groupby('categoria', observed=False)['cpu_time_selected'].agg(['mean','median','count']).reset_index()
        cpu_stats = cpu_stats.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
        # Fill NaN with 0 for categories with no data (instead of filtering)
        cpu_stats['mean'] = cpu_stats['mean'].fillna(0.0)
        cpu_stats['median'] = cpu_stats['median'].fillna(0.0)
        
        plt.figure(figsize=FIG_SIZE)
        x = np.arange(len(cpu_stats))
        bars1 = plt.bar(x - 0.2, cpu_stats['mean'], width=0.4, label='Mean', color=sns.color_palette('pastel')[5], edgecolor='black', linewidth=1)
        bars2 = plt.bar(x + 0.2, cpu_stats['median'], width=0.4, label='Median', color=sns.color_palette('pastel')[6], edgecolor='black', linewidth=1)
        plt.xticks(x, cpu_stats['categoria'],   ha='right', fontsize=20)
        plt.ylabel(f'CPU Time ({cpu_metric}) (s)', fontsize=20)
        plt.xlabel('Categoria', fontsize=20)
        plt.title(f'CPU Time ({cpu_metric}) medio e mediano per categoria', fontsize=22, pad=16)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars1, cpu_stats['mean']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        for bar, value in zip(bars2, cpu_stats['median']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        plt.savefig(OUT_DIR / f'{prefix}cpu_time_by_category.png', dpi=300)
        plt.close()

    # MHS vs columns reduced
    plt.figure(figsize=FIG_SIZE)
    scatter = plt.scatter(df['M_ridotto'], df['MHS_trovati'], 
                         c=df['MHS_trovati'], cmap='Blues', 
                         alpha=0.8, edgecolors='darkblue', linewidth=1.0, s=60)
    plt.xlabel("Numero di colonne ridotte (M')", fontsize=20)
    plt.ylabel('Numero di MHS trovati', fontsize=20)
    plt.title("MHS trovati vs colonne ridotte", fontsize=22, pad=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{prefix}mhs_vs_cols.png', dpi=300)
    plt.close()

    # Memory stats by category
    mem_stats = df.groupby('categoria', observed=False).agg({
        'mem_rss_kb':'mean',
        'mem_picco_kb':'mean'
    }).reset_index()
    mem_stats = mem_stats.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
    # Fill NaN with 0 for categories with no data
    mem_stats['mem_rss_kb'] = mem_stats['mem_rss_kb'].fillna(0.0)
    mem_stats['mem_picco_kb'] = mem_stats['mem_picco_kb'].fillna(0.0)
    
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(mem_stats))
    bars1 = plt.bar(x - 0.15, mem_stats['mem_rss_kb']/1024, width=0.3, label='RSS mean (MB)', color=sns.color_palette('pastel')[3], edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + 0.15, mem_stats['mem_picco_kb']/1024, width=0.3, label='Peak mean (MB)', color=sns.color_palette('pastel')[4], edgecolor='black', linewidth=1)
    plt.xticks(x, mem_stats['categoria'],   ha='right', fontsize=20)
    plt.ylabel('Memoria (MB)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Statistiche di memoria (media) per categoria', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    for bar, value in zip(bars1, mem_stats['mem_rss_kb']/1024):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    for bar, value in zip(bars2, mem_stats['mem_picco_kb']/1024):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}memory_stats.png', dpi=300)
    plt.close()

def generate_density_plots(df, prefix, cpu_metric='total'):
    """Generate density analysis plots"""
    print("Generando grafici analisi densità...")
    
    # Generate plots for both completed only and all matrices
    generate_density_plots_subset(df[df['completato'] == True].copy(), prefix, cpu_metric, suffix='', 
                                   title_suffix='(solo matrici completate)')
    generate_density_plots_subset(df.copy(), prefix, cpu_metric, suffix='_all', 
                                   title_suffix='(tutte le matrici)')

def generate_density_plots_subset(df_subset, prefix, cpu_metric='total', suffix='', title_suffix=''):
    """Generate density analysis plots for a subset of data"""
    
    # Standard figure size for all plots
    FIG_SIZE = (16, 10)
    
    if len(df_subset) > 0 and df_subset['densita'].notna().any():
        # Density vs Time
        plt.figure(figsize=FIG_SIZE)
        valid_data = df_subset.dropna(subset=['densita', 'tempo_reale'])
        if len(valid_data) > 1:
            scatter = plt.scatter(valid_data['densita'], valid_data['tempo_reale'], 
                                 c=valid_data['tempo_reale'], cmap='viridis', 
                                 alpha=0.7, edgecolors='darkblue', linewidth=0.5, s=50)
            if len(valid_data) >= 3:
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data['densita'], valid_data['tempo_reale'])
                    x_range = np.linspace(valid_data['densita'].min(), valid_data['densita'].max(), 100)
                    plt.plot(x_range, slope * x_range + intercept, 'r-', linewidth=2, alpha=0.8,
                            label=f'Regressione (R² = {r_value**2:.3f})')
                except:
                    pass
            plt.xlabel('Densità della matrice', fontsize=20)
            plt.ylabel('Tempo di esecuzione (s)', fontsize=20)
            plt.title(f'Correlazione densità vs tempo di esecuzione {title_suffix}', fontsize=22, pad=16)
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter, label='Tempo (s)')
            if len(valid_data) >= 3:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'Dati insufficienti per analisi', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{prefix}density_vs_time{suffix}.png', dpi=300)
        plt.close()
        
        # Density vs CPU Time
        plt.figure(figsize=FIG_SIZE)
        valid_data = df_subset.dropna(subset=['densita', 'cpu_time_selected'])
        if len(valid_data) > 1:
            scatter = plt.scatter(valid_data['densita'], valid_data['cpu_time_selected'], 
                                 c=valid_data['cpu_time_selected'], cmap='plasma', 
                                 alpha=0.7, edgecolors='darkorange', linewidth=0.5, s=50)
            if len(valid_data) >= 3:
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data['densita'], valid_data['cpu_time_selected'])
                    x_range = np.linspace(valid_data['densita'].min(), valid_data['densita'].max(), 100)
                    plt.plot(x_range, slope * x_range + intercept, 'r-', linewidth=2, alpha=0.8,
                            label=f'Regressione (R² = {r_value**2:.3f})')
                except:
                    pass
            plt.xlabel('Densità della matrice', fontsize=20)
            plt.ylabel(f'CPU Time ({cpu_metric}) (s)', fontsize=20)
            plt.title(f'Correlazione densità vs CPU Time ({cpu_metric}) {title_suffix}', fontsize=22, pad=16)
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter, label=f'CPU Time ({cpu_metric}) (s)')
            if len(valid_data) >= 3:
                plt.legend()
        else:
            plt.text(0.5, 0.5, 'Dati insufficienti per analisi', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{prefix}density_vs_cpu_time{suffix}.png', dpi=300)
        plt.close()
        
        # Density vs Memory
        plt.figure(figsize=FIG_SIZE)
        valid_data = df_subset.dropna(subset=['densita', 'mem_rss_kb'])
        if len(valid_data) > 1:
            scatter = plt.scatter(valid_data['densita'], valid_data['mem_rss_kb']/1024, 
                                 c=valid_data['mem_rss_kb']/1024, cmap='plasma', 
                                 alpha=0.7, edgecolors='darkred', linewidth=0.5, s=50)
            plt.xlabel('Densità della matrice', fontsize=20)
            plt.ylabel('Memoria RSS (MB)', fontsize=20)
            plt.title(f'Correlazione densità vs memoria RSS {title_suffix}', fontsize=22, pad=16)
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter, label='Memoria (MB)')
        else:
            plt.text(0.5, 0.5, 'Dati insufficienti per analisi', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{prefix}density_vs_memory{suffix}.png', dpi=300)
        plt.close()
        
        # Density vs MHS
        plt.figure(figsize=FIG_SIZE)
        valid_data = df_subset.dropna(subset=['densita', 'MHS_trovati'])
        if len(valid_data) > 1:
            scatter = plt.scatter(valid_data['densita'], valid_data['MHS_trovati'], 
                                 c=valid_data['MHS_trovati'], cmap='coolwarm', 
                                 alpha=0.7, edgecolors='darkgreen', linewidth=0.5, s=50)
            plt.xlabel('Densità della matrice', fontsize=20)
            plt.ylabel('MHS trovati', fontsize=20)
            plt.title(f'Correlazione densità vs MHS trovati {title_suffix}', fontsize=22, pad=16)
            plt.grid(True, alpha=0.3)
            plt.colorbar(scatter, label='MHS trovati')
        else:
            plt.text(0.5, 0.5, 'Dati insufficienti per analisi', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        
        plt.tight_layout()
        plt.savefig(OUT_DIR / f'{prefix}density_vs_mhs{suffix}.png', dpi=300)
        plt.close()
        
        # Density by category - show all categories
        plt.figure(figsize=FIG_SIZE)
        valid_data = df_subset.dropna(subset=['densita', 'categoria'])
        if len(valid_data) > 0:
            # Prepare data for all categories in order
            box_data = []
            box_labels = []
            for cat in CATEGORY_ORDER:
                cat_data = valid_data[valid_data['categoria'] == cat]['densita'].values
                if len(cat_data) > 0:
                    box_data.append(cat_data)
                    box_labels.append(cat)
                else:
                    # Add empty array for categories with no data
                    box_data.append([])
                    box_labels.append(cat)
            
            if any(len(data) > 0 for data in box_data):
                bp = plt.boxplot([data for data in box_data if len(data) > 0], 
                               positions=[i for i, data in enumerate(box_data) if len(data) > 0],
                               tick_labels=[label for label, data in zip(box_labels, box_data) if len(data) > 0], 
                               patch_artist=True, 
                               boxprops=dict(facecolor=sns.color_palette("pastel")[0], alpha=0.7),
                               medianprops=dict(color='red', linewidth=2))
                plt.xlabel('Categoria', fontsize=20)
                plt.ylabel('Densità', fontsize=20)
                plt.title(f'Distribuzione densità per categoria {title_suffix}', fontsize=22, pad=16)
                plt.xticks(  ha='right', fontsize=20)
                plt.grid(axis='y', alpha=0.3)
                
                # Add sample size labels
                for i, (label, data) in enumerate(zip(box_labels, box_data)):
                    if len(data) > 0:
                        plt.text(i+1, plt.ylim()[1] - (plt.ylim()[1]-plt.ylim()[0])*0.05, 
                                f'n={len(data)}', ha='center', va='top', fontsize=14)
            else:
                plt.text(0.5, 0.5, 'Nessun dato valido per categorie', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        else:
            plt.text(0.5, 0.5, 'Dati insufficienti per analisi categorie', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        plt.savefig(OUT_DIR / f'{prefix}density_by_category{suffix}.png', dpi=300)
        plt.close()
        
        # Density distribution
        plt.figure(figsize=FIG_SIZE)
        valid_density = df_subset['densita'].dropna()
        if len(valid_density) > 0:
            plt.hist(valid_density, bins=20, alpha=0.7, color=sns.color_palette("pastel")[8], edgecolor='black', linewidth=1)
            
            mean_density = valid_density.mean()
            median_density = valid_density.median()
            
            plt.axvline(mean_density, color='red', linestyle='--', linewidth=2, 
                       label=f'Media: {mean_density:.4f}')
            plt.axvline(median_density, color='green', linestyle='--', linewidth=2,
                       label=f'Mediana: {median_density:.4f}')
            
            plt.xlabel('Densità della matrice', fontsize=20)
            plt.ylabel('Frequenza', fontsize=20)
            plt.title(f'Distribuzione della densità delle matrici {title_suffix}', fontsize=22, pad=16)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Dati insufficienti per istogramma', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=20)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
        plt.savefig(OUT_DIR / f'{prefix}density_distribution{suffix}.png', dpi=300)
        plt.close()

        print(f"Grafici densità generati:\n  ✓ {prefix}density_vs_time{suffix}.png\n  ✓ {prefix}density_vs_memory{suffix}.png\n  ✓ {prefix}density_vs_mhs{suffix}.png\n  ✓ {prefix}density_by_category{suffix}.png\n  ✓ {prefix}density_distribution{suffix}.png")
    else:
        print("Nessun dato di densità valido trovato, saltando grafici densità")

def generate_scatter_comparison_plots(df_serial, df_parallel, prefix, log_scale=False):
    """Generate scatter comparison plots with diagonal line (seriale vs automatico)"""
    FIG_SIZE = (12, 12)
    
    # Merge dataframes on 'file' to get paired data
    df_merged = pd.merge(df_serial, df_parallel, on='file', suffixes=('_serial', '_auto'))
    
    if len(df_merged) == 0:
        print("Nessun file comune trovato per scatter comparison")
        return
    
    # Define markers and colors for each category
    category_styles = {
        'trivial': {'marker': 'o', 'color': sns.color_palette("tab10")[0], 'label': 'Trivial'},
        'tiny': {'marker': 's', 'color': sns.color_palette("tab10")[1], 'label': 'Tiny'},
        'small': {'marker': '^', 'color': sns.color_palette("tab10")[2], 'label': 'Small'},
        'medium': {'marker': 'D', 'color': sns.color_palette("tab10")[3], 'label': 'Medium'},
        'large': {'marker': 'v', 'color': sns.color_palette("tab10")[4], 'label': 'Large'},
        'xlarge': {'marker': 'X', 'color': sns.color_palette("tab10")[5], 'label': 'Xlarge'}
    }
    
    suffix = '_log' if log_scale else ''
    scale_title = ' (scala logaritmica)' if log_scale else ''
    
    # 1. Tempo reale scatter
    plt.figure(figsize=FIG_SIZE)
    x = df_merged['tempo_reale_serial']
    y = df_merged['tempo_reale_auto']
    
    # Filter out non-positive values for log scale
    if log_scale:
        valid_mask = (x > 0) & (y > 0)
        x = x[valid_mask]
        y = y[valid_mask]
        df_filtered = df_merged[valid_mask]
    else:
        df_filtered = df_merged
    
    # Plot each category separately with its own marker and color
    for category, style in category_styles.items():
        mask = df_filtered['categoria_serial'] == category
        if mask.any():
            plt.scatter(x[mask], y[mask], 
                       marker=style['marker'], 
                       color=style['color'], 
                       s=200, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=1,
                       label=style['label'])
    
    # Diagonal line
    if log_scale:
        # For log scale, use geometric mean for better visualization
        x_vals = np.logspace(np.log10(max(x.min(), 1e-6)), np.log10(x.max()), 100)
        plt.plot(x_vals, x_vals, 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
        plt.xscale('log')
        plt.yscale('log')
    else:
        max_val = max(x.max(), y.max())
        min_val = min(x.min(), y.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
    
    plt.xlabel('Tempo reale seriale (s)', fontsize=18)
    plt.ylabel('Tempo reale automatico (s)', fontsize=18)
    plt.title(f'Confronto tempo reale: seriale vs automatico{scale_title}', fontsize=20, pad=16)
    plt.legend(loc='upper left', fontsize=12, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{prefix}scatter_time{suffix}.png', dpi=300)
    plt.close()
    
    # 2. Memoria RSS scatter
    plt.figure(figsize=FIG_SIZE)
    x = df_merged['mem_rss_kb_serial'] / 1024
    y = df_merged['mem_rss_kb_auto'] / 1024
    
    # Filter out non-positive values for log scale
    if log_scale:
        valid_mask = (x > 0) & (y > 0)
        x = x[valid_mask]
        y = y[valid_mask]
        df_filtered = df_merged[valid_mask]
    else:
        df_filtered = df_merged
    
    # Plot each category separately with its own marker and color
    for category, style in category_styles.items():
        mask = df_filtered['categoria_serial'] == category
        if mask.any():
            plt.scatter(x[mask], y[mask], 
                       marker=style['marker'], 
                       color=style['color'], 
                       s=200, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=1,
                       label=style['label'])
    
    if log_scale:
        # Set equal axis limits for better comparison
        combined_min = min(x.min(), y.min())
        combined_max = max(x.max(), y.max())
        plt.xlim(combined_min, combined_max)
        plt.ylim(combined_min, combined_max)
        # For log scale, create diagonal line that spans the full axis range
        x_vals = np.logspace(np.log10(max(combined_min, 1e-6)), np.log10(combined_max), 100)
        plt.plot(x_vals, x_vals, 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
        plt.xscale('log')
        plt.yscale('log')
    else:
        max_val = max(x.max(), y.max())
        min_val = min(x.min(), y.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
    
    plt.xlabel('Memoria RSS seriale (MB)', fontsize=18)
    plt.ylabel('Memoria RSS automatico (MB)', fontsize=18)
    plt.title(f'Confronto memoria RSS: seriale vs automatico{scale_title}', fontsize=20, pad=16)
    plt.legend(loc='upper left', fontsize=12, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{prefix}scatter_memory_rss{suffix}.png', dpi=300)
    plt.close()
    
    # 3. Picco memoria scatter
    plt.figure(figsize=FIG_SIZE)
    x = df_merged['mem_picco_kb_serial'] / 1024
    y = df_merged['mem_picco_kb_auto'] / 1024
    
    # Filter out non-positive values for log scale
    if log_scale:
        valid_mask = (x > 0) & (y > 0)
        x = x[valid_mask]
        y = y[valid_mask]
        df_filtered = df_merged[valid_mask]
    else:
        df_filtered = df_merged
    
    # Plot each category separately with its own marker and color
    for category, style in category_styles.items():
        mask = df_filtered['categoria_serial'] == category
        if mask.any():
            plt.scatter(x[mask], y[mask], 
                       marker=style['marker'], 
                       color=style['color'], 
                       s=200, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=1,
                       label=style['label'])
    
    if log_scale:
        # Set equal axis limits for better comparison
        combined_min = min(x.min(), y.min())
        combined_max = max(x.max(), y.max())
        plt.xlim(combined_min, combined_max)
        plt.ylim(combined_min, combined_max)
        # For log scale, create diagonal line that spans the full axis range
        x_vals = np.logspace(np.log10(max(combined_min, 1e-6)), np.log10(combined_max), 100)
        plt.plot(x_vals, x_vals, 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
        plt.xscale('log')
        plt.yscale('log')
    else:
        max_val = max(x.max(), y.max())
        min_val = min(x.min(), y.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
    
    plt.xlabel('Picco memoria seriale (MB)', fontsize=18)
    plt.ylabel('Picco memoria automatico (MB)', fontsize=18)
    plt.title(f'Confronto picco memoria: seriale vs automatico{scale_title}', fontsize=20, pad=16)
    plt.legend(loc='upper left', fontsize=12, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{prefix}scatter_memory_peak{suffix}.png', dpi=300)
    plt.close()
    
    # 4. MHS trovati scatter
    plt.figure(figsize=FIG_SIZE)
    x = df_merged['MHS_trovati_serial']
    y = df_merged['MHS_trovati_auto']
    
    # For MHS, add small offset for log scale to handle zeros
    if log_scale:
        offset = 1e-6  # Small offset to handle zeros
        x_plot = x + offset
        y_plot = y + offset
        valid_mask = (x_plot > 0) & (y_plot > 0)
        x_plot = x_plot[valid_mask]
        y_plot = y_plot[valid_mask]
        df_filtered = df_merged[valid_mask]
    else:
        x_plot = x
        y_plot = y
        df_filtered = df_merged
    
    # Plot each category separately with its own marker and color
    for category, style in category_styles.items():
        mask = df_filtered['categoria_serial'] == category
        if mask.any():
            plt.scatter(x_plot[mask], y_plot[mask], 
                       marker=style['marker'], 
                       color=style['color'], 
                       s=200, 
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=1,
                       label=style['label'])
    
    if log_scale:
        # For log scale, use geometric mean for better visualization
        x_vals = np.logspace(np.log10(max(x_plot.min(), 1e-6)), np.log10(x_plot.max()), 100)
        plt.plot(x_vals, x_vals, 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
        plt.xscale('log')
        plt.yscale('log')
    else:
        max_val = max(x.max(), y.max())
        min_val = 0
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (prestazioni uguali)')
    
    plt.xlabel('MHS trovati seriale', fontsize=18)
    plt.ylabel('MHS trovati automatico', fontsize=18)
    plt.title(f'Confronto MHS trovati: seriale vs automatico{scale_title}', fontsize=20, pad=16)
    plt.legend(loc='upper left', fontsize=12, ncol=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{prefix}scatter_mhs{suffix}.png', dpi=300)
    plt.close()

    scale_desc = "logaritmica" if log_scale else "lineare"
    print(f"Grafici scatter ({scale_desc}) generati:\n  ✓ {prefix}scatter_time{suffix}.png\n  ✓ {prefix}scatter_memory_rss{suffix}.png\n  ✓ {prefix}scatter_memory_peak{suffix}.png\n  ✓ {prefix}scatter_mhs{suffix}.png")

def generate_comparison_plots(serial_dir_name, parallel_dir_name, cpu_metric='total', log_scale=False):
    """Generate comparison plots between serial and parallel results"""
    serial_json = ROOT / 'NEW' / serial_dir_name / 'results.json'
    parallel_json = ROOT / 'NEW' / parallel_dir_name / 'results.json'
    
    if not serial_json.exists():
        print(f"Errore: file {serial_json} non trovato!")
        return
    if not parallel_json.exists():
        print(f"Errore: file {parallel_json} non trovato!")
        return
    
    df_serial = load_and_prepare_data(serial_json, cpu_metric)
    df_parallel = load_and_prepare_data(parallel_json, cpu_metric)
    print(f"Confronto caricato: {len(df_serial)} risultati seriali, {len(df_parallel)} risultati paralleli (metrica CPU: {cpu_metric})")
    
    prefix = f"confr_{cpu_metric}_"
    
    # Generate comparison plots
    generate_comparison_plots_code(df_serial, df_parallel, prefix, cpu_metric, log_scale)
    
    print(f"Grafici di confronto (metrica CPU: {cpu_metric}) generati in: {OUT_DIR}")

def generate_comparison_plots_code(df_serial, df_parallel, prefix, cpu_metric='total', log_scale=False):
    """Generate comparison plots between serial and automatic"""
    # Standard figure size for all plots
    FIG_SIZE = (16, 10)
    
    def get_comparison_stats(df1, df2, stat_col, metric='total'):
        # Scegli la funzione di aggregazione in base alla metrica
        if metric == 'mean':
            agg_func = 'mean'
        elif metric == 'sum':
            agg_func = 'sum'
        elif metric == 'max':
            agg_func = 'max'
        else:  # 'total' o altro
            agg_func = 'sum'
            
        stats1 = df1.groupby('categoria', observed=False)[stat_col].agg(agg_func).reset_index()
        stats1 = stats1.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
        stats2 = df2.groupby('categoria', observed=False)[stat_col].agg(agg_func).reset_index()
        stats2 = stats2.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
        # Fill NaN with 0.0
        stats1[stat_col] = stats1[stat_col].fillna(0.0)
        stats2[stat_col] = stats2[stat_col].fillna(0.0)
        return stats1[stat_col], stats2[stat_col]
    
    # Time comparison
    serial_times, parallel_times = get_comparison_stats(df_serial, df_parallel, 'tempo_reale', cpu_metric)
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(CATEGORY_ORDER))
    plt.bar(x - 0.2, serial_times, width=0.4, label='Seriale', color=sns.color_palette("pastel")[0], alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + 0.2, parallel_times, width=0.4, label='Automatico', color=sns.color_palette("pastel")[1], alpha=0.8, edgecolor='black', linewidth=1)
    plt.xticks(x, CATEGORY_ORDER,   ha='right', fontsize=20)
    plt.ylabel('Tempo (s)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Confronto tempo di esecuzione: seriale vs automatico', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}time_comparison.png', dpi=300)
    plt.close()
    
    # CPU Time comparison
    serial_cpu, parallel_cpu = get_comparison_stats(df_serial, df_parallel, 'cpu_time_selected', cpu_metric)
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(CATEGORY_ORDER))
    plt.bar(x - 0.2, serial_cpu, width=0.4, label='Seriale', color=sns.color_palette("pastel")[8], alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + 0.2, parallel_cpu, width=0.4, label='Automatico', color=sns.color_palette("pastel")[9], alpha=0.8, edgecolor='black', linewidth=1)
    plt.xticks(x, CATEGORY_ORDER,   ha='right', fontsize=20)
    plt.ylabel(f'CPU Time ({cpu_metric}) (s)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title(f'Confronto CPU Time: seriale vs automatico', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}cpu_time_comparison.png', dpi=300)
    plt.close()
    
    # Memory comparison
    serial_mem, parallel_mem = get_comparison_stats(df_serial, df_parallel, 'mem_rss_kb', 'mean')
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(CATEGORY_ORDER))
    plt.bar(x - 0.2, serial_mem/1024, width=0.4, label='Seriale', color=sns.color_palette("pastel")[2], alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + 0.2, parallel_mem/1024, width=0.4, label='Automatico', color=sns.color_palette("pastel")[3], alpha=0.8, edgecolor='black', linewidth=1)
    plt.xticks(x, CATEGORY_ORDER,   ha='right', fontsize=20)
    plt.ylabel('Memoria RSS (MB)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Confronto memoria: seriale vs automatico', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}memory_comparison.png', dpi=300)
    plt.close()
    
    # MHS comparison
    serial_files = set(df_serial['file'])
    parallel_files = set(df_parallel['file'])
    common_files = serial_files.intersection(parallel_files)

    def get_mhs_stats(df, files, metric):
        # Filter dataframe to include only common files
        df_filtered = df[df['file'].isin(files)]
        
        # Choose aggregation method based on metric
        if metric == 'max':
            agg_method = 'max'
            ylabel = 'Max MHS trovati'
            title = 'Confronto max MHS trovati: seriale vs automatico'
        if metric == 'mean':
            agg_method = 'mean'
            ylabel = 'Media MHS trovati'
            title = 'Confronto media MHS trovati: seriale vs automatico'
        elif metric == 'sum':
            agg_method = 'sum'
            ylabel = 'Somma MHS trovati'
            title = 'Confronto somma MHS trovati: seriale vs automatico'
        else:  # 'total'
            agg_method = 'sum'
            ylabel = 'Totale MHS trovati'
            title = 'Confronto totale MHS trovati: seriale vs automatico'
        
        mhs_stats = df_filtered.groupby('categoria', observed=False).agg({
            'MHS_trovati': agg_method,
            'file': 'count'  # Count files per category
        }).reset_index()
        
        mhs_stats = mhs_stats.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
        mhs_stats['MHS_trovati'] = mhs_stats['MHS_trovati'].fillna(0.0)
        mhs_stats['file'] = mhs_stats['file'].fillna(0)
        return mhs_stats, ylabel, title

    serial_stats, ylabel, title = get_mhs_stats(df_serial, common_files, cpu_metric)
    parallel_stats, _, _ = get_mhs_stats(df_parallel, common_files, cpu_metric)

    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(CATEGORY_ORDER))
    plt.bar(x - 0.2, serial_stats['MHS_trovati'], width=0.4, label='Seriale', 
            color=sns.color_palette("pastel")[4], alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + 0.2, parallel_stats['MHS_trovati'], width=0.4, label='Automatico', 
            color=sns.color_palette("pastel")[5], alpha=0.8, edgecolor='black', linewidth=1)
    plt.xticks(x, CATEGORY_ORDER,   ha='right', fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title(title, fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    
    plt.savefig(OUT_DIR / f'{prefix}mhs_comparison.png', dpi=300)
    plt.close()
    
    # Completion comparison
    def get_completion_stats(df):
        completion = df.groupby('categoria', observed=False).apply(lambda g: pd.Series({
            'count': len(g),
            'completed': int(g['completato'].sum())
        }), include_groups=False).reset_index()
        completion['pct'] = completion['completed'] / completion['count'] * 100
        completion = completion.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
        # Fill NaN with 0.0
        completion['pct'] = completion['pct'].fillna(0.0)
        return completion['pct']
    
    serial_completion = get_completion_stats(df_serial)
    parallel_completion = get_completion_stats(df_parallel)
    
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(CATEGORY_ORDER))
    plt.bar(x - 0.2, serial_completion, width=0.4, label='Seriale', color=sns.color_palette("pastel")[6], alpha=0.8, edgecolor='black', linewidth=1)
    plt.bar(x + 0.2, parallel_completion, width=0.4, label='Automatico', color=sns.color_palette("pastel")[7], alpha=0.8, edgecolor='black', linewidth=1)
    plt.xticks(x, CATEGORY_ORDER,   ha='right', fontsize=20)
    plt.ylabel('Percentuale completamento (%)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Confronto tasso di completamento: seriale vs automatico', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix}completion_comparison.png', dpi=300)
    plt.close()
    
    # Generate scatter comparison plots
    if log_scale:
        generate_scatter_comparison_plots(df_serial, df_parallel, prefix, log_scale=True)
    else:
        generate_scatter_comparison_plots(df_serial, df_parallel, prefix)
        generate_scatter_comparison_plots(df_serial, df_parallel, prefix, log_scale=True)

def generate_used_plots_only(log_scale=False):
    """Generate only the plots actually used in Chapter 4"""
    print("=" * 80)
    print("GENERATING ONLY USED PLOTS FROM CHAPTER 4")
    print("=" * 80)
    
    # Load data
    serial_json = ROOT / 'NEW' / 'risultati_serial' / 'results.json'
    auto_json = ROOT / 'NEW' / 'risultati_auto' / 'results.json'
    
    if not serial_json.exists() or not auto_json.exists():
        print("Errore: file JSON non trovati!")
        return
    
    df_serial = load_and_prepare_data(serial_json, 'total')
    df_auto = load_and_prepare_data(auto_json, 'total')
    
    print(f"\nCaricati {len(df_serial)} risultati seriali e {len(df_auto)} risultati automatici")
    print("\n" + "=" * 80)
    
    # 1. Modalità automatica - 4 grafici
    print("\n[1/4] Generando grafici modalità automatica...")
    prefix_auto = "auto_total_"
    FIG_SIZE = (16, 10)
    
    # 1.1 Completion by category
    completion = df_auto.groupby('categoria', observed=False).apply(lambda g: pd.Series({
        'count': len(g),
        'completed': int(g['completato'].sum())
    }), include_groups=False).reset_index()
    completion['pct'] = completion['completed'] / completion['count'] * 100
    completion = completion.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
    completion['pct'] = completion['pct'].fillna(0.0)

    plt.figure(figsize=FIG_SIZE)
    bars = plt.bar(completion['categoria'], completion['pct'], color=sns.color_palette('pastel')[9], 
                   edgecolor='black', linewidth=1)
    plt.ylabel('Percentuale completamento (%)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Percentuale di completamento per categoria', fontsize=22, pad=16)
    plt.xticks(ha='right', fontsize=20)
    plt.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars, completion['pct']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix_auto}completion_by_category.png', dpi=300)
    plt.close()
    print(f"  ✓ {prefix_auto}completion_by_category.png")
    
    # 1.2 Time by category
    time_stats = df_auto.groupby('categoria', observed=False)['tempo_reale'].agg(['mean','median']).reset_index()
    time_stats = time_stats.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
    time_stats['mean'] = time_stats['mean'].fillna(0.0)
    time_stats['median'] = time_stats['median'].fillna(0.0)
    
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(time_stats))
    bars1 = plt.bar(x - 0.2, time_stats['mean'], width=0.4, label='Mean', 
                    color=sns.color_palette('pastel')[1], edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + 0.2, time_stats['median'], width=0.4, label='Median', 
                    color=sns.color_palette('pastel')[2], edgecolor='black', linewidth=1)
    plt.xticks(x, time_stats['categoria'], ha='right', fontsize=20)
    plt.ylabel('Tempo reale (s)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Tempo reale medio e mediano per categoria', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars1, time_stats['mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    for bar, value in zip(bars2, time_stats['median']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix_auto}time_by_category.png', dpi=300)
    plt.close()
    print(f"  ✓ {prefix_auto}time_by_category.png")
    
    # 1.3 MHS vs columns
    plt.figure(figsize=FIG_SIZE)
    scatter = plt.scatter(df_auto['M_ridotto'], df_auto['MHS_trovati'], 
                         c=df_auto['MHS_trovati'], cmap='Blues', 
                         alpha=0.8, edgecolors='darkblue', linewidth=1.0, s=60)
    plt.xlabel("Numero di colonne ridotte (M')", fontsize=20)
    plt.ylabel('Numero di MHS trovati', fontsize=20)
    plt.title("MHS trovati vs colonne ridotte", fontsize=22, pad=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f'{prefix_auto}mhs_vs_cols.png', dpi=300)
    plt.close()
    print(f"  ✓ {prefix_auto}mhs_vs_cols.png")
    
    # 1.4 Memory stats
    mem_stats = df_auto.groupby('categoria', observed=False).agg({
        'mem_rss_kb':'mean',
        'mem_picco_kb':'mean'
    }).reset_index()
    mem_stats = mem_stats.set_index('categoria').reindex(CATEGORY_ORDER).reset_index()
    mem_stats['mem_rss_kb'] = mem_stats['mem_rss_kb'].fillna(0.0)
    mem_stats['mem_picco_kb'] = mem_stats['mem_picco_kb'].fillna(0.0)
    
    plt.figure(figsize=FIG_SIZE)
    x = np.arange(len(mem_stats))
    bars1 = plt.bar(x - 0.15, mem_stats['mem_rss_kb']/1024, width=0.3, label='RSS mean (MB)', 
                    color=sns.color_palette('pastel')[3], edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + 0.15, mem_stats['mem_picco_kb']/1024, width=0.3, label='Peak mean (MB)', 
                    color=sns.color_palette('pastel')[4], edgecolor='black', linewidth=1)
    plt.xticks(x, mem_stats['categoria'], ha='right', fontsize=20)
    plt.ylabel('Memoria (MB)', fontsize=20)
    plt.xlabel('Categoria', fontsize=20)
    plt.title('Statistiche di memoria (media) per categoria', fontsize=22, pad=16)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    for bar, value in zip(bars1, mem_stats['mem_rss_kb']/1024):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    for bar, value in zip(bars2, mem_stats['mem_picco_kb']/1024):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.1f}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.9)
    plt.savefig(OUT_DIR / f'{prefix_auto}memory_stats.png', dpi=300)
    plt.close()
    print(f"  ✓ {prefix_auto}memory_stats.png")
    
    # 2. Density plots - Approccio 1 (solo completate)
    print("\n[2/4] Generando grafici densità (Approccio 1 - solo completate)...")
    df_completed = df_auto[df_auto['completato'] == True].copy()
    generate_density_plots_subset(df_completed, prefix_auto, 'total', suffix='', 
                                   title_suffix='(solo matrici completate)')
    
    # 3. Density plots - Approccio 2 (tutte)
    print("\n[3/4] Generando grafici densità (Approccio 2 - tutte)...")
    generate_density_plots_subset(df_auto.copy(), prefix_auto, 'total', suffix='_all', 
                                   title_suffix='(tutte le matrici)')
    
    # 4. Comparison plots
    print("\n[4/4] Generando grafici di confronto...")
    prefix_comp = "confr_total_"
    
    # Scatter comparison plots (only these are used in Chapter 4)
    print("  Generando scatter plots di confronto...")
    if log_scale:
        generate_scatter_comparison_plots(df_serial, df_auto, prefix_comp, log_scale=True)
    else:
        generate_scatter_comparison_plots(df_serial, df_auto, prefix_comp)
        generate_scatter_comparison_plots(df_serial, df_auto, prefix_comp, log_scale=True)
    
    print("\n" + "=" * 80)
    print("COMPLETATO! Tutti i grafici usati nel Capitolo 4 sono stati generati.")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument('mode', choices=['risultati_serial', 'risultati_auto', 'comparison', 'all', 'used'], 
                       help='Mode: risultati_serial, risultati_auto, comparison, all, or used')
    parser.add_argument('--cpu_metric', choices=['total', 'sum', 'mean', 'max'], default='total',
                       help='CPU time metric to use: total (tempo_cpu), sum/mean/max of worker times')
    parser.add_argument('--log_scale', action='store_true',
                       help='Generate scatter plots in logarithmic scale')
    
    args = parser.parse_args()
    
    if args.mode == 'used':
        # Generate only plots used in Chapter 4
        generate_used_plots_only(log_scale=args.log_scale)
    elif args.mode == 'all':
        print("Generating all plots...")
        for metric in ['total', 'sum', 'mean', 'max']:
            print(f"\n--- Generating plots for CPU metric: {metric} ---")
            generate_plots_for_mode('risultati_serial', metric)
            generate_plots_for_mode('risultati_auto', metric) 
            generate_comparison_plots('risultati_serial', 'risultati_auto', metric, log_scale=args.log_scale)
            # Also generate scatter plots for comparison
            serial_json = ROOT / 'NEW' / 'risultati_serial' / 'results.json'
            auto_json = ROOT / 'NEW' / 'risultati_auto' / 'results.json'
            if serial_json.exists() and auto_json.exists():
                df_serial = load_and_prepare_data(serial_json, metric)
                df_auto = load_and_prepare_data(auto_json, metric)
                if args.log_scale:
                    generate_scatter_comparison_plots(df_serial, df_auto, f"confr_{metric}_", log_scale=True)
                else:
                    generate_scatter_comparison_plots(df_serial, df_auto, f"confr_{metric}_")
                    generate_scatter_comparison_plots(df_serial, df_auto, f"confr_{metric}_", log_scale=True)
    elif args.mode == 'comparison':
        generate_comparison_plots('risultati_serial', 'risultati_auto', args.cpu_metric, log_scale=args.log_scale)
    else:
        generate_plots_for_mode(args.mode, args.cpu_metric)
