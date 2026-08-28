import os
import sys
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr
def create_correlation_report(df, metrics, classnames):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Chest X-Ray Quality Analysis Pipeline: Automated Statistical Diagnostics", fontsize=18, weight='bold', y=0.98)
    
    # 1. Dynamic Metric vs Metric Matrix
    ax1 = axes[0, 0]
    corr_matrix = df[metrics].corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f", ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title("Global Metric Cross-Correlations", weight='bold')
    ax1.set_xticklabels(metrics, rotation=30, ha='right')
    
    # 2. Dynamic Point-Biserial Significance Chart (Top 5 Strongest Drivers)
    ax2 = axes[0, 1]
    pb_records = []
    for cls in classnames:
        for met in metrics:
            corr, p_val = pointbiserialr(df[cls], df[met])
            if p_val < 0.05:
                pb_records.append({'label': f"{cls} -> {met}", 'r': corr, 'abs_r': abs(corr)})
    
    if pb_records:
        pb_df = pd.DataFrame(pb_records).sort_values(by='abs_r', ascending=False).head(5)
        colors = ['#2ca02c' if x > 0 else '#d62728' for x in pb_df['r']]
        ax2.barh(pb_df['label'], pb_df['r'], color=colors, edgecolor='black', alpha=0.8)
        ax2.axvline(0, color='black', linewidth=1, linestyle='--')
        ax2.set_xlim(-1, 1)
    ax2.set_title("Top 5 Driving Factors (Point-Biserial $r$)", weight='bold')
    ax2.set_xlabel("Correlation Strength ($r$)")

    # 3. Stratified Population Split Dynamics
    ax3 = axes[1, 0]
    df_good = df[df['quality_category'] == 'good']
    df_bad = df[df['quality_category'] == 'bad']
    
    # Dynamically track specific high-variance pairs present in the metrics list
    labels_strat = []
    good_strat = []
    bad_strat = []
    
    # Safe check to see which specific target metrics exist to build cross-signatures
    base_pairs = [('niqe', 'skewness'), ('entropy', 'skewness')]
    if 'contrast_ratio' in metrics and 'entropy' in metrics:
        base_pairs.append(('contrast_ratio', 'entropy'))
    if 'brisque' in metrics and 'entropy' in metrics:
        base_pairs.append(('brisque', 'entropy'))
        
    for m1, m2 in base_pairs:
        if m1 in df.columns and m2 in df.columns:
            labels_strat.append(f"{m1[:5]}-{m2[:5]}")
            good_strat.append(df_good[m1].corr(df_good[m2]) if not df_good.empty else 0)
            bad_strat.append(df_bad[m1].corr(df_bad[m2]) if not df_bad.empty else 0)
            
    x = np.arange(len(labels_strat))
    if len(x) > 0:
        ax3.bar(x - 0.2, good_strat, 0.4, label='Good Pop', color='#2ca02c', alpha=0.8, edgecolor='black')
        ax3.bar(x + 0.2, bad_strat, 0.4, label='Bad Pop', color='#d62728', alpha=0.8, edgecolor='black')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels_strat, rotation=15)
    ax3.set_title("Population Dynamics: Artifact Signatures", weight='bold')
    ax3.set_ylabel("Correlation Coefficient")
    ax3.legend()

    # 4. Dynamic Cohen's d Effect Size Power Plot
    ax4 = axes[1, 1]
    cohen_d_vals = []
    for met in metrics:
        if not df_good.empty and not df_bad.empty:
            n_g, n_b = len(df_good), len(df_bad)
            pooled_std = np.sqrt(((n_g - 1) * df_good[met].var() + (n_b - 1) * df_bad[met].var()) / (n_g + n_b - 2))
            d_val = (df_good[met].mean() - df_bad[met].mean()) / pooled_std if pooled_std != 0 else 0
            cohen_d_vals.append(d_val)
        else:
            cohen_d_vals.append(0)
            
    ax4.bar(metrics, cohen_d_vals, color='#1f77b4', edgecolor='black', alpha=0.8, width=0.5)
    ax4.axhline(0, color='black', linewidth=1)
    ax4.set_title("Class Separation Power (Cohen's $d$)", weight='bold')
    ax4.set_ylabel("Effect Size Magnitude ($d$)")
    ax4.set_xticklabels(metrics, rotation=30, ha='right')
    
    plt.tight_layout()
    plt.savefig('./pipeline_correlation_report.png', dpi=300)
    plt.close()

def main():
    CLASSNAMES = [
        "collimation", 
        "good_contrast", 
        "medical_device", 
        "truncated", 
        "underexposed_noisy",
        "rotated", 
        "lateral"
    ]
    METRICS = [
        "niqe", 
        "contrast_ratio",
        "entropy",
        "sharpness",
        "skewness",
        "brisque"
    ]

    df = pd.read_csv("/home/nicolasg/dev/correlation/merged_image_data.csv")
    
    for cls in CLASSNAMES:
        df[cls] = df[cls].astype(bool)

    no_reference_metrics = df[METRICS]
    label = df[CLASSNAMES]
    color_palette = {'good': '#2ca02c', 'bad': '#d62728', 'other': '#7f7f7f'}

    df['quality_category'] = 'other'
    good_mask = (df[CLASSNAMES[1]] == True)  
    df.loc[good_mask, 'quality_category'] = 'good'
    bad_mask = df[CLASSNAMES[4]] == True
    df.loc[bad_mask, 'quality_category'] = 'bad'

    # 1. Metric vs Metric & Label vs Label Correlations
    mask = np.triu(np.ones((len(METRICS), len(METRICS)), dtype=bool), k=1)
    print("\n--- Metric vs Metric Correlations ---")
    print(no_reference_metrics.corr().where(mask).stack())

    mask_lbl = np.triu(np.ones((len(CLASSNAMES), len(CLASSNAMES)), dtype=bool), k=1)
    print("\n--- Label vs Label Correlations ---")
    print(label.corr().where(mask_lbl).stack())

    # 2. Point-Biserial Correlations
    print("\n--- Point-Biserial Correlations (Label vs Metric) ---")
    for cls in CLASSNAMES:
        for met in METRICS:
            corr, p_val = pointbiserialr(df[cls], df[met])
            if p_val < 0.05:
                print(f"{cls} <-> {met}: r = {corr:.4f} (p = {p_val:.4e})")

    # 3. Grouped Population Correlations
    df_good = df[df['quality_category'] == 'good']
    df_bad = df[df['quality_category'] == 'bad']

    if not df_good.empty:
        print("\n--- Within GOOD Population Metric Correlations ---")
        print(df_good[METRICS].corr().where(mask).stack())

    if not df_bad.empty:
        print("\n--- Within BAD Population Metric Correlations ---")
        print(df_bad[METRICS].corr().where(mask).stack())

    # 4. Class Separation Power
    print("\n--- Class Separation Power (Good vs Bad) ---")
    if not df_good.empty and not df_bad.empty:
        for met in METRICS:
            mean_g, mean_b = df_good[met].mean(), df_bad[met].mean()
            var_g, var_b = df_good[met].var(), df_bad[met].var()
            n_g, n_b = len(df_good), len(df_bad)
            
            pooled_std = np.sqrt(((n_g - 1) * var_g + (n_b - 1) * var_b) / (n_g + n_b - 2))
            cohens_d = (mean_g - mean_b) / pooled_std if pooled_std != 0 else 0
            print(f"{met} Separation (Cohen's d): {cohens_d:.4f} (Good Mean: {mean_g:.3f}, Bad Mean: {mean_b:.3f})")

    # 5. Pairplot
    sns.pairplot(
        df[METRICS + ['quality_category']], 
        hue='quality_category', 
        palette=color_palette,
        diag_kind='kde',
        plot_kws={'alpha': 0.6}
    )
    plt.savefig('./pairplot_all_images_colored.png')
    plt.close()

    # 7. Generate Multi-Panel PNG Diagnostics Report
    print("\nGenerating visual consolidated diagnostic report canvas...")
    create_correlation_report(df,METRICS,CLASSNAMES)
    print("Report compiled successfully! File saved at: ./pipeline_correlation_report.png")

    return 

if __name__ == "__main__":
    main()