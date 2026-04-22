import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# Load corporate colors
C_DARK = "#1B3A6B"
C_MID  = "#2E86C1"
C_GREEN = "#27AE60"
C_RED   = "#E74C3C"

# Load the data
csv_path = "/home/giova/Hermes/Fintech_proj2/Buisness_cases_Fintech/BuisnessCase2/Output/02_baselines/02_baselines_results.csv"
df = pd.read_csv(csv_path)
df = df.dropna(subset=['Target'])

def parse_cv_f1(s):
    # Extracts "0.631" from "0.631 (±0.010)"
    match = re.search(r"([\d\.]+)", str(s))
    if match:
        return float(match.group(1))
    return 0.0

def format_delta_abs(val, base_val):
    delta = val - base_val
    return f"{val:.3f} ({delta:+.3f})"

def format_cv_f1(eng_s, base_s):
    # eng_s: "0.682 (±0.009)"
    # base_s: "0.631 (±0.010)"
    v_eng = parse_cv_f1(eng_s)
    v_base = parse_cv_f1(base_s)
    delta = v_eng - v_base
    return f"{eng_s} ({delta:+.3f})"

def generate_report(target_name, output_png):
    target_df = df[df['Target'] == target_name]
    base_df = target_df[target_df['Features'] == 'Base'].set_index('Model')
    eng_df = target_df[target_df['Features'] == 'Engineered'].set_index('Model')
    
    # Sort models based on Test ROC-AUC (Engineered) descending
    all_models = eng_df.index.unique().tolist()
    sorted_models = eng_df.loc[all_models].sort_values("Test ROC-AUC", ascending=False).index.tolist()
    
    table_data = []
    cell_colors = []
    
    # Updated columns: removed CV F1
    columns = ["Model", "CV ROC-AUC (vs Raw)", "Test F1 (vs Raw)", "Test ROC-AUC (vs Raw)"]
    
    for model in sorted_models:
        if model in base_df.index:
            row = [model]
            row_colors = ["#F8F9F9"] 
            
            # Metrics to show
            metrics = [
                (eng_df.loc[model, 'CV ROC-AUC'], base_df.loc[model, 'CV ROC-AUC'], f"{eng_df.loc[model, 'CV ROC-AUC']:.3f}"),
                (eng_df.loc[model, 'Test F1'], base_df.loc[model, 'Test F1'], f"{eng_df.loc[model, 'Test F1']:.3f}"),
                (eng_df.loc[model, 'Test ROC-AUC'], base_df.loc[model, 'Test ROC-AUC'], f"{eng_df.loc[model, 'Test ROC-AUC']:.3f}")
            ]
            
            for eng_v, base_v, label in metrics:
                delta = eng_v - base_v
                # NEW FORMAT: Horizontal delta
                row.append(f"{label} ({delta:+.3f})")
                row_colors.append("#EAFAF1" if delta >= 0 else "#FDEDEC")
            
            table_data.append(row)
            cell_colors.append(row_colors)

    # Matplotlib Table Generation
    fig_height = max(5, len(sorted_models) * 0.7)
    fig, ax = plt.subplots(figsize=(15, fig_height))
    ax.axis('off')
    
    the_table = ax.table(cellText=table_data,
                         cellColours=cell_colors,
                         colLabels=columns,
                         loc='center',
                         cellLoc='center')
    
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(11)
    the_table.scale(1.2, 2.5)
    
    # Header styling
    for i, col in enumerate(columns):
        cell = the_table[0, i]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor(C_DARK)
    
    # Bold the top 3 models (Podium)
    for row_idx in range(1, min(len(table_data) + 1, 4)): # 1 to 3
        for col_idx in range(len(columns)):
            cell = the_table[row_idx, col_idx]
            cell.set_text_props(weight='bold')
    
    plt.title(f"Model Performance Assessment: Engineered vs Raw Data\n{target_name} | Sorted by Test ROC-AUC", 
              fontsize=16, fontweight='bold', color=C_DARK, pad=35)
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Clean Audit Table salvata: {output_png}")

if __name__ == "__main__":
    out_dir = "/home/giova/Hermes/Fintech_proj2/Buisness_cases_Fintech/BuisnessCase2/Output/Pipeline_Y"
    generate_report("AccumulationInvestment", f"{out_dir}/07y_baseline_acc_table.png")
    generate_report("IncomeInvestment", f"{out_dir}/07y_baseline_inc_table.png")
