import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np

REPORTS_DIR = "reports"

# Input CSV files
RESULTS_ORIGINAL_CSV = "analysis_results_original.csv"
RESULTS_BALANCED_CSV = "analysis_results_balanced_norm.csv"

# Specify which models to include in the report.
TARGET_MODELS = [
    "gemini_flash",
    "gemini_pro",
    "gpt_image",
]

# Fitzpatrick Skin Type Labels
FST_LABELS = ["I", "II", "III", "IV", "V", "VI"]
FST_COLORS = ["#ffc0c0", "#f4a985", "#d9886c", "#c66747", "#8d4330", "#4b2f23"]

def load_data(csv_path):
    """Load CSV and filter by target models."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return None
    try:
        df = pd.read_csv(csv_path)
        if TARGET_MODELS:
            df = df[df['Model'].isin(TARGET_MODELS)]
        return df
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def plot_fst_distribution(df, method_prefix, title_suffix, output_path):
    """Plot FST Type distribution by model."""
    col = f"{method_prefix}_FST_Type"
    if col not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Count FST types per model
    grouped = df.groupby(['Model', col]).size().unstack(fill_value=0)
    
    # Ensure all FST types 1-6 are present
    for fst in range(1, 7):
        if fst not in grouped.columns:
            grouped[fst] = 0
    grouped = grouped[[1, 2, 3, 4, 5, 6]]
    
    # Plot stacked bar
    ax = grouped.plot(kind='bar', stacked=True, color=FST_COLORS, edgecolor='black', figsize=(12, 6))
    
    plt.title(f"FST Distribution by Model - {method_prefix} ({title_suffix})")
    plt.xlabel("Model")
    plt.ylabel("Count")
    plt.legend(title="FST Type", labels=[f"FST {l}" for l in FST_LABELS])
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_mst_distribution(df, method_prefix, title_suffix, output_path):
    """Plot MST Index distribution by model."""
    col = f"{method_prefix}_MST_Index"
    if col not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create boxplot
    sns.boxplot(data=df, x='Model', y=col, palette="YlOrBr")
    plt.ylim(0.5, 10.5)
    plt.ylabel("MST Index (1=Light, 10=Dark)")
    plt.title(f"MST Distribution by Model - {method_prefix} ({title_suffix})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_mst_by_gender(df, method_prefix, title_suffix, output_path):
    """Plot MST Index by Gender and Model."""
    col = f"{method_prefix}_MST_Index"
    if col not in df.columns or 'Gender' not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Model', y=col, hue='Gender', palette="Set2")
    plt.ylim(0.5, 10.5)
    plt.ylabel("MST Index (1=Light, 10=Dark)")
    plt.title(f"MST by Gender - {method_prefix} ({title_suffix})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_fst_by_gender(df, method_prefix, title_suffix, output_path):
    """Plot FST Type by Gender and Model."""
    col = f"{method_prefix}_FST_Type"
    if col not in df.columns or 'Gender' not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Model', y=col, hue='Gender', palette="Set2")
    plt.ylim(0.5, 6.5)
    plt.ylabel("FST Type (I-VI)")
    plt.title(f"FST by Gender - {method_prefix} ({title_suffix})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_gender_distribution(df, title_suffix, output_path):
    """Plot Gender distribution by model."""
    if 'Gender' not in df.columns:
        return
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Model', hue='Gender', palette="pastel")
    plt.title(f"Gender Distribution ({title_suffix})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_race_distribution(df, title_suffix, output_path):
    """Plot Race distribution by model."""
    if 'Race' not in df.columns:
        return
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Model', hue='Race', palette="viridis")
    plt.title(f"Race Distribution ({title_suffix})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_text_report(df, title_suffix, output_path):
    """Generate detailed text report with per-prompt and PERLA statistics."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"SKIN TONE ANALYSIS REPORT - {title_suffix}\n")
        f.write("=" * 70 + "\n\n")
        
        # Overall Statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        
        for prefix, method_name in [('P', 'Precise Mask')]:
            mst_col = f"{prefix}_MST_Index"
            perla_col = f"{prefix}_PERLA_Index"
            fst_col = f"{prefix}_FST_Type"
            
            f.write(f"\n{method_name} Method:\n")
            
            if mst_col in df.columns:
                f.write(f"  MST Index: Mean={df[mst_col].mean():.2f}, Std={df[mst_col].std():.2f}\n")
            
            if perla_col in df.columns:
                f.write(f"  PERLA Index: Mean={df[perla_col].mean():.2f}, Std={df[perla_col].std():.2f}\n")
            
            if fst_col in df.columns:
                f.write(f"  FST Type: Mean={df[fst_col].mean():.2f}, Mode={df[fst_col].mode().iloc[0]}\n")
                # FST Distribution
                fst_counts = df[fst_col].value_counts().sort_index()
                f.write("  FST Distribution:\n")
                for fst_type, count in fst_counts.items():
                    pct = count / len(df) * 100
                    f.write(f"    Type {FST_LABELS[int(fst_type)-1]}: {count} ({pct:.1f}%)\n")
        
        f.write("\n\n")
        
        # Per-Model Statistics
        f.write("PER-MODEL STATISTICS\n")
        f.write("=" * 70 + "\n")
        
        for model in sorted(df['Model'].unique()):
            model_df = df[df['Model'] == model]
            f.write(f"\n{'='*50}\n")
            f.write(f"MODEL: {model.upper()}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Total Samples: {len(model_df)}\n")
            
            # Gender
            if 'Gender' in model_df.columns:
                gender_counts = model_df['Gender'].value_counts()
                total = len(model_df)
                f.write("Gender Distribution:\n")
                for g, c in gender_counts.items():
                    f.write(f"  {g}: {c} ({c/total*100:.1f}%)\n")
            
            # Race
            if 'Race' in model_df.columns:
                race_counts = model_df['Race'].value_counts()
                f.write("Race Distribution:\n")
                for r, c in race_counts.items():
                    f.write(f"  {r}: {c} ({c/len(model_df)*100:.1f}%)\n")
            
            # Skin Tone by Method
            # Skin Tone by Method
            for prefix, method_name in [('P', 'Precise Mask')]:
                mst_col = f"{prefix}_MST_Index"
                perla_col = f"{prefix}_PERLA_Index"
                fst_col = f"{prefix}_FST_Type"
                
                f.write(f"\n  {method_name} Skin Tone:\n")
                if mst_col in model_df.columns:
                    f.write(f"    MST Index: Mean={model_df[mst_col].mean():.2f} (Std={model_df[mst_col].std():.2f})\n")
                if perla_col in model_df.columns:
                    f.write(f"    PERLA Index: Mean={model_df[perla_col].mean():.2f} (Std={model_df[perla_col].std():.2f})\n")
                if fst_col in model_df.columns:
                    f.write(f"    FST Type: Mean={model_df[fst_col].mean():.2f} (Mode={model_df[fst_col].mode().iloc[0]})\n")
            
            # Per-Prompt Breakdown
            if 'Prompt' in model_df.columns:
                f.write(f"\n  Per-Prompt Breakdown:\n")
                f.write(f"  {'-'*45}\n")
                
                for prompt in sorted(model_df['Prompt'].unique()):
                    prompt_df = model_df[model_df['Prompt'] == prompt]
                    f.write(f"\n  Prompt: {prompt}\n")
                    f.write(f"    Samples: {len(prompt_df)}\n")
                    
                    # Gender for prompt
                    if 'Gender' in prompt_df.columns:
                        gc = prompt_df['Gender'].value_counts()
                        f.write(f"    Gender: {dict(gc)}\n")
                    
                    # Skin tones
                    for prefix, mn in [('P', 'Precise')]:
                        mst_col = f"{prefix}_MST_Index"
                        perla_col = f"{prefix}_PERLA_Index"
                        fst_col = f"{prefix}_FST_Type"
                        
                        if mst_col in prompt_df.columns:
                            f.write(f"    {mn}: MST={prompt_df[mst_col].mean():.2f}, ")
                        if perla_col in prompt_df.columns:
                            f.write(f"PERLA={prompt_df[perla_col].mean():.2f}, ")
                        if fst_col in prompt_df.columns:
                            f.write(f"FST={prompt_df[fst_col].mean():.2f}\n")
        
        # By Gender Analysis
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("BY GENDER ANALYSIS\n")
        f.write("=" * 70 + "\n")
        
        if 'Gender' in df.columns:
            for gender in sorted(df['Gender'].unique()):
                gender_df = df[df['Gender'] == gender]
                f.write(f"\n{gender.upper()}:\n")
                f.write(f"  Total Count: {len(gender_df)}\n")
                
                for prefix, method_name in [('P', 'Precise Mask')]:
                    mst_col = f"{prefix}_MST_Index"
                    perla_col = f"{prefix}_PERLA_Index"
                    fst_col = f"{prefix}_FST_Type"
                    
                    f.write(f"  {method_name}:\n")
                    if mst_col in gender_df.columns:
                        f.write(f"    MST Mean: {gender_df[mst_col].mean():.2f}\n")
                    if perla_col in gender_df.columns:
                        f.write(f"    PERLA Mean: {gender_df[perla_col].mean():.2f}\n")
                    if fst_col in gender_df.columns:
                        f.write(f"    FST Mean: {gender_df[fst_col].mean():.2f}\n")
        
        # Race-Gender Cross-Tabulation
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("RACE-GENDER CROSS-TABULATION\n")
        f.write("=" * 70 + "\n")
        
        if 'Race' in df.columns and 'Gender' in df.columns:
            # Overall cross-tab
            f.write("\nOverall (All Models):\n")
            f.write("-" * 50 + "\n")
            
            crosstab = pd.crosstab(df['Race'], df['Gender'], margins=True)
            crosstab_pct = pd.crosstab(df['Race'], df['Gender'], normalize='index') * 100
            
            genders = sorted(df['Gender'].unique())
            header = "Race".ljust(20) + "".join([g.ljust(15) for g in genders]) + "Total".ljust(10) + "% Women"
            f.write(f"  {header}\n")
            f.write("  " + "-" * len(header) + "\n")
            
            for race in crosstab.index:
                if race == 'All': continue
                row = str(race).ljust(20)
                for g in genders:
                    count = crosstab.loc[race, g] if g in crosstab.columns else 0
                    row += str(count).ljust(15)
                total = crosstab.loc[race, 'All'] if 'All' in crosstab.columns else 0
                women_pct = crosstab_pct.loc[race, 'Woman'] if 'Woman' in crosstab_pct.columns and race in crosstab_pct.index else 0
                row += str(total).ljust(10) + f"{women_pct:.1f}%"
                f.write(f"  {row}\n")
            
            # Per-model race-gender
            f.write("\nPer-Model Race-Gender:\n")
            f.write("-" * 50 + "\n")
            
            for model in sorted(df['Model'].unique()):
                model_df = df[df['Model'] == model]
                f.write(f"\n  {model.upper()}:\n")
                
                for race in sorted(model_df['Race'].unique()):
                    race_df = model_df[model_df['Race'] == race]
                    total = len(race_df)
                    if total < 5:  # Skip very small groups
                        continue
                    women_count = len(race_df[race_df['Gender'] == 'Woman'])
                    men_count = len(race_df[race_df['Gender'] == 'Man'])
                    women_pct = women_count / total * 100
                    f.write(f"    {race}: {total} total ({women_count} Women/{men_count} Men = {women_pct:.1f}% Women)\n")
        
        # Age Analysis
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("AGE ANALYSIS (Mean Age per Prompt)\n")
        f.write("=" * 70 + "\n")

        for model in sorted(df['Model'].unique()):
            model_df = df[df['Model'] == model]
            f.write(f"\n  {model.upper()}:\n")
            
            # Overall mean age for model
            mean_age_model = model_df['Age'].mean()
            f.write(f"    Overall Mean Age: {mean_age_model:.1f} years\n")
            f.write(f"    Breakdown by Prompt:\n")

            for prompt in sorted(model_df['Prompt'].unique()):
                prompt_df = model_df[model_df['Prompt'] == prompt]
                mean_age = prompt_df['Age'].mean()
                f.write(f"      - {prompt}: {mean_age:.1f} years (n={len(prompt_df)})\n")
        
        # PERLA Distribution Summary
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("PERLA INDEX DISTRIBUTION (Overall)\n")
        f.write("=" * 70 + "\n")
        
        for prefix, method_name in [('P', 'Precise Mask')]:
            perla_col = f"{prefix}_PERLA_Index"
            if perla_col in df.columns:
                f.write(f"\n{method_name}:\n")
                perla_counts = df[perla_col].value_counts().sort_index()
                for perla_idx, count in perla_counts.items():
                    pct = count / len(df) * 100
                    f.write(f"  PERLA {int(perla_idx)}: {count} ({pct:.1f}%)\n")
        
        # Performance Statistical Tests
        perform_statistical_tests(df, f)
    
    print(f"Text report saved: {output_path}")

def generate_report_for_file(csv_path, title_suffix):
    """Generate complete report for a CSV file."""
    print(f"\n{'='*60}")
    print(f"Generating report: {title_suffix}")
    print(f"{'='*60}")
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    df = load_data(csv_path)
    if df is None or df.empty:
        print("No data to report.")
        return
    
    safe_suffix = title_suffix.lower().replace(' ', '_').replace('(', '').replace(')', '')
    
    sns.set_theme(style="whitegrid")
    
    # Generate plots for each method
    for prefix, method_name in [('P', 'Precise_Mask')]:
        plot_fst_distribution(df, prefix, title_suffix, 
                             os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_{method_name}_fst.png"))
        plot_mst_distribution(df, prefix, title_suffix,
                             os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_{method_name}_mst_box.png"))
        plot_mst_by_gender(df, prefix, title_suffix,
                          os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_{method_name}_mst_gender.png"))
        plot_fst_by_gender(df, prefix, title_suffix,
                          os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_{method_name}_fst_gender.png"))
    
    # Method comparison
    # plot_method_comparison(df, title_suffix,
    #                       os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_method_comparison.png"))
    
    # General demographics
    plot_gender_distribution(df, title_suffix,
                            os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_gender.png"))
    plot_race_distribution(df, title_suffix,
                          os.path.join(REPORTS_DIR, f"plot_{safe_suffix}_race.png"))
    
    # Text report
    generate_text_report(df, title_suffix,
                        os.path.join(REPORTS_DIR, f"report_{safe_suffix}.txt"))
    
    print(f"Report complete for: {title_suffix}")

def perform_statistical_tests(df, f):
    """
    Performs statistical tests between the two main models (Gemini Flash vs GPT-4 Image).
    Appends results to the file object f.
    """
    models = df['Model'].unique()
    if 'gemini_flash' not in models or 'gpt_image' not in models:
        return

    f.write("\n\n" + "=" * 70 + "\n")
    f.write("STATISTICAL SIGNIFICANCE (Gemini Flash vs GPT-4 Image)\n")
    f.write("=" * 70 + "\n")

    df_gemini = df[df['Model'] == 'gemini_flash']
    df_gpt = df[df['Model'] == 'gpt_image']

    # 1. AGE (T-Test)
    # T-test for independent samples (assuming normal distribution/large sample size)
    t_stat, p_val_age = stats.ttest_ind(df_gemini['Age'].dropna(), df_gpt['Age'].dropna(), equal_var=False)
    sig_age = "***" if p_val_age < 0.001 else "**" if p_val_age < 0.01 else "*" if p_val_age < 0.05 else "ns"
    f.write(f"\n1. Mean Age Difference:\n")
    f.write(f"   Gemini ({df_gemini['Age'].mean():.1f}) vs GPT ({df_gpt['Age'].mean():.1f})\n")
    f.write(f"   T-Statistic: {t_stat:.2f}, P-Value: {p_val_age:.4e} ({sig_age})\n")

    # 2. SKIN TONE (Mann-Whitney U Test)
    # Ordinal/Non-normal data -> Mann-Whitney U
    f.write(f"\n2. Skin Tone Differences (Mann-Whitney U):\n")
    
    for metric, name in [('P_MST_Index', 'MST'), ('P_PERLA_Index', 'PERLA'), ('P_FST_Type', 'FST')]:
        if metric in df.columns:
            # Drop NaNs just in case
            g_data = df_gemini[metric].dropna()
            gpt_data = df_gpt[metric].dropna()
            
            u_stat, p_val = stats.mannwhitneyu(g_data, gpt_data, alternative='two-sided')
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            
            f.write(f"   {name}: Mann-Whitney U={u_stat:.1f}, P-Value={p_val:.4e} ({sig})\n")
            f.write(f"      Mean Ranks: Gemini={g_data.mean():.2f} vs GPT={gpt_data.mean():.2f}\n")

    # 3. GENDER DISTRIBUTION (Chi-Square)
    f.write(f"\n3. Gender Distribution (Chi-Square):\n")
    # Create contingency table
    contingency_gender = pd.crosstab(df['Model'], df['Gender'])
    # Filter to only relevant models and columns if needed
    if 'gemini_flash' in contingency_gender.index and 'gpt_image' in contingency_gender.index:
        contingency_gender = contingency_gender.loc[['gemini_flash', 'gpt_image']]
        if 'Woman' in contingency_gender.columns and 'Man' in contingency_gender.columns:
            contingency_gender = contingency_gender[['Woman', 'Man']] # Ensure order/columns
            chi2, p_val_gender, dof, ex = stats.chi2_contingency(contingency_gender)
            sig_gen = "***" if p_val_gender < 0.001 else "**" if p_val_gender < 0.01 else "*" if p_val_gender < 0.05 else "ns"
            f.write(f"   Chi-Square Stat: {chi2:.2f}, P-Value: {p_val_gender:.4e} ({sig_gen})\n")
            f.write(f"   (Testing independence between Model and Gender)\n")

    # 4. RACE DISTRIBUTION (Chi-Square)
    f.write(f"\n4. Race Distribution (Chi-Square):\n")
    contingency_race = pd.crosstab(df['Model'], df['Race'])
    if 'gemini_flash' in contingency_race.index and 'gpt_image' in contingency_race.index:
        contingency_race = contingency_race.loc[['gemini_flash', 'gpt_image']]
        chi2_r, p_val_race, dof_r, ex_r = stats.chi2_contingency(contingency_race)
        sig_race = "***" if p_val_race < 0.001 else "**" if p_val_race < 0.01 else "*" if p_val_race < 0.05 else "ns"
        f.write(f"   Chi-Square Stat: {chi2_r:.2f}, P-Value: {p_val_race:.4e} ({sig_race})\n")

if __name__ == "__main__":
    generate_report_for_file(RESULTS_ORIGINAL_CSV, "Original (Raw)")
    generate_report_for_file(RESULTS_BALANCED_CSV, "Balanced Norm (Bg-Only WB)")
    print("\n✓ All reports generated!")

