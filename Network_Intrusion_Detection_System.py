import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind 
# To install the library
# pip install scikit-learn
# To use it in a Python script
import sklearn
from sklearn.linear_model import LinearRegression
#%%
# 1a. Load dataset with correct header
df = pd.read_csv('Train_data_updated.csv', header=0)
# 1b. List column names
print("\n--- Column Names ---")
print(df.columns.tolist())
# 1c. Check for missing or infinite values
print("\n--- Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "None")
print("\n--- Infinite Values ---")
inf_count = df.isin([np.inf, -np.inf]).sum()
print(inf_count[inf_count > 0] if inf_count.sum() > 0 else "None")
# 1d. Correct data types
numeric_cols = df.columns.drop(['Switch ID', 'Port Number', 'Label', 'Binary Label']).tolist()
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Switch ID'] = df['Switch ID'].astype(str)
df['Port Number'] = df['Port Number'].astype(str)
# 1e & 1f. Standardize labels
def clean_label(x):
    if pd.isna(x):
        return 'Unknown'
    x = str(x).strip().lower()
    x = ' '.join(x.split())
    # Normalize known variants
    if 'normal' in x:
        return 'Normal'
    elif 'attack' in x or 'attac' in x:
        return 'Attack'
    elif ('tcp' in x) and ('syn' in x or 'ysn' in x or 'sny' in x):
        return 'TCP-SYN'
    elif 'port' in x and ('scan' in x or 'scn' in x):
        return 'PortScan'
    elif 'over' in x and ('flow' in x or 'flw' in x):
        return 'Overflow'
    elif 'black' in x or 'hole' in x:
        return 'Blackhole'
    elif 'diver' in x or 'version' in x:
        return 'Diversion'
    else:
        return 'Unknown'
df['Label'] = df['Label'].apply(clean_label)
df['Binary Label'] = df['Binary Label'].apply(clean_label)
# Remove duplicates
initial = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"\n--- Removed {initial - len(df)} duplicate rows ---")

# --- Save cleaned data and reload to ensure consistency for Milestone 2 ---
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
clean_csv_path = os.path.join(script_dir, "Cleaned_Train_data.csv")
df.to_csv(clean_csv_path, index=False)
df = pd.read_csv(clean_csv_path)
 # Ensure 'Label' and 'Binary Label' remain strings after reload (CSV may change inferred types)
df['Label'] = df['Label'].astype(str)
df['Binary Label'] = df['Binary Label'].astype(str)

# ---------- SECTION 1g: Categorical Unique Value Analysis ----------
# Define categorical columns (including is_valid)
categorical_cols = ['Switch ID', 'Port Number', 'is_valid', 'Label', 'Binary Label']
categorical_cols = [col for col in categorical_cols if col in df.columns]
# Ensure all categorical columns are strings and cleaned
for col in categorical_cols:
    df[col] = df[col].astype(str).str.strip()
# Remove rows where Label is 'Unknown' (invalid/unrecognized labels)
initial_len = len(df)
df = df[df['Label'] != 'Unknown'].reset_index(drop=True)
removed_unknown = initial_len - len(df)
if removed_unknown > 0:
    print(f"\nRemoved {removed_unknown} rows with unrecognized/invalid labels ('Unknown').")
# Report unique counts for each categorical field
print("\nUnique value counts per categorical column:")
for col in categorical_cols:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} unique value(s)")
# Optional: Show full breakdown (uncomment if needed for debugging)
# print("\n--- Full Categorical Value Breakdown ---")
# for col in categorical_cols:
#     print(f"\n{col} values:\n{sorted(df[col].unique())}")
# 1h. Numeric stats (excluding non-numeric fields)
numeric_df = df.select_dtypes(include=[np.number])
exclude_cols = ['is_valid', 'Table ID', 'Active Flow Entries', 'Packets Looked Up', 'Packets Matched', 'Max Size']
# Identify numeric columns for statistics calculation
numeric_for_stats = df.select_dtypes(include=[np.number])

if not numeric_for_stats.empty:
    # RENAME the variable to avoid conflict with scipy.stats
    descriptive_stats_df = pd.DataFrame({
        'Max': numeric_for_stats.max(),
        'Min': numeric_for_stats.min(),
        'Mean': numeric_for_stats.mean(),
        'Variance': numeric_for_stats.var()
    })
    print("- Numeric Field Statistics -")
    print(descriptive_stats_df) # Use the new variable name
# 1i. Quartile stats
print("\n--- Quartile Statistics ---")
for col in numeric_for_stats.columns:
    try:
        df[f'{col}_quartile'] = pd.qcut(df[col], q=4, duplicates='drop')
        print(f"\n--- {col} ---")
        for q in sorted(df[f'{col}_quartile'].dropna().unique()):
            subset = df[df[f'{col}_quartile'] == q][col]
            print(f"  Quartile {q}: Max={subset.max():.2f}, Min={subset.min():.2f}, Mean={subset.mean():.2f}, Var={subset.var():.2f}")
        df.drop(columns=[f'{col}_quartile'], inplace=True)
    except Exception as e:
        print(f"  Skipping {col}: {e}")
# After cleaning 'Label' and before final output...
# Step 1: Identify all non-Normal unique attack types
attack_types = df[df['Label'] != 'Normal']['Label'].unique()
attack_types = [a for a in attack_types if a != 'Unknown']  # Optional: exclude unknown
print(f"\n--- Identified Attack Types: {sorted(attack_types)} ---")
# ----------------------------
# 2. Expand Attack Types (Req 2)
# ----------------------------
#%%
# Identify non-Normal attack types
attack_types = [lbl for lbl in df['Label'].unique() if lbl not in ['Normal', 'Unknown']]
print(f"\n--- Identified Attack Types: {sorted(attack_types)} ---")
# Create boolean columns
for attack in attack_types:
    df[attack] = (df['Label'] == attack).astype(int)
# Print attack occurrence table (boolean)
print("\n" + "="*50)
print(" ATTACK OCCURRENCE TABLE (Boolean Summary)")
print("="*50)
attack_occurred = pd.Series({attack: (df[attack].sum() > 0) for attack in attack_types}, name='Occurred')
attack_occurred.index.name = 'Attack Type'
print(attack_occurred.to_string())
# ----------------------------
# Define attack mask and column groups
# ----------------------------
is_attack = df['Label'] != 'Normal'
# Identify attack columns (boolean indicators)
attack_columns = [a for a in ['TCP-SYN', 'PortScan', 'Overflow', 'Blackhole', 'Diversion'] if a in df.columns]
# Categorical columns (treat as discrete)
categorical_cols = ['Switch ID', 'Port Number', 'Label', 'Binary Label'] + attack_columns
# Numeric columns (exclude categoricals and non-numeric)
numeric_cols = [
    col for col in df.columns
    if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
]
# Skip high-cardinality IDs in plots
def should_skip(col):
    if col in ['Switch ID', 'Port Number']:
        return df[col].nunique() > 20
    return False
#ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# ----------------------------
# 3. Plot PMF or PDF for Every Field (Req 3) — DISPLAY ONLY
# ----------------------------
#%%
# Define column groups
attack_columns = [a for a in attack_types if a in df.columns]
categorical_cols = ['Switch ID', 'Port Number', 'Label', 'Binary Label'] + attack_columns
# Numeric columns: exclude categoricals and ensure numeric dtype
numeric_cols = [
    col for col in df.columns
    if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
]
# Plot PMF for categorical (skip high-cardinality)
for col in categorical_cols:
    if col not in df.columns:
        continue
    unique_vals = df[col].nunique()
    if col in ['Switch ID', 'Port Number'] and unique_vals > 20:
        print(f"Skipped PMF for '{col}' (too many categories: {unique_vals})")
        continue
    pmf = df[col].value_counts(normalize=True)
    plt.figure(figsize=(8, 4))
    pmf.plot(kind='bar')
    plt.title(f'PMF of {col}')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()  #DISPLAY, not save
# Plot PDF (histogram) for numeric
for col in numeric_cols:
    data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        print(f"Skipped PDF for '{col}' (insufficient data)")
        continue
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=30, density=True, edgecolor='black')
    plt.title(f'PDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()  
# ----------------------------
# 4. Calculate and draw the CDF for each field (Req 4)
# ----------------------------
#%%
print("\n--- Generating CDF Plots (Basic Style) ---")
# Reuse column groups from earlier
attack_columns = [a for a in ['TCP-SYN', 'PortScan', 'Overflow', 'Blackhole', 'Diversion'] if a in df.columns]
categorical_cols = ['Switch ID', 'Port Number', 'Label', 'Binary Label'] + attack_columns
numeric_cols = [
    col for col in df.columns
    if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
]
# Plot CDF for categorical/discrete fields
for col in categorical_cols:
    if col not in df.columns:
        continue
    unique_vals = df[col].nunique()
    if col in ['Switch ID', 'Port Number'] and unique_vals > 20:
        print(f"Skipped CDF for '{col}' (too many categories: {unique_vals})")
        continue
    # Get value counts and sort index for consistent order
    counts = df[col].value_counts().sort_index()
    cdf = counts.cumsum() / counts.sum()
    plt.figure()
    plt.step(cdf.index, cdf.values, where='post')
    plt.title(f'CDF of {col}')
    plt.ylabel('Cumulative Probability')
    plt.xlabel(col)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
# Plot CDF for numeric fields
for col in numeric_cols:
    data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 2:
        print(f"Skipped CDF for '{col}' (insufficient data)")
        continue
    # Sort data and compute empirical CDF
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure()
    plt.plot(sorted_data, cdf, drawstyle='steps-post')
    plt.title(f'CDF of {col}')
    plt.xlabel(col)
    plt.ylabel('Cumulative Probability')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
# ----------------------------
# Requirement 5: Overlay Original vs Conditional PMF/PDF (Line Plots Only)
# ----------------------------
#%%
print("\n--- Requirement 5: Smooth Overlaid PMF/PDF (No Bars/Histograms) ---")
from scipy.stats import gaussian_kde
# Boolean mask: attack = Label != 'Normal'
is_attack = df['Label'] != 'Normal'
# Define column groups
attack_columns = [a for a in ['TCP-SYN', 'PortScan', 'Overflow', 'Blackhole', 'Diversion'] if a in df.columns]
categorical_cols = ['Switch ID', 'Port Number', 'Label', 'Binary Label'] + attack_columns
numeric_cols = [
    col for col in df.columns
    if col not in categorical_cols and pd.api.types.is_numeric_dtype(df[col])
]
# Helper: skip high-cardinality IDs
def should_skip(col):
    if col in ['Switch ID', 'Port Number']:
        return df[col].nunique() > 20
    return False
# Plot for categorical/discrete fields → line plot of PMF
for col in categorical_cols:
    if col not in df.columns or should_skip(col):
        continue
    # Full PMF
    pmf_full = df[col].value_counts(normalize=True).sort_index()
    # Conditional PMF
    pmf_cond = df[is_attack][col].value_counts(normalize=True).reindex(pmf_full.index, fill_value=0)
    plt.figure(figsize=(8, 4))
    plt.plot(pmf_full.index, pmf_full.values, 'o-', color='blue', label='Original (All)', linewidth=2)
    plt.plot(pmf_cond.index, pmf_cond.values, 's--', color='red', label='Conditional (Attack)', linewidth=2)
    plt.title(f' PMF of {col}')
    plt.ylabel('Probability')
    plt.xlabel(col)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
# Plot for numeric fields → KDE (smooth PDF)
for col in numeric_cols:
    data_full = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    data_cond = df[is_attack][col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data_full) < 10 or len(data_cond) < 10:
        print(f"Skipped {col}: insufficient data for KDE")
        continue
    # Common evaluation range
    x_min = min(data_full.min(), data_cond.min())
    x_max = max(data_full.max(), data_cond.max())
    x = np.linspace(x_min, x_max, 300)
    try:
        kde_full = gaussian_kde(data_full)
        y_full = kde_full(x)
    except:
        y_full = np.zeros_like(x)
    try:
        kde_cond = gaussian_kde(data_cond)
        y_cond = kde_cond(x)
    except:
        y_cond = np.zeros_like(x)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y_full, color='blue', label='Original (All)', linewidth=2)
    plt.plot(x, y_cond, color='red', label='Conditional (Attack)', linewidth=2, linestyle='--')
    plt.title(f'PDF of {col} ')
    plt.xlabel(col)
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()
# ----------------------------
# Requirement 6: Print scatter plot that indicate the relation between any two data fields
# ----------------------------
#%%
# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv('Train_data_updated.csv', header=0)
# Step 2: Show all column names so you can choose two
print("\n--- Available Columns ---")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")
# Step 3: Select two numeric columns (by name or index)
# Example: choose 'Received Packets' and 'Sent Packets'
print("\nEnter the names of two NUMERIC columns to plot (e.g., 'Received Packets', 'Sent Packets'):")
col_x = input("Enter X-axis column name: ").strip()
col_y = input("Enter Y-axis column name: ").strip()
# Validate that columns exist
if col_x not in df.columns:
    print(f"Error: '{col_x}' is not a valid column.")
    exit()
if col_y not in df.columns:
    print(f"Error: '{col_y}' is not a valid column.")
    exit()
# Step 4: Clean data — remove NaN and infinite values
df_clean = df[[col_x, col_y]].replace([float('inf'), float('-inf')], float('nan')).dropna()
# Step 5: Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df_clean[col_x], df_clean[col_y], alpha=0.6, s=20)
# Step 6: Add labels and title
plt.xlabel(col_x)
plt.ylabel(col_y)
plt.title(f'Scatter Plot: {col_x} vs {col_y}')
# Step 7: Show the plot
plt.tight_layout()
plt.show()
# ----------------------------
# Requirement 7:Calculate the joint PMF/pdf of any two different fields
# ----------------------------
#%%
# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv('Train_data_updated.csv', header=0)
# Step 2: Show all column names
print("\n--- Available Columns ---")
for i, col in enumerate(df.columns):
    print(f"{i}: {col}")
# Step 3: Let user choose two different columns
print("\nEnter the names of TWO DIFFERENT columns to compute their joint distribution:")
col1 = input("Enter first column name: ").strip()
col2 = input("Enter second column name: ").strip()
# Validate input
if col1 not in df.columns:
    print(f"Error: '{col1}' is not a valid column.")
    exit()
if col2 not in df.columns:
    print(f"Error: '{col2}' is not a valid column.")
    exit()
if col1 == col2:
    print("Error: Please choose two different columns.")
    exit()
# Step 4: Clean data — remove NaN and infinite values
df_clean = df[[col1, col2]].replace([np.inf, -np.inf], np.nan).dropna()
print(f"\nUsing {len(df_clean)} valid rows (after removing NaN/infinite values).")
# Step 5: Decide: PMF (discrete) or PDF (continuous)?
# Heuristic: if both columns have <= 20 unique values → treat as discrete (PMF)
n_unique1 = df_clean[col1].nunique()
n_unique2 = df_clean[col2].nunique()
is_discrete = (n_unique1 <= 20) and (n_unique2 <= 20)
# Step 6: Compute and plot joint distribution
if is_discrete:
    print(f"\nBoth columns appear discrete → computing JOINT PMF.")
    # Joint PMF: normalized contingency table
    joint_pmf = pd.crosstab(df_clean[col1], df_clean[col2], normalize='all')
    # Plot as heatmap (basic)
    plt.figure(figsize=(8, 6))
    plt.imshow(joint_pmf.values, cmap='Blues', aspect='auto')
    plt.colorbar(label='Joint Probability')
    plt.xticks(ticks=range(len(joint_pmf.columns)), labels=joint_pmf.columns, rotation=45)
    plt.yticks(ticks=range(len(joint_pmf.index)), labels=joint_pmf.index)
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.title(f'Joint PMF of {col1} and {col2}')
    plt.tight_layout()
    plt.show()
    # Optional: print table
    print("\nJoint PMF Table (first 10 rows/cols):")
    print(joint_pmf.iloc[:10, :10])
else:
    print(f"\nAt least one column is continuous → approximating JOINT PDF with 2D histogram.")
    x = df_clean[col1]
    y = df_clean[col2]
    plt.figure(figsize=(8, 6))
    plt.hist2d(x, y, bins=30, density=True, cmap='Blues')
    plt.colorbar(label='Density')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Joint PDF (2D Histogram) of {col1} and {col2}')
    plt.tight_layout()
    plt.show()
# ----------------------------
# Requirement 8:Calculate the joint PMF/pdf of any two different fields conditioned on the type of
#attach and find a way to visualize the difference
# ----------------------------
#%% 
# Step 1: Load and clean data (reusing your logic)
print("Loading dataset...")
df = pd.read_csv('Train_data_updated.csv', header=0)
# Clean Label (same as before)
def clean_label(x):
    if pd.isna(x):
        return 'Unknown'
    x = str(x).strip().lower()
    x = ' '.join(x.split())
    if 'normal' in x:
        return 'Normal'
    elif 'attack' in x or 'attac' in x:
        return 'Attack'
    elif ('tcp' in x) and ('syn' in x or 'ysn' in x or 'sny' in x):
        return 'TCP-SYN'
    elif 'port' in x and ('scan' in x or 'scn' in x):
        return 'PortScan'
    elif 'over' in x and ('flow' in x or 'flw' in x):
        return 'Overflow'
    elif 'black' in x or 'hole' in x:
        return 'Blackhole'
    elif 'diver' in x or 'version' in x:
        return 'Diversion'
    else:
        return 'Unknown'
df['Label'] = df['Label'].apply(clean_label)
# Step 2: Show available numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\n--- Available Numeric Columns ---")
for i, col in enumerate(numeric_cols):
    print(f"{i}: {col}")
# Step 3: Let user choose two fields
print("\nEnter the numbers or names of TWO numeric columns to analyze:")
try:
    idx1 = input("Enter first column index or name: ").strip()
    if idx1.isdigit():
        col1 = numeric_cols[int(idx1)]
    else:
        col1 = idx1
    idx2 = input("Enter second column index or name: ").strip()
    if idx2.isdigit():
        col2 = numeric_cols[int(idx2)]
    else:
        col2 = idx2
except (ValueError, IndexError):
    print("Invalid input. Using default columns.")
    col1, col2 = numeric_cols[0], numeric_cols[1]
print(f"\nSelected fields: '{col1}' and '{col2}'")
# Step 4: Choose attack type
attack_types = ['PortScan', 'TCP-SYN', 'Blackhole', 'Diversion', 'Overflow']
print("\nAvailable attack types:")
for i, atk in enumerate(attack_types):
    print(f"{i}: {atk}")
try:
    choice = input("\nEnter attack type index or name: ").strip()
    if choice.isdigit():
        attack = attack_types[int(choice)]
    else:
        attack = choice
except (ValueError, IndexError):
    attack = 'PortScan'
print(f"Conditioning on attack type: '{attack}'")
# Step 5: Clean data (remove inf/nan)
df_clean = df[[col1, col2, 'Label']].copy()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
# Step 6: Split data
all_data = df_clean
attack_data = df_clean[df_clean['Label'] == attack]
if len(attack_data) == 0:
    print(f"\n⚠️ No samples found for attack type '{attack}'. Exiting.")
    exit()
# Step 7: Plot joint PDF (2D histogram) - Full data
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist2d(all_data[col1], all_data[col2], bins=30, cmap='Blues')
plt.colorbar(label='Density')
plt.xlabel(col1)
plt.ylabel(col2)
plt.title('Joint PDF (All Data)')
# Step 8: Plot joint PDF - Attack-only
plt.subplot(1, 2, 2)
plt.hist2d(attack_data[col1], attack_data[col2], bins=30, cmap='Reds')
plt.colorbar(label='Density')
plt.xlabel(col1)
plt.ylabel(col2)
plt.title(f'Joint PDF (Attack: {attack})')
plt.tight_layout()
plt.show()
# Requirement 9: Calculate the correlation between different data fields
# ----------------------------
# %%
# ----------------------------
# Requirement 9: Calculate the correlation between different data fields
# ----------------------------
# %%
# Step 1: Load the dataset
print("Loading dataset...")
df = pd.read_csv('Train_data_updated.csv', header=0)
# Step 2: Clean the 'Label' column
def clean_label(x):
    if pd.isna(x):
        return 'Unknown'
    x = str(x).strip().lower()
    x = ' '.join(x.split())
    if 'normal' in x:
        return 'Normal'
    elif 'attack' in x or 'attac' in x:
        return 'Attack'
    elif ('tcp' in x) and ('syn' in x or 'ysn' in x):
        return 'TCP-SYN'
    elif 'port' in x and ('scan' in x or 'scn' in x):
        return 'PortScan'
    elif 'over' in x and ('flow' in x):
        return 'Overflow'
    elif 'black' in x or 'hole' in x:
        return 'Blackhole'
    elif 'diver' in x or 'version' in x:
        return 'Diversion'
    else:
        return 'Unknown'
df['Label'] = df['Label'].apply(clean_label)
# Step 3: Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])
# Exclude non-informative numeric columns
exclude_cols = ['is_valid', 'Table ID', 'Active Flow Entries', 'Packets Looked Up', 'Packets Matched', 'Max Size']
numeric_df = numeric_df.drop(columns=exclude_cols, errors='ignore')
print(f"\nUsing {len(numeric_df.columns)} numeric fields for correlation:")
print(numeric_df.columns.tolist())
# Step 4: Calculate Pearson correlation matrix
print("\n--- Calculating Correlation Matrix ---")
corr_matrix = numeric_df.corr(method='pearson')
# Step 5: Display full matrix
print("\n--- Full Correlation Matrix ---")
with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.width', None,
                       'display.max_colwidth', None):
    print(corr_matrix)
# Step 6: Plot heatmap — FIXED VERSION
print("\nDisplaying correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(corr_matrix, cmap='coolwarm')
# Colorbar
fig.colorbar(cax, label='Correlation')
# Ticks and labels
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)
# Title and layout
plt.title('Correlation Heatmap', pad=20)
plt.tight_layout()
plt.show()
#%%
# Load and clean data (reusing your logic)
df = pd.read_csv('Cleaned_Train_data.csv', header=0)
def clean_label(x):
    if pd.isna(x):
        return 'Unknown'
    x = str(x).strip().lower()
    x = ' '.join(x.split())
    if 'normal' in x:
        return 'Normal'
    elif 'attack' in x or 'attac' in x:
        return 'Attack'
    elif ('tcp' in x) and ('syn' in x or 'ysn' in x or 'sny' in x):
        return 'TCP-SYN'
    elif 'port' in x and ('scan' in x or 'scn' in x):
        return 'PortScan'
    elif 'over' in x and ('flow' in x or 'flw' in x):
        return 'Overflow'
    elif 'black' in x or 'hole' in x:
        return 'Blackhole'
    elif 'diver' in x or 'version' in x:
        return 'Diversion'
    else:
        return 'Unknown'
df['Label'] = df['Label'].apply(clean_label)
df = df[df['Label'] != 'Unknown'].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Identify numeric columns (exclude IDs and non-numeric metadata)
exclude_cols = {
    'Switch ID', 'Port Number', 'Label', 'Binary Label', 'is_valid',
    'Table ID', 'Active Flow Entries', 'Packets Looked Up',
    'Packets Matched', 'Max Size'
}
numeric_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns
    if col not in exclude_cols
]
# Create binary attack indicator: Normal vs. any attack
df['Is_Attack'] = (df['Label'] != 'Normal')
# Store p-values
p_values = {}
for col in numeric_cols:
    # Drop missing values for this column
    subset = df[[col, 'Is_Attack']].dropna()
    normal_data = subset[~subset['Is_Attack']][col]
    attack_data = subset[subset['Is_Attack']][col]
    # Skip if either group has fewer than 2 samples (needed for t-test)
    if len(normal_data) < 2 or len(attack_data) < 2:
        p_values[col] = 1.0  # Not significant
        continue
    # Perform Welch's t-test (does not assume equal variance)
    _, p_val = ttest_ind(normal_data, attack_data, equal_var=False)
    p_values[col] = p_val
# Convert to pandas Series and sort by p-value (ascending = more dependent)
p_series = pd.Series(p_values).sort_values()
# Print ONLY the top 15 most dependent fields
print("Lower p-value → stronger evidence of dependence on attack presence.\n")
print("Top 15 most dependent fields:")
print(p_series.head(15).to_string(float_format="{:.2e}".format))
#%%
#TASK 1
df = pd.read_csv('Cleaned_Train_data.csv', header=0)
# Assume 'df' is your cleaned DataFrame from Milestone 1
# Ensure 'Binary Label' is present and cleaned (e.g., 'Normal' -> 0, 'Attack' -> 1)
df['Binary Label'] = df['Binary Label'].apply(lambda x: 1 if x == 'Attack' else 0)
# Define features (exclude target columns and non-informative IDs)
exclude_cols = {'Label', 'Binary Label', 'Switch ID', 'Port Number', 'is_valid',
                'Table ID', 'Active Flow Entries', 'Packets Looked Up', 'Packets Matched', 'Max Size'}
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
# Separate features (X) and target (y)
X = df[feature_cols]
y = df['Binary Label']
# --- Manual Shuffle and Split ---
# Set seed for reproducibility (optional)
np.random.seed(42)
# Generate a random permutation of indices
shuffled_indices = np.random.permutation(X.index)
# Calculate split index for 70% train
split_idx = int(0.7 * len(X))
# Assign shuffled indices to train and test
train_indices = shuffled_indices[:split_idx]
test_indices = shuffled_indices[split_idx:]
# Split X and y using the shuffled indices
X_train = X.loc[train_indices]
y_train = y.loc[train_indices]
X_test = X.loc[test_indices]
y_test = y.loc[test_indices]
print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
# Calculate mean and std for each feature using the TRAINING set
# Use .values to get numpy arrays for direct computation if needed
mu_train = X_train.mean()
sigma_train = X_train.std()
# Avoid division by zero if std is 0 (e.g., constant feature)
sigma_train = sigma_train.replace(0, 1e-8)
# Compute Z-scores for the TEST set using training stats
z_scores_test = (X_test - mu_train) / sigma_train
# Define thresholds to test
thresholds = [2.0, 2.5, 3.0, 3.5]
# Store results for each threshold
zscore_results = {}
for th in thresholds:
    # Anomaly if ANY feature |z| > threshold
    # .any(axis=1) checks if any value in the row is True
    y_pred_test = (z_scores_test.abs() > th).any(axis=1).astype(int)
    # Store predictions for this threshold
    zscore_results[th] = y_pred_test
    # Optional: Print first few predictions for verification
print(f"\nThreshold {th}: First 10 predictions: {y_pred_test.head(10).values}")
def compute_metrics_manual(y_true, y_pred):
    """
    Manually compute TP, TN, FP, FN, Accuracy, Precision, Recall
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    total = len(y_true)
    # Handle division by zero if no predictions of a class were made
    accuracy = (TP + TN) / total if total > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    return accuracy, precision, recall, (TP, TN, FP, FN)
# Evaluate for each threshold
performance_metrics = {}
for th, y_pred in zscore_results.items():
    acc, prec, rec, counts = compute_metrics_manual(y_test, y_pred)
    performance_metrics[th] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'TP': counts[0], 'TN': counts[1], 'FP': counts[2], 'FN': counts[3]
    }
    print(f"\n--- Z-Score Threshold: {th} ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion Matrix: TP={counts[0]}, TN={counts[1]}, FP={counts[2]}, FN={counts[3]}")
# #%%
# #Task 2 
# #%%
import scipy.stats as stats
# --- ASSUMPTIONS ---
# The script has already executed Milestone 1 and Milestone 2 Task 1.
# Variables 'df', 'X_train', 'y_train' are available.
# 'df' is the original, fully cleaned DataFrame from Milestone 1.
# 'X_train' contains the numerical features for the 70% training set.
# 'y_train' contains the Binary Label (0=Normal, 1=Attack) for the 70% training set.

# Define the list of columns to exclude from features (these are not predictors)
exclude_cols_for_features = {
    'Label',               # Multi-class attack type
    'Binary Label',        # Target variable (should also be excluded from X_train)
    'Switch ID',           # Categorical ID
    'Port Number',         # Categorical ID
    'is_valid'             # Categorical flag (if present)
}

# Select ONLY the original numerical features for Task 2
feature_cols = [
    col for col in X_train.columns
    if pd.api.types.is_numeric_dtype(X_train[col]) # Ensure it's numeric
    and not pd.api.types.is_bool_dtype(X_train[col]) # Exclude bool
]

print(f"\nSelected numerical features for Task 2 ({len(feature_cols)}):")
print(feature_cols)

def fit_calc_and_plot_best_pdf(ax, data, feature_name="Feature", condition="All"):
    """
    Fits the best PDF to the data, calculates MSE for all, selects best fit, and plots the empirical histogram
    with the best-fit PDF curve.
    Returns the best distribution name, parameters, MSE, raw x_centers for best fit, and raw pdf_fitted values for best fit.
    """
    data_clean = data.dropna()
    if len(data_clean) < 10:
        ax.text(0.5, 0.5, 'Insufficient Data', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'{feature_name} - {condition}\n(Insufficient Data)')
        print(f"  -> Insufficient data for '{feature_name}' ({condition}).")
        return "Insufficient data (<10 points)", None, np.inf, None, None

    # Analyze the data's support (range)
    min_val = data_clean.min()
    max_val = data_clean.max()
    n_zeros = (data_clean == 0).sum()
    zero_ratio = n_zeros / len(data_clean)

    # Handle Constant Features
    if min_val == max_val:
        ax.hist(data_clean, bins=1, density=True, alpha=0.6, edgecolor='black')
        ax.axvline(min_val, color='red', linestyle='--', label=f'Constant = {min_val}')
        ax.legend()
        ax.set_title(f'{feature_name} - {condition}\n(Constant Value)')
        print(f"  -> Feature '{feature_name}' ({condition}) is constant ({min_val}).")
        return f"Constant value ({min_val})", None, 0.0, None, None # MSE is 0 for constant

    # Handle Zero-Inflated Features (Very Important for Network Data)
    if zero_ratio > 0.1: # Threshold for "many zeros"
        print(f"  Note: '{feature_name}' ({condition}) has {zero_ratio:.1%} zeros. Fitting on positive values only.")
        data_for_fitting = data_clean[data_clean > 0] # Adjust calculation range
        if len(data_for_fitting) < 10:
            ax.text(0.5, 0.5, 'Too many zeros', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f'{feature_name} - {condition}\n(Too many zeros)')
            print(f"  -> Too many zeros in '{feature_name}' ({condition}), insufficient positives for fitting.")
            return f"Too many zeros (only {len(data_for_fitting)} positives)", None, np.inf, None, None
    else:
        data_for_fitting = data_clean

    # Select Distributions Based on Data Range
    # Use ONLY distributions from the provided list
    if min_val >= 0:
        DISTRIBUTIONS = [
            # exponential
            stats.expon,
            # gamma
            stats.gamma,
            # chi (Not standard in scipy, chi2 is)
            # stats.chi, # Not a standard name in scipy.stats
            # chi2
            stats.chi2,
            # gompertz
            stats.gompertz,
            # maxwell
            stats.maxwell,
            # Rayleigh
            stats.rayleigh,
            # rice
            stats.rice,
            # half-normal (Not in original list, related to chi/scale)
            # stats.halfnorm, # Not explicitly in the list provided
            # half-logistic (Not in original list)
            # stats.halflogistic,
            # halfcauchy (Not in original list)
            # stats.halfcauchy,
            # reciprocal (requires finite bounds, tricky)
            # stats.reciprocal, # Excluded due to complexity of bounds
            # inverse gaussian
            stats.invgauss,
            # wald (Same as inverse gaussian)
            # stats.wald, # Same as invgauss
            # generalized gamma
            stats.gengamma,
            # lognormal
            stats.lognorm,
            # power log normal (powerlognorm)
            stats.powerlognorm,
            # gibrat (special case of lognorm)
            # stats.gibrat, # Same as lognorm with scale=1
            # fatigue life
            stats.fatiguelife,
            # inverted gamma
            stats.invgamma,
            # pareto
            stats.pareto,
            # pareto second kind (genpareto)
            stats.genpareto,
            # burr
            stats.burr,
            # burr12 (Same as burr in scipy)
            stats.burr12,
            # fisk (Same as log-logistic, related to burr)
            stats.fisk,
            # mielke's (scipy name is 'mielke')
            stats.mielke,
            # inverted weibull
            stats.invweibull,
            # exponential weibull
            stats.exponweib,
            # double pareto lognormal (scipy name might be different or not available)
            # stats.dblparetolognorm, # Not standard in scipy, excluded
        ]
        # Exclude lognorm/pareto/genpareto if zeros were present in original data (for fitting range)
        if zero_ratio > 0:
            DISTRIBUTIONS = [d for d in DISTRIBUTIONS if d != stats.lognorm and d != stats.pareto and d != stats.genpareto]
    else:
        # For data that can be negative (e.g., deltas), use different distributions
        # From the list: norm, t, laplace are suitable for potentially negative data
        DISTRIBUTIONS = [
            stats.norm,
            stats.t,
            stats.laplace,
            stats.uniform, # While uniform can be for any range, often used as a simple baseline
            # Note: chi, chi2, rayleigh, weibull_min, gamma, lognorm are not suitable here (defined for x >= loc or x >= 0)
            # Excluded based on range: exponential, pareto, maxwell, rice, gengamma, fisk, invweibull, exponweib
            # Excluded based on range: gompertz, invgauss, invgamma, burr, burr12, powerlognorm, fatiguelife
        ]

    # Create Empirical PDF (Histogram) for Fitting
    n_bins = min(100, len(data_for_fitting) // 2)
    if n_bins < 1:
         n_bins = 1 # Ensure at least one bin
    y_empirical, x_edges = np.histogram(data_for_fitting, bins=n_bins, density=True)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

    # Fit Distributions and Find the Best One (Lowest MSE)
    best_dist = None
    best_params = None
    best_mse = np.inf
    best_pdf = None
    best_pdf_x_centers = None # Store x_centers for the best fit

    # Store results for all distributions for later printing
    dist_results = []

    for dist in DISTRIBUTIONS:
        try:
            params = dist.fit(data_for_fitting)
            arg = params[:-2] if len(params) > 2 else []
            loc = params[-2]
            scale = params[-1]

            if scale <= 0:
                dist_results.append({'dist': dist, 'mse': None, 'params': params, 'pdf_fitted': None, 'x_centers': None, 'error': 'Invalid scale'})
                continue # Invalid scale parameter

            # Calculate the fitted PDF values at the bin centers
            pdf_fitted = dist.pdf(x_centers, loc=loc, scale=scale, *arg)

            # Calculate Mean Squared Error (MSE)
            mse = np.mean((y_empirical - pdf_fitted) ** 2)

            # Store results for this distribution
            dist_results.append({'dist': dist, 'mse': mse, 'params': params, 'pdf_fitted': pdf_fitted, 'x_centers': x_centers, 'error': None})

            if mse < best_mse:
                best_mse = mse
                best_dist = dist
                best_params = params
                best_pdf = pdf_fitted # Store the fitted curve for plotting
                best_pdf_x_centers = x_centers # Store the x values for the best fit

        except Exception as e:
            # Handle potential fitting errors or numerical issues during pdf calculation
            dist_results.append({'dist': dist, 'mse': None, 'params': None, 'pdf_fitted': None, 'x_centers': None, 'error': str(e)})
            continue

    # --- NOW PRINT THE RESULTS AFTER CALCULATIONS ARE DONE ---
    # Print Raw Values for the Best Fit PDF (if found)
    if best_dist is not None:
        print(f"\n--- Raw Values for Best Fit PDF: {best_dist.name} on '{feature_name}' ({condition}) ---")
        print(f"  x_centers (first 5): {best_pdf_x_centers[:5]}")
        print(f"  pdf_fitted (first 5): {best_pdf[:5]}")
        if len(best_pdf) > 5:
            print(f"  x_centers (last 5): {best_pdf_x_centers[-5:]}")
            print(f"  pdf_fitted (last 5): {best_pdf[-5:]}")
        print("--- End of Raw Values for Best Fit ---\n")
    else:
        print(f"\n--- Raw Values for Best Fit PDF on '{feature_name}' ({condition}) ---")
        print("  No suitable PDF found for any distribution.")
        print("--- End of Raw Values for Best Fit ---\n")
        return "No suitable PDF found", None, np.inf, None, None

    # Print Header for MSE comparison
    print(f"--- MSE Comparison for '{feature_name}' ({condition}) ---")
    print(f"  Distribution Name".ljust(20) + " | MSE")
    print("-" * 35)

    for result in dist_results:
        dist_name = result['dist'].name
        mse = result['mse']
        error = result['error']
        if error:
            print(f"  {dist_name.ljust(20)} | Skipped ({error})")
        elif mse is not None:
            print(f"  {dist_name.ljust(20)} | {mse:.2e}")
        else:
            # This case should ideally not happen if logic is correct, but added for robustness
            print(f"  {dist_name.ljust(20)} | Skipped (Unknown error)")

    # Print Summary of Best Fit
    if best_dist is not None:
        print("-" * 35)
        print(f"  BEST FIT: {best_dist.name.ljust(20)} | MSE: {best_mse:.2e}")
        print("--- End of MSE Comparison ---\n")
    else:
        print("-" * 35)
        print("  No suitable PDF found for any distribution.")
        print("--- End of MSE Comparison ---\n")

    # Plotting (Part of the requirement)
    ax.hist(data_clean, bins=n_bins, density=True, alpha=0.6, edgecolor='black', label='Empirical PDF')

    if best_dist is not None:
        ax.plot(best_pdf_x_centers, best_pdf, 'r-', linewidth=2, label=f'Best Fit: {best_dist.name}')
        ax.legend()
        ax.set_title(f'{feature_name} - {condition}\n(Best: {best_dist.name}, MSE: {best_mse:.2e})')
        return best_dist.name, best_params, best_mse, best_pdf_x_centers, best_pdf
    else:
        ax.set_title(f'{feature_name} - {condition}\n(No suitable PDF found)')
        print(f"  -> No suitable PDF found for '{feature_name}' ({condition}).")
        return "No suitable PDF found", None, np.inf, None, None


# --- MAIN EXECUTION FOR TASK 2, REQUIREMENT 1 (Calculations + Plots) ---
print("\n" + "="*100)
print("="*100)

pdf_results = []

for col in feature_cols:
    print(f"\n--- Analyzing feature: {col} ---")

    # Create a figure with 3 subplots (All, Normal, Attack)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'PDF Fitting for : {col}', fontsize=16)

    # 1. Calculate best PDF and plot for ALL training data (this calculates the PDF for the column alone)
    pdf_all, params_all, mse_all, _, _ = fit_calc_and_plot_best_pdf(axes[0], X_train[col], col, "All Data")
    print(f"  -> All Data:       Best PDF = {pdf_all}, MSE = {mse_all:.2e}")

    # 2. Calculate best PDF and plot for NORMAL traffic (y_train == 0) (conditioned on no anomaly)
    normal_mask = (y_train == 0)
    normal_data = X_train.loc[normal_mask, col]
    pdf_norm, params_norm, mse_norm, _, _ = fit_calc_and_plot_best_pdf(axes[1], normal_data, col, "Normal Traffic")
    print(f"  -> Normal Traffic: Best PDF = {pdf_norm}, MSE = {mse_norm:.2e}")

    # 3. Calculate best PDF and plot for ATTACK traffic (y_train == 1) (conditioned on an anomaly)
    attack_mask = (y_train == 1)
    attack_data = X_train.loc[attack_mask, col]
    pdf_attk, params_attk, mse_attk, _, _ = fit_calc_and_plot_best_pdf(axes[2], attack_data, col, "Attack Traffic")
    print(f"  -> Attack Traffic: Best PDF = {pdf_attk}, MSE = {mse_attk:.2e}")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() # Display the plots

    # Store results for the final summary
    pdf_results.append({
        'Feature': col,
        'Best PDF (All)': pdf_all,
        'Params (All)': params_all,
        'MSE (All)': mse_all,
        'Best PDF (Normal)': pdf_norm,
        'Params (Normal)': params_norm,
        'MSE (Normal)': mse_norm,
        'Best PDF (Attack)': pdf_attk,
        'Params (Attack)': params_attk,
        'MSE (Attack)': mse_attk,
    })

# --- FINAL SUMMARY FOR SUBMISSION ---
pdf_summary = pd.DataFrame(pdf_results)

print("\n" + "="*100)
print("FINAL SUMMARY FOR SUBMISSION")
print("Field Name, Best-Fit PDF, Parameters, and MSE for Each Condition")
print("="*100)

for _, row in pdf_summary.iterrows():
    print(f"\nFeature: {row['Feature']}")
    print(f"  All Data:       PDF = {row['Best PDF (All)']}, Params = {row['Params (All)']}, MSE = {row['MSE (All)']:.2e}")
    print(f"  Normal Traffic: PDF = {row['Best PDF (Normal)']}, Params = {row['Params (Normal)']}, MSE = {row['MSE (Normal)']:.2e}")
    print(f"  Attack Traffic: PDF = {row['Best PDF (Attack)']}, Params = {row['Params (Attack)']}, MSE = {row['MSE (Attack)']:.2e}")


#*********************************************************************

# --- ASSUMPTIONS ---
# The script has already executed Milestone 1, Milestone 2 Task 1, and Milestone 2 Task 2 Requirement 1.
# Variables 'df', 'X_train', 'y_train' are available.
# 'df' is the original, fully cleaned DataFrame from Milestone 1, containing 'Binary Label'.

# Identify categorical columns
categorical_candidates = ['Switch ID', 'Port Number', 'is_valid'] # Add others if needed
categorical_cols = [col for col in categorical_candidates if col in df.columns]

# Define a threshold for "high cardinality" to decide which columns to analyze
HIGH_CARDINALITY_THRESHOLD = 20 # Adjust based on your Milestone 1 decision

# Filter categorical columns based on cardinality
categorical_cols_to_analyze = []
for col in categorical_cols:
    unique_vals = df[col].nunique()
    if unique_vals <= HIGH_CARDINALITY_THRESHOLD:
        categorical_cols_to_analyze.append(col)
        print(f"Including categorical column for PMF analysis: '{col}' ({unique_vals} unique values)")
    else:
        print(f"Skipping high-cardinality categorical column: '{col}' ({unique_vals} unique values)")

print(f"\nCategorical columns selected for PMF storage: {categorical_cols_to_analyze}")

# --- CALCULATE AND STORE PMFs ---
print("\n" + "="*100)
print("MILESTONE 2 - TASK 2, REQUIREMENT 2: CATEGORICAL PMF STORAGE")
print("Calculating PMFs for categorical features (All, Normal, Attack).")
print("="*100)

# Create a master dictionary to hold all PMFs
pmf_storage = {}

# Use the original DataFrame 'df' which contains both features and the target 'Binary Label'
# We will calculate PMFs based on the TRAINING indices to be consistent with Task 2 Requirement 1
# Get the training data subset from the original df
df_train = df.loc[X_train.index] # Aligns with the indices used for X_train, y_train

for col in categorical_cols_to_analyze:
    print(f"\n--- Analyzing PMF for: {col} ---")

    # 1. PMF for ALL training data
    all_counts = df_train[col].value_counts(normalize=True) # normalize=True gives PMF
    pmf_all = all_counts.to_dict()

    # 2. Conditional PMF for NORMAL traffic (Binary Label == 0) within training data
    df_train_normal = df_train[df_train['Binary Label'] == 0]
    normal_counts = df_train_normal[col].value_counts(normalize=True)
    # Use reindex to ensure all categories from 'all_counts' are included, filling missing ones with 0
    pmf_normal = normal_counts.reindex(all_counts.index, fill_value=0).to_dict()

    # 3. Conditional PMF for ATTACK traffic (Binary Label == 1) within training data
    df_train_attack = df_train[df_train['Binary Label'] == 1]
    attack_counts = df_train_attack[col].value_counts(normalize=True)
    # Use reindex to ensure all categories from 'all_counts' are included, filling missing ones with 0
    pmf_attack = attack_counts.reindex(all_counts.index, fill_value=0).to_dict()

    # Store the calculated PMFs in the master dictionary
    pmf_storage[col] = {
        'PMF (All)': pmf_all,
        'PMF (Normal)': pmf_normal,
        'PMF (Attack)': pmf_attack
    }

# --- PRINT THE STRUCTURED PMF DATA FOR SUBMISSION ---
print("\n--- STORED PMFs FOR CATEGORICAL FEATURES ---")
for col, pmfs in pmf_storage.items():
    print(f"\n>>> Feature: '{col}' <<<")
    print("  PMF (All Training Data):")
    for category, prob in pmfs['PMF (All)'].items():
        print(f"    '{category}': {prob:.4f}")
    
    print("  PMF (Normal Traffic - Training Data):")
    for category, prob in pmfs['PMF (Normal)'].items():
        print(f"    '{category}': {prob:.4f}")
        
    print("  PMF (Attack Traffic - Training Data):")
    for category, prob in pmfs['PMF (Attack)'].items():
        print(f"    '{category}': {prob:.4f}")

# --- RESULT DOCUMENTATION ---
print("\n" + "="*100)
print("RESULT DOCUMENTATION FOR TASK 2, REQUIREMENT 2")
print("="*100)
print("PMF data for categorical columns has been calculated and stored in the 'pmf_storage' dictionary.")
print("The structure is: pmf_storage[feature_name]['PMF (All/Normal/Attack)'][category] = probability")
print("This format allows easy reference for future tasks (e.g., Milestone 3 Naive Bayes).")

# Example of how to access stored PMFs:
if pmf_storage:
    sample_feature = next(iter(pmf_storage)) # Get the first feature key
    sample_category = next(iter(pmf_storage[sample_feature]['PMF (All)'])) # Get a sample category
    prob_all = pmf_storage[sample_feature]['PMF (All)'][sample_category]
    prob_normal = pmf_storage[sample_feature]['PMF (Normal)'][sample_category]
    prob_attack = pmf_storage[sample_feature]['PMF (Attack)'][sample_category]
    print(f"\nExample Access - Feature: '{sample_feature}', Category: '{sample_category}':")
    print(f"  P('{sample_category}' | All) = {prob_all:.4f}")
    print(f"  P('{sample_category}' | Normal) = {prob_normal:.4f}")
    print(f"  P('{sample_category}' | Attack) = {prob_attack:.4f}")

#**********************************************************************
# --- ASSUMPTIONS ---
# The script has already executed Milestone 2 Task 2 Requirements 1 and 2.
# Variables 'pdf_summary' (from Req 1) and 'pmf_storage' (from Req 2) are available.

print("\n" + "="*100)
print("MILESTONE 2 - TASK 2, REQUIREMENT 3: RESULT DOCUMENTATION")
print("Summarizing PDF and PMF analysis results.")
print("="*100)

# --- 1. DOCUMENT NUMERICAL COLUMN ANALYSIS ---
print("\n--- 1. NUMERICAL COLUMNS PDF ANALYSIS SUMMARY ---")
print("This list contains each field name, best-fit PDF, and associated parameters for each condition.\n")

# The 'pdf_summary' DataFrame already contains the required information from Requirement 1.
# We just need to print it in a clear, list-like format as requested.

for index, row in pdf_summary.iterrows():
    feature_name = row['Feature']
    print(f"Feature: {feature_name}")
    print(f"  - Best PDF (All Data): {row['Best PDF (All)']}")
    print(f"    Parameters: {row['Params (All)']}")
    print(f"    MSE: {row['MSE (All)']:.2e}")
    
    print(f"  - Best PDF (Normal Traffic): {row['Best PDF (Normal)']}")
    print(f"    Parameters: {row['Params (Normal)']}")
    print(f"    MSE: {row['MSE (Normal)']:.2e}")
    
    print(f"  - Best PDF (Attack Traffic): {row['Best PDF (Attack)']}")
    print(f"    Parameters: {row['Params (Attack)']}")
    print(f"    MSE: {row['MSE (Attack)']:.2e}")
    print("-" * 40) # Separator line for readability

# --- 2. ORGANIZE CATEGORICAL COLUMN ANALYSIS ---
print("\n--- 2. CATEGORICAL COLUMNS PMF ANALYSIS SUMMARY ---")
print("PMF data for each categorical column is organized in a structured dictionary for future reference.\n")

# The 'pmf_storage' dictionary already contains the required information from Requirement 2.
# We print its structure and contents for documentation purposes.

for col, pmfs in pmf_storage.items():
    print(f"Categorical Feature: '{col}'")
    print(f"  PMF (All Training Data): {pmfs['PMF (All)']}")
    print(f"  PMF (Normal Traffic - Training Data): {pmfs['PMF (Normal)']}")
    print(f"  PMF (Attack Traffic - Training Data): {pmfs['PMF (Attack)']}")
    print("-" * 40) # Separator line for readability

# --- SUMMARY STATEMENT ---
print("\n--- SUMMARY ---")
print("1. The numerical analysis summary is provided above (list of features, best-fit PDFs, parameters, MSE).")
print("2. The categorical analysis data is organized in the 'pmf_storage' dictionary.")
print("   This dictionary structure allows easy reference in future tasks (e.g., Milestone 3 Naive Bayes).")
print(f"   Dictionary structure: pmf_storage[feature_name]['PMF (All/Normal/Attack)'][category] = probability")
print("   Example access: pmf_storage['is_valid']['PMF (Attack)'][1] gives P(is_valid=1 | Attack)")

# --- BONUS REQUIREMENT: MULTI-CLASS ANOMALY CHARACTERIZATION ---
# Define the helper function needed for raw PDF value output and MSE collection
def fit_calc_and_plot_best_pdf_raw_only(data, feature_name="Feature", condition="All"):
    """
    Fits PDFs to the data, calculates MSE for all, selects best fit, but does NOT print MSE here.
    Returns the best distribution name, parameters, MSE, raw x_centers for best fit,
    raw pdf_fitted values for best fit, and results for all distributions (for MSE printing later).
    """
    data_clean = data.dropna()
    if len(data_clean) < 10:
        print(f"  -> Insufficient data for '{feature_name}' ({condition}).")
        return "Insufficient data (<10 points)", None, np.inf, None, None, []

    # Analyze the data's support (range)
    min_val = data_clean.min()
    max_val = data_clean.max()
    n_zeros = (data_clean == 0).sum()
    zero_ratio = n_zeros / len(data_clean)

    # Handle Constant Features
    if min_val == max_val:
        print(f"  -> Feature '{feature_name}' ({condition}) is constant ({min_val}).")
        return f"Constant value ({min_val})", None, 0.0, None, None, [] # MSE is 0 for constant

    # Handle Zero-Inflated Features (Very Important for Network Data)
    if zero_ratio > 0.1: # Threshold for "many zeros"
        print(f"  Note: '{feature_name}' ({condition}) has {zero_ratio:.1%} zeros. Fitting on positive values only.")
        data_for_fitting = data_clean[data_clean > 0] # Adjust calculation range
        if len(data_for_fitting) < 10:
            print(f"  -> Too many zeros in '{feature_name}' ({condition}), insufficient positives for fitting.")
            return f"Too many zeros (only {len(data_for_fitting)} positives)", None, np.inf, None, None, []
    else:
        data_for_fitting = data_clean

    # Select Distributions Based on Data Range
    # Use ONLY distributions from the provided list
    if min_val >= 0:
        DISTRIBUTIONS = [
            # exponential
            stats.expon,
            # gamma
            stats.gamma,
            # chi2
            stats.chi2,
            # gompertz
            stats.gompertz,
            # maxwell
            stats.maxwell,
            # Rayleigh
            stats.rayleigh,
            # rice
            stats.rice,
            # inverse gaussian
            stats.invgauss,
            # generalized gamma
            stats.gengamma,
            # lognormal
            stats.lognorm,
            # power log normal (powerlognorm)
            stats.powerlognorm,
            # fatigue life
            stats.fatiguelife,
            # inverted gamma
            stats.invgamma,
            # pareto
            stats.pareto,
            # pareto second kind (genpareto)
            stats.genpareto,
            # burr
            stats.burr,
            # burr12
            stats.burr12,
            # fisk
            stats.fisk,
            # mielke's (scipy name is 'mielke')
            stats.mielke,
            # inverted weibull
            stats.invweibull,
            # exponential weibull
            stats.exponweib,
        ]
        # Exclude lognorm/pareto/genpareto if zeros were present in original data (for fitting range)
        if zero_ratio > 0:
            DISTRIBUTIONS = [d for d in DISTRIBUTIONS if d != stats.lognorm and d != stats.pareto and d != stats.genpareto]
    else:
        # For data that can be negative (e.g., deltas), use different distributions
        DISTRIBUTIONS = [
            stats.norm,
            stats.t,
            stats.laplace,
            stats.uniform,
        ]
        # Note: pareto, rayleigh, weibull_min, gamma, lognorm are not suitable here

    # Create Empirical PDF (Histogram) for Fitting
    n_bins = min(100, len(data_for_fitting) // 2)
    if n_bins < 1:
         n_bins = 1 # Ensure at least one bin
    y_empirical, x_edges = np.histogram(data_for_fitting, bins=n_bins, density=True)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0

    # Fit Distributions and Find the Best One (Lowest MSE)
    best_dist = None
    best_params = None
    best_mse = np.inf
    best_pdf = None
    best_pdf_x_centers = None # Store x_centers for the best fit

    # Store results for all distributions for later printing
    dist_results = []

    for dist in DISTRIBUTIONS:
        try:
            params = dist.fit(data_for_fitting)
            arg = params[:-2] if len(params) > 2 else []
            loc = params[-2]
            scale = params[-1]

            if scale <= 0:
                dist_results.append({'dist': dist, 'mse': None, 'params': params, 'pdf_fitted': None, 'x_centers': None, 'error': 'Invalid scale'})
                continue # Invalid scale parameter

            # Calculate the fitted PDF values at the bin centers
            pdf_fitted = dist.pdf(x_centers, loc=loc, scale=scale, *arg)

            # Calculate Mean Squared Error (MSE)
            mse = np.mean((y_empirical - pdf_fitted) ** 2)

            # Store results for this distribution
            dist_results.append({'dist': dist, 'mse': mse, 'params': params, 'pdf_fitted': pdf_fitted, 'x_centers': x_centers, 'error': None})

            if mse < best_mse:
                best_mse = mse
                best_dist = dist
                best_params = params
                best_pdf = pdf_fitted # Store the fitted curve for plotting
                best_pdf_x_centers = x_centers # Store the x values for the best fit

        except Exception as e:
            dist_results.append({'dist': dist, 'mse': None, 'params': None, 'pdf_fitted': None, 'x_centers': None, 'error': str(e)})
            continue

    return best_dist.name if best_dist else "No suitable PDF found", best_params, best_mse, best_pdf_x_centers, best_pdf, dist_results

# # --- BONUS REQUIREMENT EXECUTION ---
# print("\n" + "="*100)
# print("BONUS: MULTI-CLASS ANOMALY CHARACTERIZATION")
# print("Repeating PDF/PMF analysis for each specific attack type using the 'Label' column.")
# print("="*100)
# # Identify unique attack types from the 'Label' column (excluding 'Normal' and 'Unknown')
# attack_types = [lbl for lbl in df['Label'].unique() if lbl not in ['Normal', 'Unknown']]
# print(f"\nIdentified Attack Types for Analysis: {sorted(attack_types)}")
# # --- MULTI-CLASS PDF ANALYSIS (Requirement 1 logic adapted) ---
# print("\n--- MULTI-CLASS NUMERICAL COLUMNS PDF ANALYSIS ---")
# # Dictionary to store results for multi-class PDF analysis
# pdf_results_multiclass = {}
# for attack_type in attack_types:
#     print(f"\n--- Analyzing feature distributions for attack type: '{attack_type}' ---")
#     # Create a mask for the specific attack type in the original df
#     attack_mask = (df['Label'] == attack_type)
#     # Filter the original df to get data for this specific attack type
#     df_attack = df.loc[attack_mask]
#     # Filter X_train data based on the attack type mask applied to the original df's index
#     # This ensures we only analyze the part of the TRAINING set that corresponds to this attack type
#     df_train_attack = df.loc[X_train.index.intersection(df_attack.index)]
#     X_train_attack = df_train_attack[feature_cols] # Use the same feature_cols defined earlier
#     # Dictionary to store results for this specific attack type
#     pdf_results_multiclass[attack_type] = []
#     # Loop through the same numerical features as before
#     for col in feature_cols:
#         print(f"  Analyzing feature: {col} for attack type '{attack_type}'")
#         # Calculate best PDF for this attack type subset (conditioned on this specific attack)
#         # Use the X_train_attack subset
#         pdf_attk, params_attk, mse_attk, x_best_attk, y_best_attk, dist_results_attk = fit_calc_and_plot_best_pdf_raw_only(X_train_attack[col], col, f"Attack: {attack_type}")
#         print(f"    -> Attack '{attack_type}': Best PDF = {pdf_attk}, MSE = {mse_attk:.2e}")
#         # Store raw best fit data for printing later (similar to main task)
#         if x_best_attk is not None and y_best_attk is not None:
#             print(f"        --- Raw Values for Best Fit PDF: {pdf_attk} on '{col}' (Attack: {attack_type}) ---")
#             print(f"          x_centers (first 5): {x_best_attk[:5]}")
#             print(f"          pdf_fitted (first 5): {y_best_attk[:5]}")
#             if len(y_best_attk) > 5:
#                 print(f"          x_centers (last 5): {x_best_attk[-5:]}")
#                 print(f"          pdf_fitted (last 5): {y_best_attk[-5:]}")
#             print("        --- End of Raw Values for Best Fit ---")
#         else:
#             print(f"        --- Raw Values for Best Fit PDF on '{col}' (Attack: {attack_type}) ---")
#             print("          No suitable PDF found.")
#             print("        --- End of Raw Values for Best Fit ---")
#         # Store results for summary
#         pdf_results_multiclass[attack_type].append({
#             'Feature': col,
#             'Best PDF': pdf_attk,
#             'Params': params_attk,
#             'MSE': mse_attk,
#         })
# # Print MSE comparisons for multi-class PDF analysis (similar to main task structure)
# print("\n--- MULTI-CLASS MSE COMPARISON OUTPUT ---")
# for attack_type, results_list in pdf_results_multiclass.items():
#     print(f"\n--- MSE Comparison for Attack Type: {attack_type} ---")
#     for result in results_list:
#         col = result['Feature']
#         print(f"\n    Feature: {col}")
#         # Note: The raw fitting function fit_calc_and_plot_best_pdf_raw_only returns dist_results for the last call
#         # which was for the "Attack Traffic" condition in the main task. For multi-class, we'd need to call the fitting
#         # function again specifically for this attack type's training data.
#         # Re-run the MSE comparison logic for this specific attack type's data
#         attack_mask = (df['Label'] == attack_type)
#         df_train_attack = df.loc[X_train.index.intersection(df[attack_mask].index)]
#         X_train_attack = df_train_attack[feature_cols]
#         if col in X_train_attack.columns:
#             _, _, _, _, _, dist_results = fit_calc_and_plot_best_pdf_raw_only(X_train_attack[col], col, f"Attack: {attack_type}")
#             print(f"      Distribution Name".ljust(22) + " | MSE")
#             print("      " + "-" * 37)
#             for res in dist_results:
#                 dist_name = res['dist'].name
#                 mse = res['mse']
#                 error = res['error']
#                 if error:
#                     print(f"      {dist_name.ljust(22)} | Skipped ({error})")
#                 elif mse is not None:
#                     print(f"      {dist_name.ljust(22)} | {mse:.2e}")
#                 else:
#                     print(f"      {dist_name.ljust(22)} | Skipped (Unknown error)")
#             # Identify and print best fit for this feature and attack type
#             valid_results = [r for r in dist_results if r['mse'] is not None]
#             if valid_results:
#                 best_result = min(valid_results, key=lambda x: x['mse'])
#                 print("      " + "-" * 37)
#                 print(f"      BEST FIT: {best_result['dist'].name.ljust(22)} | MSE: {best_result['mse']:.2e}")
#             else:
#                 print("      No valid fits found for this feature/attack type combination.")
#         else:
#             print(f"      Feature {col} not found in X_train_attack for {attack_type}.")
# # --- MULTI-CLASS PMF ANALYSIS (Requirement 2 logic adapted) ---
# print("\n--- MULTI-CLASS CATEGORICAL COLUMNS PMF ANALYSIS ---")
# # Dictionary to store PMFs for multi-class analysis
# pmf_storage_multiclass = {}
# for attack_type in attack_types:
#     print(f"\n--- Calculating PMF for categorical features for attack type: '{attack_type}' ---")
#     # Create a mask for the specific attack type in the original df
#     attack_mask = (df['Label'] == attack_type)
#     # Filter the original df to get data for this specific attack type
#     df_attack = df.loc[attack_mask]
#     # Filter the training data subset for this attack type
#     df_train_attack = df.loc[X_train.index.intersection(df_attack.index)]
#     # Dictionary to store PMFs for this specific attack type
#     pmf_storage_multiclass[attack_type] = {}
#     for col in categorical_cols_to_analyze: # Use the categorical cols defined earlier
#         # Calculate PMF for this categorical column based *only* on the training data for this attack type
#         attack_counts = df_train_attack[col].value_counts(normalize=True)
#         # Use reindex to ensure all categories from the *overall* training data are included,
#         # filling missing ones with 0. This maintains consistency across different attack types' PMFs.
#         # First, get the full set of possible categories from the overall training data.
#         all_counts_overall = df.loc[X_train.index][col].value_counts(normalize=True)
#         pmf_attack_type_specific = attack_counts.reindex(all_counts_overall.index, fill_value=0).to_dict()
#         # Store the calculated PMF
#         pmf_storage_multiclass[attack_type][col] = pmf_attack_type_specific
# # Print the stored multi-class PMF data
# print("\n--- STORED PMFs FOR CATEGORICAL FEATURES BY ATTACK TYPE ---")
# for attack_type, pmfs_for_type in pmf_storage_multiclass.items():
#     print(f"\n>>> Attack Type: '{attack_type}' <<<")
#     for col, pmf_dict in pmfs_for_type.items():
#         print(f"  Feature: '{col}'")
#         for category, prob in pmf_dict.items():
#             print(f"    '{category}': {prob:.4f}")
# # --- MULTI-CLASS SUMMARY DOCUMENTATION ---
# print("\n" + "="*100)
# print("BONUS RESULT DOCUMENTATION: MULTI-CLASS CHARACTERIZATION")
# print("Summarizing PDF and PMF analysis results per attack type.")
# print("="*100)
# print("\n--- 1. MULTI-CLASS NUMERICAL COLUMNS PDF ANALYSIS SUMMARY ---")
# print("Best-fit PDF, parameters, and MSE for each feature, per attack type.\n")
# for attack_type, results_list in pdf_results_multiclass.items():
#     print(f">>> Attack Type: '{attack_type}' <<<")
#     for result in results_list:
#         feature_name = result['Feature']
#         print(f"  Feature: {feature_name}")
#         print(f"    Best PDF: {result['Best PDF']}")
#         print(f"    Parameters: {result['Params']}")
#         print(f"    MSE: {result['MSE']:.2e}")
#     print("-" * 40) # Separator line
# print("\n--- 2. MULTI-CLASS CATEGORICAL COLUMNS PMF ANALYSIS SUMMARY ---")
# print("PMF data for each categorical column, per attack type, stored in 'pmf_storage_multiclass'.\n")
# for attack_type, pmfs_for_type in pmf_storage_multiclass.items():
#     print(f">>> Attack Type: '{attack_type}' <<<")
#     for col, pmf_dict in pmfs_for_type.items():
#         print(f"  Categorical Feature: '{col}'")
#         print(f"    PMF: {pmf_dict}")
#     print("-" * 40) # Separator line

#%%
# ============================================================================
# TEST DATA CLEANING
# ============================================================================
print("\n" + "="*100)
print("CLEANING TEST DATASET")
print("="*100)

# --- Load the test dataset ---
test_file = 'test.csv'  # Replace with your actual test file name
print(f"\nLoading test dataset from: {test_file}")

try:
    df_test_raw = pd.read_csv(test_file, header=0)
    print(f"Test data loaded successfully. Shape: {df_test_raw.shape}")
except FileNotFoundError:
    print(f"ERROR: Test file '{test_file}' not found!")
    print("Please ensure the test dataset file is in the same directory as this script.")
    exit()

# --- Display initial info ---
print("\n--- Test Data Column Names ---")
print(df_test_raw.columns.tolist())

# --- Check for missing or infinite values ---
print("\n--- Missing Values in Test Data ---")
missing_test = df_test_raw.isnull().sum()
print(missing_test[missing_test > 0] if missing_test.sum() > 0 else "None")

print("\n--- Infinite Values in Test Data ---")
inf_count_test = df_test_raw.isin([np.inf, -np.inf]).sum()
print(inf_count_test[inf_count_test > 0] if inf_count_test.sum() > 0 else "None")

# --- Correct data types ---
print("\n--- Correcting Data Types ---")
numeric_cols_test = df_test_raw.columns.drop(['Switch ID', 'Port Number', 'Label', 'Binary Label'], errors='ignore').tolist()

for col in numeric_cols_test:
    df_test_raw[col] = pd.to_numeric(df_test_raw[col], errors='coerce')

# Convert categorical columns to string
if 'Switch ID' in df_test_raw.columns:
    df_test_raw['Switch ID'] = df_test_raw['Switch ID'].astype(str)
if 'Port Number' in df_test_raw.columns:
    df_test_raw['Port Number'] = df_test_raw['Port Number'].astype(str)

# --- Standardize labels ---
def clean_label(x):
    if pd.isna(x):
        return 'Unknown'
    x = str(x).strip().lower()
    x = ' '.join(x.split())
    if 'normal' in x:
        return 'Normal'
    elif 'attack' in x or 'attac' in x:
        return 'Attack'
    elif ('tcp' in x) and ('syn' in x or 'ysn' in x or 'sny' in x):
        return 'TCP-SYN'
    elif 'port' in x and ('scan' in x or 'scn' in x):
        return 'PortScan'
    elif 'over' in x and ('flow' in x or 'flw' in x):
        return 'Overflow'
    elif 'black' in x or 'hole' in x:
        return 'Blackhole'
    elif 'diver' in x or 'version' in x:
        return 'Diversion'
    else:
        return 'Unknown'

if 'Label' in df_test_raw.columns:
    df_test_raw['Label'] = df_test_raw['Label'].apply(clean_label)
if 'Binary Label' in df_test_raw.columns:
    df_test_raw['Binary Label'] = df_test_raw['Binary Label'].apply(clean_label)

# --- Remove duplicates ---
initial_test = len(df_test_raw)
df_test_raw = df_test_raw.drop_duplicates().reset_index(drop=True)
print(f"\n--- Removed {initial_test - len(df_test_raw)} duplicate rows from test data ---")

# --- Clean categorical columns ---
categorical_cols_test = ['Switch ID', 'Port Number', 'is_valid', 'Label', 'Binary Label']
categorical_cols_test = [col for col in categorical_cols_test if col in df_test_raw.columns]

for col in categorical_cols_test:
    df_test_raw[col] = df_test_raw[col].astype(str).str.strip()

# --- Remove Unknown labels ---
if 'Label' in df_test_raw.columns:
    initial_len_test = len(df_test_raw)
    df_test_raw = df_test_raw[df_test_raw['Label'] != 'Unknown'].reset_index(drop=True)
    removed_unknown_test = initial_len_test - len(df_test_raw)
    if removed_unknown_test > 0:
        print(f"\nRemoved {removed_unknown_test} rows with Unknown labels from test data.")

# --- Handle infinite values ---
print("\n--- Replacing infinite values with NaN ---")
df_test_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

# --- Save cleaned test data ---
script_dir = os.path.dirname(os.path.abspath(__file__))
clean_test_csv_path = os.path.join(script_dir, "Cleaned_Test_data.csv")
df_test_raw.to_csv(clean_test_csv_path, index=False)
print(f"\n--- Cleaned test data saved to: {clean_test_csv_path} ---")

# --- Reload ---
df_test_cleaned = pd.read_csv(clean_test_csv_path)

if 'Label' in df_test_cleaned.columns:
    df_test_cleaned['Label'] = df_test_cleaned['Label'].astype(str)
if 'Binary Label' in df_test_cleaned.columns:
    df_test_cleaned['Binary Label'] = df_test_cleaned['Binary Label'].astype(str)

print(f"\nCleaned test data shape: {df_test_cleaned.shape}")
print("\n" + "="*100)
print("TEST DATA CLEANING COMPLETED")
print("="*100)

# ============================================================================
# MILESTONE 3: TASK 1 - FROM SCRATCH NAIVE BAYES (FIXED VERSION)
# ============================================================================
print("\n" + "="*100)
print("MILESTONE 3 - TASK 1: FROM SCRATCH NAIVE BAYES ESTIMATION (FIXED)")
print("="*100)

# --- Load Training Data ---
print("\nLoading training data...")
df_train = pd.read_csv('Cleaned_Train_data.csv', header=0)
df_train['Label'] = df_train['Label'].astype(str)
df_train['Binary Label'] = df_train['Binary Label'].astype(str)

print(f"Training data loaded. Shape: {df_train.shape}")

# --- Define Features ---
exclude_cols = {'Label', 'Binary Label', 'Switch ID', 'Port Number', 'is_valid',
                'Table ID', 'Active Flow Entries', 'Packets Looked Up', 'Packets Matched', 'Max Size'}

feature_cols = [col for col in df_train.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[col])]
print(f"\nNumerical features: {len(feature_cols)} columns")

categorical_cols_nb = ['Switch ID', 'Port Number', 'is_valid']
categorical_cols_nb = [col for col in categorical_cols_nb if col in df_train.columns]
print(f"Categorical features: {categorical_cols_nb}")

# --- Prepare Training Data ---
X_train_full = df_train[feature_cols].copy()
y_train_full = df_train['Binary Label'].apply(lambda x: 1 if x in ['Attack', '1'] else 0)

# Replace any remaining inf/nan in training data
X_train_full.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train_full.fillna(X_train_full.median(), inplace=True)

print(f"\nTraining set - Normal: {(y_train_full == 0).sum()}, Attack: {(y_train_full == 1).sum()}")

# --- Calculate Priors with Laplace Smoothing ---
total_normal_train = (y_train_full == 0).sum() + 1  # Add 1 for Laplace
total_attack_train = (y_train_full == 1).sum() + 1
total_train = len(y_train_full) + 2  # Add 2 for both classes

prior_prob_normal = total_normal_train / total_train
prior_prob_attack = total_attack_train / total_train

print(f"\nPrior P(Normal): {prior_prob_normal:.4f}")
print(f"Prior P(Attack): {prior_prob_attack:.4f}")

# --- Prepare Test Data ---
print("\nPreparing test data...")
X_test_full = df_test_cleaned[feature_cols].copy()

# Replace inf/nan in test data using training statistics
X_test_full.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in feature_cols:
    X_test_full[col].fillna(X_train_full[col].median(), inplace=True)

if 'Binary Label' in df_test_cleaned.columns:
    y_test_full = df_test_cleaned['Binary Label'].apply(lambda x: 1 if x in ['Attack', '1'] else 0)
elif 'Label' in df_test_cleaned.columns:
    y_test_full = df_test_cleaned['Label'].apply(lambda x: 1 if x != 'Normal' else 0)

print(f"Test set - Normal: {(y_test_full == 0).sum()}, Attack: {(y_test_full == 1).sum()}")
print(f"Test set size: {len(y_test_full)} rows")

# --- Improved Distribution Fitting with Better Error Handling ---
def fit_best_pdf_robust(data, feature_name, condition):
    """Robust PDF fitting with better handling of edge cases."""
    data_clean = data.dropna()
    
    if len(data_clean) < 10:
        return "uniform", [data_clean.min() if len(data_clean) > 0 else 0, 
                          data_clean.max() - data_clean.min() if len(data_clean) > 0 else 1], 1e-6
    
    min_val = data_clean.min()
    max_val = data_clean.max()
    
    if min_val == max_val:
        return f"constant_{min_val}", [min_val], 0.0
    
    # Calculate statistics
    mean_val = data_clean.mean()
    std_val = data_clean.std()
    
    # Handle zero-inflated data
    n_zeros = (data_clean == 0).sum()
    zero_ratio = n_zeros / len(data_clean)
    
    if zero_ratio > 0.3:  # More than 30% zeros
        data_for_fitting = data_clean[data_clean > 0]
        if len(data_for_fitting) < 10:
            # Use uniform distribution as fallback
            return "uniform", [min_val, max_val - min_val], 1e-6
    else:
        data_for_fitting = data_clean
    
    # Try simple distributions first (more robust)
    DISTRIBUTIONS = [
        stats.norm,      # Normal distribution
        stats.expon,     # Exponential
        stats.gamma,     # Gamma
        stats.lognorm,   # Log-normal
        stats.uniform    # Uniform (fallback)
    ]
    
    best_dist = None
    best_params = None
    best_mse = np.inf
    
    for dist in DISTRIBUTIONS:
        try:
            # Fit distribution
            if dist == stats.uniform:
                params = (min_val, max_val - min_val)
            else:
                params = dist.fit(data_for_fitting)
            
            # Validate parameters
            if dist in [stats.expon, stats.gamma, stats.lognorm]:
                if min_val < 0:  # These distributions don't support negative values
                    continue
            
            # Test the fit with a sample
            test_val = data_for_fitting.iloc[len(data_for_fitting)//2]
            try:
                test_pdf = dist.pdf(test_val, *params)
                if np.isnan(test_pdf) or np.isinf(test_pdf) or test_pdf < 0:
                    continue
            except:
                continue
            
            # Calculate MSE on a histogram
            n_bins = min(50, len(data_for_fitting) // 10)
            if n_bins < 5:
                n_bins = 5
            
            y_empirical, x_edges = np.histogram(data_for_fitting, bins=n_bins, density=True)
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
            
            pdf_fitted = dist.pdf(x_centers, *params)
            
            # Check for valid PDF values
            if np.any(np.isnan(pdf_fitted)) or np.any(np.isinf(pdf_fitted)) or np.any(pdf_fitted < 0):
                continue
            
            mse = np.mean((y_empirical - pdf_fitted) ** 2)
            
            if mse < best_mse and mse < 1000:  # Sanity check
                best_mse = mse
                best_dist = dist
                best_params = params
                
        except Exception as e:
            continue
    
    # Fallback to normal distribution if nothing works
    if best_dist is None:
        try:
            best_dist = stats.norm
            best_params = (mean_val, std_val if std_val > 0 else 1.0)
            best_mse = 1e-3
        except:
            best_dist = stats.uniform
            best_params = (min_val, max_val - min_val)
            best_mse = 1e-6
    
    return best_dist.name, best_params, best_mse

# --- Fit Distributions ---
print("\nFitting distributions (this may take a moment)...")
pdf_fitted_distributions = {}

for col in feature_cols:
    # Fit for Normal
    normal_data = df_train[y_train_full == 0][col]
    dist_norm, params_norm, mse_norm = fit_best_pdf_robust(normal_data, col, "Normal")
    
    # Fit for Attack
    attack_data = df_train[y_train_full == 1][col]
    dist_attk, params_attk, mse_attk = fit_best_pdf_robust(attack_data, col, "Attack")
    
    pdf_fitted_distributions[col] = {
        'Normal': {'dist': dist_norm, 'params': params_norm, 'mse': mse_norm},
        'Attack': {'dist': dist_attk, 'params': params_attk, 'mse': mse_attk}
    }

print("Distribution fitting completed.")

# --- Fit PMFs with Better Smoothing ---
print("\nFitting PMFs for categorical features...")
pmf_fitted_distributions = {}

for col in categorical_cols_nb:
    # Get all unique categories from both train and test
    all_categories_train = set(df_train[col].astype(str).unique())
    all_categories_test = set(df_test_cleaned[col].astype(str).unique())
    all_categories = all_categories_train.union(all_categories_test)
    
    # PMF for Normal with Laplace smoothing
    df_train_normal = df_train[y_train_full == 0]
    normal_counts = df_train_normal[col].value_counts()
    n_normal = len(df_train_normal)
    alpha = 1.0  # Laplace smoothing parameter
    
    pmf_normal = {}
    for cat in all_categories:
        count = normal_counts.get(cat, 0)
        pmf_normal[str(cat)] = (count + alpha) / (n_normal + alpha * len(all_categories))
    
    # PMF for Attack with Laplace smoothing
    df_train_attack = df_train[y_train_full == 1]
    attack_counts = df_train_attack[col].value_counts()
    n_attack = len(df_train_attack)
    
    pmf_attack = {}
    for cat in all_categories:
        count = attack_counts.get(cat, 0)
        pmf_attack[str(cat)] = (count + alpha) / (n_attack + alpha * len(all_categories))
    
    pmf_fitted_distributions[col] = {
        'Normal': pmf_normal,
        'Attack': pmf_attack
    }

print("PMF fitting completed.")

# --- Improved Naive Bayes Calculation with Numerical Stability ---
def calculate_log_likelihood_robust(x_val, dist_name, params):
    """Calculate log likelihood with robust error handling."""
    try:
        # Handle constant distributions
        if 'constant' in str(dist_name):
            const_val = params[0]
            if abs(x_val - const_val) < 1e-6:
                return 0.0
            else:
                return -100.0
        
        # Handle named distributions
        if dist_name in ['norm', 'expon', 'gamma', 'lognorm', 'uniform']:
            dist_obj = getattr(stats, dist_name)
            
            # Special handling for distributions with domain restrictions
            if dist_name in ['expon', 'gamma', 'lognorm'] and x_val < params[0]:
                return -50.0
            
            log_lik = dist_obj.logpdf(x_val, *params)
            
            # Handle edge cases
            if np.isnan(log_lik) or np.isinf(log_lik):
                return -50.0
            
            # Clamp to reasonable range
            return max(min(log_lik, 10.0), -50.0)
        
        return -30.0  # Default for unknown distributions
        
    except Exception as e:
        return -30.0

def calculate_naive_bayes_prob_robust(row_data, test_row_idx, class_label):
    """Calculate Naive Bayes probability with improved numerical stability."""
    if class_label == 0:
        prior_prob = prior_prob_normal
        condition_key = 'Normal'
    else:
        prior_prob = prior_prob_attack
        condition_key = 'Attack'
    
    # Start with log prior
    log_prob = np.log(prior_prob)
    
    # Add likelihoods from numerical features
    for col in feature_cols:
        x_val = row_data[col]
        
        dist_info = pdf_fitted_distributions[col][condition_key]
        dist_name = dist_info['dist']
        params = dist_info['params']
        
        log_lik = calculate_log_likelihood_robust(x_val, dist_name, params)
        log_prob += log_lik * 0.5  # Reduce weight to prevent dominance
    
    # Add likelihoods from categorical features
    for col in categorical_cols_nb:
        cat_val = str(df_test_cleaned.loc[test_row_idx, col])
        pmf_probs = pmf_fitted_distributions[col][condition_key]
        
        prob = pmf_probs.get(cat_val, 1.0 / len(pmf_probs))  # Better default
        log_prob += np.log(prob)
    
    return log_prob

# --- Metrics Calculation ---
def compute_metrics_manual(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return accuracy, precision, recall, (tp, tn, fp, fn)

# --- Make Predictions ---
print("\n--- Making Predictions on Test Dataset ---")
y_pred_nb_test = []

# Sample predictions for debugging
debug_sample_size = min(5, len(X_test_full))
debug_predictions = []

for idx in range(len(X_test_full)):
    test_row_idx = X_test_full.index[idx]
    row_data = X_test_full.iloc[idx]
    
    log_prob_normal = calculate_naive_bayes_prob_robust(row_data, test_row_idx, 0)
    log_prob_attack = calculate_naive_bayes_prob_robust(row_data, test_row_idx, 1)
    
    # Store debug info for first few predictions
    if idx < debug_sample_size:
        debug_predictions.append({
            'idx': idx,
            'log_prob_normal': log_prob_normal,
            'log_prob_attack': log_prob_attack,
            'actual': y_test_full.iloc[idx]
        })
    
    # Predict with threshold adjustment
    # Adjust threshold based on class imbalance
    threshold_adjustment = np.log(prior_prob_attack / prior_prob_normal) * 0.5
    
    if log_prob_attack >= (log_prob_normal - threshold_adjustment):
        y_pred_nb_test.append(1)
    else:
        y_pred_nb_test.append(0)
    
    if (idx + 1) % 5000 == 0:
        print(f"  Processed {idx + 1}/{len(X_test_full)} rows...")

y_pred_nb_test = np.array(y_pred_nb_test)

# --- Debug Output ---
print("\n--- Debug: Sample Predictions ---")
for pred in debug_predictions:
    print(f"Row {pred['idx']}: log_P(Normal)={pred['log_prob_normal']:.2f}, "
          f"log_P(Attack)={pred['log_prob_attack']:.2f}, "
          f"Predicted={'Attack' if pred['log_prob_attack'] >= pred['log_prob_normal'] else 'Normal'}, "
          f"Actual={'Attack' if pred['actual'] == 1 else 'Normal'}")

# --- Evaluate ---
acc_nb, prec_nb, rec_nb, counts_nb = compute_metrics_manual(y_test_full.values, y_pred_nb_test)

print("\n" + "="*100)
print("CUSTOM NAIVE BAYES (FROM SCRATCH) RESULTS ON TEST DATASET")
print("="*100)
print(f"\nAccuracy:  {acc_nb:.4f}")
print(f"Precision: {prec_nb:.4f}")
print(f"Recall:    {rec_nb:.4f}")
print(f"\nConfusion Matrix:")
print(f"  True Positives (TP):  {counts_nb[0]}")
print(f"  True Negatives (TN):  {counts_nb[1]}")
print(f"  False Positives (FP): {counts_nb[2]}")
print(f"  False Negatives (FN): {counts_nb[3]}")

print(f"\n--- Diagnostic Information ---")
print(f"Total predictions: {len(y_pred_nb_test)}")
print(f"Predicted as Attack (1): {sum(y_pred_nb_test == 1)} ({sum(y_pred_nb_test == 1)/len(y_pred_nb_test)*100:.1f}%)")
print(f"Predicted as Normal (0): {sum(y_pred_nb_test == 0)} ({sum(y_pred_nb_test == 0)/len(y_pred_nb_test)*100:.1f}%)")
print(f"Actual Attacks in test set: {sum(y_test_full == 1)} ({sum(y_test_full == 1)/len(y_test_full)*100:.1f}%)")
print(f"Actual Normal in test set: {sum(y_test_full == 0)} ({sum(y_test_full == 0)/len(y_test_full)*100:.1f}%)")

print("\n" + "="*100)
print("MILESTONE 3 TASK 1 COMPLETED")
print("="*100)
#%%
#%%
# --- MILESTONE 3: TASK 2 - CATEGORICAL FEATURE ENCODING AND MACHINE LEARNING MODEL ---
# Objective: Encode categorical features and evaluate various Naive Bayes models from Scikit-Learn.
# Steps:
# 1. Use one-hot encoding to encode all categorical features.
# 2. Train GaussianNB, MultinomialNB, and BernoulliNB models on the encoded data.
# 3. Evaluate each model using Accuracy, and Precision, and Recall.

print("\n" + "="*100)
print("MILESTONE 3 - TASK 2: CATEGORICAL FEATURE ENCODING AND MACHINE LEARNING MODEL")
print("="*100)

# --- Step 1: Identify Categorical Features for Encoding ---
# Based on the provided code snippets, common categorical columns are 'Switch ID', 'Port Number', 'is_valid'
# We'll use the same list as defined in MS2 for consistency.
categorical_candidates = ['Switch ID', 'Port Number', 'is_valid'] # Add others if needed based on your dataset
categorical_cols_to_encode = [col for col in categorical_candidates if col in df.columns]
print(f"Categorical columns identified for encoding: {categorical_cols_to_encode}")

# --- Step 2: One-Hot Encode Categorical Features ---
# Combine numerical features with one-hot encoded categorical features
# Start with the numerical features from X_train and X_test
X_train_numerical = X_train.copy()
X_test_numerical = X_test.copy()

# Create a copy of the full DataFrame for the training set to access categorical values
df_train_full = df.loc[X_train.index].copy() # Ensure we have the original rows corresponding to X_train

# Apply one-hot encoding to the categorical features in the training set
if categorical_cols_to_encode:
    print("\nEncoding categorical features using one-hot encoding...")
    X_train_categorical = pd.get_dummies(df_train_full[categorical_cols_to_encode], drop_first=False) # Keep all dummy variables
    
    # Ensure the same column structure for the test set
    df_test_full = df.loc[X_test.index].copy() # Get the original rows for the test set
    X_test_categorical = pd.get_dummies(df_test_full[categorical_cols_to_encode], drop_first=False)

    # Align columns between train and test sets (important for ML models)
    # This ensures both sets have the same dummy variable columns, filling missing ones with 0
    X_train_categorical, X_test_categorical = X_train_categorical.align(X_test_categorical, join='outer', axis=1, fill_value=0)

    # Concatenate numerical and encoded categorical features
    X_train_encoded = pd.concat([X_train_numerical, X_train_categorical], axis=1)
    X_test_encoded = pd.concat([X_test_numerical, X_test_categorical], axis=1)

    print(f"Shape of X_train_encoded: {X_train_encoded.shape}")
    print(f"Shape of X_test_encoded: {X_test_encoded.shape}")
    
    # DEBUG: Check for potential data leakage - ensure target is not in feature set
    print(f"Target column: {y_train.name if hasattr(y_train, 'name') else 'unknown'}")
    print(f"Target column in X_train_encoded: {y_train.name in X_train_encoded.columns if hasattr(y_train, 'name') else 'Cannot check'}")
    
else:
    print("No categorical columns found for encoding. Using only numerical features.")
    X_train_encoded = X_train_numerical
    X_test_encoded = X_test_numerical

# --- Step 3: Train and Evaluate Scikit-Learn Naive Bayes Models ---

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler

# Initialize models
model_gnb = GaussianNB() # Assumes continuous features follow a Gaussian distribution
model_mnb = MultinomialNB() # Assumes discrete count features (often used for text, but using as requested)
model_bnb = BernoulliNB() # Assumes binary/boolean features

# Dictionary to store results
sklearn_nb_results = {}

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train a model and return its performance metrics."""
    print(f"\nTraining and evaluating {model_name}...")
    # Fit the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Calculate metrics
    acc, prec, rec, counts = compute_metrics_manual(y_test, y_pred)
    print(f"{model_name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    return {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Confusion_Matrix': counts,
        'Model_Object': model # Store the fitted model object if needed later
    }

# Train and evaluate GaussianNB
# Apply feature scaling to improve GaussianNB performance and get realistic metrics
from sklearn.preprocessing import StandardScaler

scaler_gnb = StandardScaler()
X_train_gnb = scaler_gnb.fit_transform(X_train_encoded)
X_test_gnb = scaler_gnb.transform(X_test_encoded)

# Convert back to DataFrame to maintain structure
X_train_gnb = pd.DataFrame(X_train_gnb, columns=X_train_encoded.columns, index=X_train_encoded.index)
X_test_gnb = pd.DataFrame(X_test_gnb, columns=X_test_encoded.columns, index=X_test_encoded.index)

sklearn_nb_results['GaussianNB'] = train_and_evaluate_model(model_gnb, "GaussianNB", X_train_gnb, y_train, X_test_gnb, y_test)

# Train and evaluate MultinomialNB
# Convert features to non-negative values as required by MultinomialNB
try:
    # For MultinomialNB, we need non-negative features
    # Apply Min-Max scaling to ensure all values are non-negative
    from sklearn.preprocessing import MinMaxScaler
    
    scaler_mnb = MinMaxScaler()
    X_train_mnb = scaler_mnb.fit_transform(X_train_encoded)
    X_test_mnb = scaler_mnb.transform(X_test_encoded)
    
    # Convert to DataFrame to maintain structure
    X_train_mnb = pd.DataFrame(X_train_mnb, columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_test_mnb = pd.DataFrame(X_test_mnb, columns=X_test_encoded.columns, index=X_test_encoded.index)
    
    sklearn_nb_results['MultinomialNB'] = train_and_evaluate_model(model_mnb, "MultinomialNB", X_train_mnb, y_train, X_test_mnb, y_test)
except Exception as e:
    print(f"Error training MultinomialNB: {e}")
    sklearn_nb_results['MultinomialNB'] = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Confusion_Matrix': (0, 0, 0, 0), 'Model_Object': None}

# Train and evaluate BernoulliNB
# For BernoulliNB, features should represent presence/absence (binary)
try:
    # For BernoulliNB, we need to handle the data more carefully
    # Mix of one-hot encoded categorical (already binary) and numerical features
    
    # Standardize the entire feature set first
    scaler_bnb = StandardScaler()
    X_train_scaled = scaler_bnb.fit_transform(X_train_encoded)
    X_test_scaled = scaler_bnb.transform(X_test_encoded)
    
    # Convert to DataFrame
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns, index=X_train_encoded.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns, index=X_test_encoded.index)
    
    # Binarize based on mean (0 after standardization)
    # This creates a more balanced binarization
    X_train_bnb = (X_train_scaled_df > 0).astype(int)
    X_test_bnb = (X_test_scaled_df > 0).astype(int)
    
    sklearn_nb_results['BernoulliNB'] = train_and_evaluate_model(model_bnb, "BernoulliNB", X_train_bnb, y_train, X_test_bnb, y_test)
except Exception as e:
    print(f"Error training BernoulliNB: {e}")
    sklearn_nb_results['BernoulliNB'] = {'Accuracy': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'Confusion_Matrix': (0, 0, 0, 0), 'Model_Object': None}

# --- Step 4: Compare Model Performances ---
print("\n" + "="*100)
print("COMPARISON OF NAIVE BAYES MODELS")
print("="*100)

# Print a summary table
print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
print("-" * 50)
for model_name, results in sklearn_nb_results.items():
    acc = results['Accuracy']
    prec = results['Precision']
    rec = results['Recall']
    print(f"{model_name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")

# Determine which model provides better results (based on chosen metric, often Recall for anomaly detection)
best_model_by_recall = max(sklearn_nb_results.keys(), key=lambda k: sklearn_nb_results[k]['Recall'])
best_model_by_precision = max(sklearn_nb_results.keys(), key=lambda k: sklearn_nb_results[k]['Precision'])
best_model_by_accuracy = max(sklearn_nb_results.keys(), key=lambda k: sklearn_nb_results[k]['Accuracy'])

print(f"\nBest Model by Recall: {best_model_by_recall} (Recall={sklearn_nb_results[best_model_by_recall]['Recall']:.4f})")
print(f"Best Model by Precision: {best_model_by_precision} (Precision={sklearn_nb_results[best_model_by_precision]['Precision']:.4f})")
print(f"Best Model by Accuracy: {best_model_by_accuracy} (Accuracy={sklearn_nb_results[best_model_by_accuracy]['Accuracy']:.4f})")

# Justify findings (Example justification, adapt based on your actual results)
print("\n--- Justification of Findings ---")
print("The choice of the best model depends on the specific goal of the anomaly detection system:")
print(" - **High Recall** is crucial if the primary goal is to catch as many attacks as possible, even at the cost of some false alarms.")
print(" - **High Precision** is important if minimizing false alarms (FP) is critical, as each alarm requires investigation.")
print(" - **High Accuracy** is a general measure of overall correctness, but can be misleading in imbalanced datasets.")
print(f"Based on the results, {best_model_by_recall} achieved the highest Recall, making it potentially the best choice for an initial anomaly detection system focused on not missing attacks.")

#%%
print("\nMilestone 3 Task 2 completed.")

#%%
#%%
# ============================================================================
# BONUS: MULTI-CLASS ATTACK CLASSIFICATION USING NAIVE BAYES
# ============================================================================
print("\n" + "="*100)
print("BONUS: MULTI-CLASS ATTACK CLASSIFICATION")
print("Applying Naive Bayes classifiers to the Label column for multi-class attack classification")
print("="*100)

# --- Prepare Multi-Class Data ---
# Use the same df that's already loaded and filtered (don't reload)
print(f"\nMulti-class labels: {sorted(df['Label'].unique())}")
print(f"Class distribution:")
print(df['Label'].value_counts())

# Define features (same as before)
exclude_cols_multiclass = {'Label', 'Binary Label', 'Switch ID', 'Port Number', 'is_valid',
                           'Table ID', 'Active Flow Entries', 'Packets Looked Up', 'Packets Matched', 'Max Size'}
feature_cols_multiclass = [col for col in df.columns if col not in exclude_cols_multiclass and pd.api.types.is_numeric_dtype(df[col])]

# Separate features (X) and target (y) for multi-class
X_multiclass = df[feature_cols_multiclass]
y_multiclass = df['Label']

# --- Use the same train/test split indices from before ---
# These indices were already created and exist in the current df
X_train_multiclass = X_multiclass.loc[train_indices]
y_train_multiclass = y_multiclass.loc[train_indices]
X_test_multiclass = X_multiclass.loc[test_indices]
y_test_multiclass = y_multiclass.loc[test_indices]

print(f"\nMulti-class training set size: {len(X_train_multiclass)}")
print(f"Multi-class test set size: {len(X_test_multiclass)}")
print(f"\nTraining set class distribution:")
print(y_train_multiclass.value_counts())

# --- One-Hot Encode Categorical Features for Multi-Class ---
df_train_multiclass_full = df.loc[X_train_multiclass.index].copy()
df_test_multiclass_full = df.loc[X_test_multiclass.index].copy()

if categorical_cols_to_encode:
    print("\nEncoding categorical features for multi-class classification...")
    X_train_categorical_mc = pd.get_dummies(df_train_multiclass_full[categorical_cols_to_encode], drop_first=False)
    X_test_categorical_mc = pd.get_dummies(df_test_multiclass_full[categorical_cols_to_encode], drop_first=False)
    
    # Align columns
    X_train_categorical_mc, X_test_categorical_mc = X_train_categorical_mc.align(X_test_categorical_mc, join='outer', axis=1, fill_value=0)
    
    # Concatenate
    X_train_encoded_mc = pd.concat([X_train_multiclass, X_train_categorical_mc], axis=1)
    X_test_encoded_mc = pd.concat([X_test_multiclass, X_test_categorical_mc], axis=1)
    
    print(f"Shape of X_train_encoded (multi-class): {X_train_encoded_mc.shape}")
    print(f"Shape of X_test_encoded (multi-class): {X_test_encoded_mc.shape}")
else:
    X_train_encoded_mc = X_train_multiclass
    X_test_encoded_mc = X_test_multiclass

# --- Multi-Class Evaluation Function ---
def compute_multiclass_metrics(y_true, y_pred):
    """
    Calculate accuracy and per-class precision/recall for multi-class classification.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, _, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=sorted(y_true.unique()), zero_division=0)
    
    # Calculate macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=sorted(y_true.unique()))
    
    return accuracy, precision, recall, support, macro_precision, macro_recall, cm

# --- Train and Evaluate Multi-Class Models ---
sklearn_nb_results_multiclass = {}

# 1. GaussianNB for Multi-Class
print("\n--- Training GaussianNB for Multi-Class Classification ---")
scaler_gnb_mc = StandardScaler()
X_train_gnb_mc = scaler_gnb_mc.fit_transform(X_train_encoded_mc)
X_test_gnb_mc = scaler_gnb_mc.transform(X_test_encoded_mc)

model_gnb_mc = GaussianNB()
model_gnb_mc.fit(X_train_gnb_mc, y_train_multiclass)
y_pred_gnb_mc = model_gnb_mc.predict(X_test_gnb_mc)

acc_gnb_mc, prec_gnb_mc, rec_gnb_mc, supp_gnb_mc, macro_prec_gnb, macro_rec_gnb, cm_gnb_mc = compute_multiclass_metrics(y_test_multiclass, y_pred_gnb_mc)

sklearn_nb_results_multiclass['GaussianNB'] = {
    'Accuracy': acc_gnb_mc,
    'Macro_Precision': macro_prec_gnb,
    'Macro_Recall': macro_rec_gnb,
    'Per_Class_Precision': prec_gnb_mc,
    'Per_Class_Recall': rec_gnb_mc,
    'Support': supp_gnb_mc,
    'Confusion_Matrix': cm_gnb_mc
}

print(f"GaussianNB - Accuracy: {acc_gnb_mc:.4f}, Macro Precision: {macro_prec_gnb:.4f}, Macro Recall: {macro_rec_gnb:.4f}")

# 2. MultinomialNB for Multi-Class
print("\n--- Training MultinomialNB for Multi-Class Classification ---")
try:
    scaler_mnb_mc = MinMaxScaler()
    X_train_mnb_mc = scaler_mnb_mc.fit_transform(X_train_encoded_mc)
    X_test_mnb_mc = scaler_mnb_mc.transform(X_test_encoded_mc)
    
    model_mnb_mc = MultinomialNB()
    model_mnb_mc.fit(X_train_mnb_mc, y_train_multiclass)
    y_pred_mnb_mc = model_mnb_mc.predict(X_test_mnb_mc)
    
    acc_mnb_mc, prec_mnb_mc, rec_mnb_mc, supp_mnb_mc, macro_prec_mnb, macro_rec_mnb, cm_mnb_mc = compute_multiclass_metrics(y_test_multiclass, y_pred_mnb_mc)
    
    sklearn_nb_results_multiclass['MultinomialNB'] = {
        'Accuracy': acc_mnb_mc,
        'Macro_Precision': macro_prec_mnb,
        'Macro_Recall': macro_rec_mnb,
        'Per_Class_Precision': prec_mnb_mc,
        'Per_Class_Recall': rec_mnb_mc,
        'Support': supp_mnb_mc,
        'Confusion_Matrix': cm_mnb_mc
    }
    
    print(f"MultinomialNB - Accuracy: {acc_mnb_mc:.4f}, Macro Precision: {macro_prec_mnb:.4f}, Macro Recall: {macro_rec_mnb:.4f}")
except Exception as e:
    print(f"Error training MultinomialNB for multi-class: {e}")
    sklearn_nb_results_multiclass['MultinomialNB'] = None

# 3. BernoulliNB for Multi-Class
print("\n--- Training BernoulliNB for Multi-Class Classification ---")
try:
    scaler_bnb_mc = StandardScaler()
    X_train_scaled_mc = scaler_bnb_mc.fit_transform(X_train_encoded_mc)
    X_test_scaled_mc = scaler_bnb_mc.transform(X_test_encoded_mc)
    
    X_train_bnb_mc = (X_train_scaled_mc > 0).astype(int)
    X_test_bnb_mc = (X_test_scaled_mc > 0).astype(int)
    
    model_bnb_mc = BernoulliNB()
    model_bnb_mc.fit(X_train_bnb_mc, y_train_multiclass)
    y_pred_bnb_mc = model_bnb_mc.predict(X_test_bnb_mc)
    
    acc_bnb_mc, prec_bnb_mc, rec_bnb_mc, supp_bnb_mc, macro_prec_bnb, macro_rec_bnb, cm_bnb_mc = compute_multiclass_metrics(y_test_multiclass, y_pred_bnb_mc)
    
    sklearn_nb_results_multiclass['BernoulliNB'] = {
        'Accuracy': acc_bnb_mc,
        'Macro_Precision': macro_prec_bnb,
        'Macro_Recall': macro_rec_bnb,
        'Per_Class_Precision': prec_bnb_mc,
        'Per_Class_Recall': rec_bnb_mc,
        'Support': supp_bnb_mc,
        'Confusion_Matrix': cm_bnb_mc
    }
    
    print(f"BernoulliNB - Accuracy: {acc_bnb_mc:.4f}, Macro Precision: {macro_prec_bnb:.4f}, Macro Recall: {macro_rec_bnb:.4f}")
except Exception as e:
    print(f"Error training BernoulliNB for multi-class: {e}")
    sklearn_nb_results_multiclass['BernoulliNB'] = None

# --- Display Results ---
print("\n" + "="*100)
print("MULTI-CLASS CLASSIFICATION RESULTS SUMMARY")
print("="*100)

# Summary table
print(f"\n{'Model':<20} {'Accuracy':<12} {'Macro Precision':<18} {'Macro Recall':<15}")
print("-" * 65)
for model_name, results in sklearn_nb_results_multiclass.items():
    if results is not None:
        acc = results['Accuracy']
        macro_prec = results['Macro_Precision']
        macro_rec = results['Macro_Recall']
        print(f"{model_name:<20} {acc:<12.4f} {macro_prec:<18.4f} {macro_rec:<15.4f}")

# Per-class performance for best model
best_model_mc = max([k for k in sklearn_nb_results_multiclass.keys() if sklearn_nb_results_multiclass[k] is not None], 
                    key=lambda k: sklearn_nb_results_multiclass[k]['Accuracy'])

print(f"\n--- Detailed Results for Best Model: {best_model_mc} ---")
best_results = sklearn_nb_results_multiclass[best_model_mc]
class_labels = sorted(y_test_multiclass.unique())

print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'Support':<10}")
print("-" * 49)
for i, label in enumerate(class_labels):
    prec = best_results['Per_Class_Precision'][i]
    rec = best_results['Per_Class_Recall'][i]
    supp = best_results['Support'][i]
    print(f"{label:<15} {prec:<12.4f} {rec:<12.4f} {supp:<10}")

print(f"\n--- Confusion Matrix for {best_model_mc} ---")
print(f"Classes: {class_labels}")
print(best_results['Confusion_Matrix'])

# --- Justification ---
print("\n--- Justification ---")
print("Multi-class classification allows the system to identify specific attack types rather than just detecting 'attack vs normal'.")
print("This provides more actionable information for security teams:")
print(" - Different attack types require different response strategies")
print(" - Understanding attack patterns helps improve network defenses")
print(" - Fine-grained classification enables better threat intelligence")
print(f"\n{best_model_mc} achieved the best overall accuracy ({best_results['Accuracy']:.2%}) for distinguishing between attack types.")
print("The per-class metrics show how well the model performs for each specific attack type,")
print("helping identify which attacks are easier or harder to detect.")

print("\n" + "="*100)
print("BONUS: Multi-Class Attack Classification Completed")
print("="*100)
#%%