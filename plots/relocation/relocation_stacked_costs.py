# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 16:52:28 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from matplotlib.cm import get_cmap
import math


# %%

# Custom category mapping function
def map_category(name):
    if any(keyword in name.lower() for keyword in ["urban","rural","CHP","boiler","thermal"]):
        return "Heating"
    if any(keyword in name.lower() for keyword in ["DAC"]):
        return "DAC"
    if any(keyword in name.lower() for keyword in ["limestone",]):
        return "Limestone"
    if any(keyword in name.lower() for keyword in ["iron"]):
        return "Iron"
    if any(keyword in name.lower() for keyword in ["steel process"]):
        return "BF-BOF TGR" 
    if any(keyword in name.lower() for keyword in ["steel"]):
        return "Steel" 
    if any(keyword in name.lower() for keyword in ["fischer","sabatier","process","industry"]):
        return "Industry"
    if any(keyword in name.lower() for keyword in ["uranium","reservoir", "battery", "run","wind","lignite", "solar", "generator", "coal", "oil", "nuclear", "hydro", "gas","electricity", "ac", "dc"]):
        return "Power System"
    if any(keyword in name.lower() for keyword in ["cement","limestone"]):
        return "Cement"
    if any(keyword in name.lower() for keyword in ["naphtha","hvc"]):
        return "HVC"
    if any(keyword in name.lower() for keyword in ["haber"]):
        return "Ammonia"
    if any(keyword in name.lower() for keyword in ["kerosene", "bev","transport", "shipping"]):
        return "Transport"
    if any(keyword in name.lower() for keyword in ["methanolisation","methanol"]):
        return "Methanol"
    if "h2" in name.lower() or any(keyword in name.lower() for keyword in ["electrolysis", "fuel cell", "pipeline", "smr","h2"]):
        return "Hydrogen"
    if "co2" in name.lower():
        return "CO2 Infrastructure"
    if "agriculture" in name.lower():
        return "Agriculture"
    if any(keyword in name.lower() for keyword in ["biogas","biomass","liquid"]):
        return "Biomass"
    return name

def compute_diff_by_costtype(df_full_index, cost_type, eu_scenario="policy_eu_regain", reg_scenario="policy_reg_regain"):
    # Filter only one cost type (CAPEX or OPEX)
    df_filtered = df_full_index[df_full_index["CostType"] == cost_type]

    # Group and pivot
    grouped = df_filtered.groupby(["Scenario", "Year", "Category"], as_index=False)["Value"].sum()
    pivot = grouped.pivot_table(index="Year", columns=["Scenario", "Category"], values="Value", fill_value=0)
    pivot = pivot.sort_index(axis=1, level=[0, 1])

    categories = pivot.columns.levels[1]

    # Compute % difference
    diff = pd.DataFrame(index=pivot.index)
    for cat in categories:
        eu_val = pivot.get((eu_scenario, cat), pd.Series(0, index=pivot.index))
        reg_val = pivot.get((reg_scenario, cat), pd.Series(0, index=pivot.index))
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_diff = (reg_val - eu_val) / eu_val.replace(0, np.nan) * 100
            pct_diff = pct_diff.fillna(0)
        diff[cat] = pct_diff

    # Map and group categories
    mapped_cols = [map_category(col) for col in diff.columns]
    diff.columns = mapped_cols
    diff = diff.groupby(diff.columns, axis=1).sum()
    return diff

def plot_diff_subplots(diff_df, title_prefix):
    
    # Remove columns containing "BF-BOF TGR"
    diff_df = diff_df.loc[:, ~diff_df.columns.str.contains("BF-BOF TGR")]

    # Filter columns to only those with at least one value >= 0.01 absolute
    filtered_cols = [col for col in diff_df.columns if (diff_df[col].abs() >= 0).any()]
    filtered_df = diff_df[filtered_cols]

    n_cols = 3
    n_rows = math.ceil(len(filtered_cols) / n_cols) if filtered_cols else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)
    
    # Handle case when only one subplot (axes not array)
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = filtered_df.index
    cmap = get_cmap("tab20")
    color_map = {category: cmap(i % 20) for i, category in enumerate(sorted(filtered_cols))}

    for i, category in enumerate(filtered_cols):
        ax = axes[i]
        values = filtered_df[category]
        color = color_map[category]
        ax.plot(x, values, marker='o', color=color)
        ax.axhline(0, color='black', linewidth=0.8, linestyle="--")
        ax.set_title(category)
        ax.set_ylabel("Δ Cost [%]")
        ax.grid(True)
        ax.set_xticks([2030, 2040, 2050])

    # Remove unused subplots if any
    for j in range(len(filtered_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{title_prefix} - Percentage Cost Difference (REG vs EU)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    filename = f"graphs/{title_prefix.lower().replace(' ', '_')}_diff_subplots.png"
    plt.savefig(filename, dpi=300)

    plt.show()

    
def get_side_by_side_values(df_full_index, cost_type, scenario1="policy_eu_regain", scenario2="policy_reg_regain"):
    # Filter by cost type
    df_filtered = df_full_index[df_full_index["CostType"] == cost_type]

    # Group data by Scenario, Year, and Category
    grouped = df_filtered.groupby(["Scenario", "Year", "Category"], as_index=False)["Value"].sum()

    # Pivot to get (Scenario-Year) columns per Category
    pivot_df = grouped.pivot_table(index="Category", columns=["Scenario", "Year"], values="Value", fill_value=0)

    # Optional: sort columns for readability
    pivot_df = pivot_df.sort_index(axis=1, level=[0, 1])
    
    # Optionally map categories
    pivot_df.index = [map_category(cat) for cat in pivot_df.index]

    return pivot_df


def plot_diff_subplots_combined(diff_capex_df, diff_opex_df, title_prefix):
    import math
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    # Remove columns containing "BF-BOF TGR"
    diff_capex_df = diff_capex_df.loc[:, ~diff_capex_df.columns.str.contains("BF-BOF TGR")]
    diff_opex_df = diff_opex_df.loc[:, ~diff_opex_df.columns.str.contains("BF-BOF TGR")]

    # Combine columns with a label prefix to keep track
    capex_cols = [f"Capex: {col}" for col in diff_capex_df.columns]
    opex_cols = [f"Opex: {col}" for col in diff_opex_df.columns]

    combined_df = pd.concat([
        diff_capex_df.rename(columns=dict(zip(diff_capex_df.columns, capex_cols))),
        diff_opex_df.rename(columns=dict(zip(diff_opex_df.columns, opex_cols)))
    ], axis=1)

    # Filter columns to those with at least one value >= 0.01 absolute
    filtered_cols = [col for col in combined_df.columns if (combined_df[col].abs() >= 1e-1).any()]
    filtered_df = combined_df[filtered_cols]

    n_cols = 3
    n_rows = math.ceil(len(filtered_cols) / n_cols) if filtered_cols else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)
    
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = filtered_df.index
    cmap = get_cmap("tab20")
    color_map = {category: cmap(i % 20) for i, category in enumerate(sorted(filtered_cols))}

    for i, category in enumerate(filtered_cols):
        ax = axes[i]
        values = filtered_df[category]
        color = color_map[category]
        ax.plot(x, values, marker='o', color=color)
        ax.axhline(0, color='black', linewidth=0.8, linestyle="--")
        ax.set_title(category)
        ax.set_ylabel("Δ Cost [%]")
        ax.grid(True)
        ax.set_xticks([2030, 2040, 2050])

    for j in range(len(filtered_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{title_prefix} - Percentage Cost Difference (REG vs EU)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"graphs/{title_prefix.lower().replace(' ', '_')}_diff_subplots_combined.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
    
def compute_diff_by_costtype_noH2(df_full_index, cost_type, reg_scenario="policy_reg_regain", noH2_scenario="policy_reg_regain_noH2"):
    # Filter only one cost type (CAPEX or OPEX)
    df_filtered = df_full_index[df_full_index["CostType"] == cost_type]

    # Group and pivot
    grouped = df_filtered.groupby(["Scenario", "Year", "Category"], as_index=False)["Value"].sum()
    pivot = grouped.pivot_table(index="Year", columns=["Scenario", "Category"], values="Value", fill_value=0)
    pivot = pivot.sort_index(axis=1, level=[0, 1])

    categories = pivot.columns.levels[1]

    # Compute % difference
    diff = pd.DataFrame(index=pivot.index)
    for cat in categories:
        reg_val = pivot.get((noH2_scenario, cat), pd.Series(0, index=pivot.index))
        noH2_val = pivot.get((reg_scenario, cat), pd.Series(0, index=pivot.index))
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_diff = (noH2_val - reg_val) / reg_val.replace(0, np.nan) * 100
            pct_diff = pct_diff.fillna(0)
        diff[cat] = pct_diff

    # Map and group categories
    mapped_cols = [map_category(col) for col in diff.columns]
    diff.columns = mapped_cols
    diff = diff.groupby(diff.columns, axis=1).sum()
    return diff


def plot_diff_subplots_combined_noH2(diff_capex_df, diff_opex_df, title_prefix):
    import math
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    # Remove columns containing "BF-BOF TGR"
    diff_capex_df = diff_capex_df.loc[:, ~diff_capex_df.columns.str.contains("BF-BOF TGR")]
    diff_opex_df = diff_opex_df.loc[:, ~diff_opex_df.columns.str.contains("BF-BOF TGR")]

    # Combine columns with a label prefix to keep track
    capex_cols = [f"Capex: {col}" for col in diff_capex_df.columns]
    opex_cols = [f"Opex: {col}" for col in diff_opex_df.columns]

    combined_df = pd.concat([
        diff_capex_df.rename(columns=dict(zip(diff_capex_df.columns, capex_cols))),
        diff_opex_df.rename(columns=dict(zip(diff_opex_df.columns, opex_cols)))
    ], axis=1)

    # Filter columns to those with at least one value >= 0.01 absolute
    filtered_cols = [col for col in combined_df.columns if (combined_df[col].abs() >= 1e-1).any()]
    filtered_df = combined_df[filtered_cols]

    n_cols = 3
    n_rows = math.ceil(len(filtered_cols) / n_cols) if filtered_cols else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False)
    
    if n_rows * n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    x = filtered_df.index
    cmap = get_cmap("tab20")
    color_map = {category: cmap(i % 20) for i, category in enumerate(sorted(filtered_cols))}

    for i, category in enumerate(filtered_cols):
        ax = axes[i]
        values = filtered_df[category]
        color = color_map[category]
        ax.plot(x, values, marker='o', color=color)
        ax.axhline(0, color='black', linewidth=0.8, linestyle="--")
        ax.set_title(category)
        ax.set_ylabel("Δ Cost [%]")
        ax.grid(True)
        ax.set_xticks([2030, 2040, 2050])

    for j in range(len(filtered_cols), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{title_prefix} - Percentage Cost Difference (REG noH2 vs REG)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = f"graphs/{title_prefix.lower().replace(' ', '_')}_diff_subplots_combined_noH2.png"
    plt.savefig(filename, dpi=300)
    plt.show()
    
# %%

# Define scenarios and years
scenarios = ["policy_eu_regain", "policy_reg_regain"]
years = [2030, 2040, 2050]
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)

# Initialize DataFrames
df = pd.DataFrame(columns=["Scenario", "Year", "Category", "Value"])
df_full_index = pd.DataFrame(columns=["Scenario", "Year", "ComponentType", "Category", "CostType", "Value"])

exclude = ["Bus", "Load", "SnapshotWeightings"]

# Collect expenditure data
for scenario in scenarios:
    for year in years:
        file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        n = pypsa.Network(file_path)
        stats = n.statistics()

        capex = stats["Capital Expenditure"]
        opex = stats["Operational Expenditure"]

        # Filter excluded components
        capex = capex[~capex.index.get_level_values(0).isin(exclude)]
        opex = opex[~opex.index.get_level_values(0).isin(exclude)]

        # Save detailed rows with full index for CAPEX
        for (component_type, item), value in capex.items():
            if value > 2500:
                df_full_index.loc[len(df_full_index)] = [
                    scenario, year, component_type, item, "CAPEX", value
                ]
                df.loc[len(df)] = [scenario, year, item, value]

        # Save detailed rows with full index for OPEX
        for (component_type, item), value in opex.items():
            if value > 2500:
                df_full_index.loc[len(df_full_index)] = [
                    scenario, year, component_type, item, "OPEX", value
                ]
                df.loc[len(df)] = [scenario, year, item, value]
                
capex_diff = compute_diff_by_costtype(df_full_index, "CAPEX")
opex_diff = compute_diff_by_costtype(df_full_index, "OPEX")

capex_values = get_side_by_side_values(df_full_index, "CAPEX")
opex_values = get_side_by_side_values(df_full_index, "OPEX")


plot_diff_subplots(capex_diff, "CAPEX")
plot_diff_subplots(opex_diff, "OPEX")

plot_diff_subplots_combined(capex_diff, opex_diff, "Cost Differences by Category")


# %%


# Define scenarios and years
scenarios = ["policy_reg_regain", "policy_reg_regain_noH2"]
years = [2030, 2040, 2050]
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)

# Initialize DataFrames
df = pd.DataFrame(columns=["Scenario", "Year", "Category", "Value"])
df_full_index = pd.DataFrame(columns=["Scenario", "Year", "ComponentType", "Category", "CostType", "Value"])

exclude = ["Bus", "Load", "SnapshotWeightings"]

# Collect expenditure data
for scenario in scenarios:
    for year in years:
        file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        n = pypsa.Network(file_path)
        stats = n.statistics()

        capex = stats["Capital Expenditure"]
        opex = stats["Operational Expenditure"]

        # Filter excluded components
        capex = capex[~capex.index.get_level_values(0).isin(exclude)]
        opex = opex[~opex.index.get_level_values(0).isin(exclude)]

        # Save detailed rows with full index for CAPEX
        for (component_type, item), value in capex.items():
            if value > 2500:
                df_full_index.loc[len(df_full_index)] = [
                    scenario, year, component_type, item, "CAPEX", value
                ]
                df.loc[len(df)] = [scenario, year, item, value]

        # Save detailed rows with full index for OPEX
        for (component_type, item), value in opex.items():
            if value > 2500:
                df_full_index.loc[len(df_full_index)] = [
                    scenario, year, component_type, item, "OPEX", value
                ]
                df.loc[len(df)] = [scenario, year, item, value]
                
capex_diff = compute_diff_by_costtype_noH2(df_full_index, "CAPEX")
opex_diff = compute_diff_by_costtype_noH2(df_full_index, "OPEX")

capex_values = get_side_by_side_values(df_full_index, "CAPEX")
opex_values = get_side_by_side_values(df_full_index, "OPEX")


plot_diff_subplots(capex_diff, "CAPEX")
plot_diff_subplots(opex_diff, "OPEX")

plot_diff_subplots_combined_noH2(capex_diff, opex_diff, "Cost Differences by Category")
