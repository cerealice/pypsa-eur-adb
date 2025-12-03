# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 16:33:51 2025

@author: Dibella
"""

import os
import pypsa
import matplotlib.pyplot as plt
import numpy as np

years = [2030, 2040, 2050]

# Define your scenario names and mappings
scenarios = [
    "policy_reg_deindustrial",
    "policy_reg_deindustrial_flex",
    "policy_reg_regain",
    "policy_reg_regain_flex"
]

scenario_labels = {
    "policy_reg_deindustrial": "Deindustrialization",
    "policy_reg_deindustrial_flex": "Deindustrialization + Flex",
    "policy_reg_regain": "Reindustrialization",
    "policy_reg_regain_flex": "Reindustrialization + Flex",
}

scenario_colors = {
    "policy_reg_deindustrial": "#00B050",
    "policy_reg_deindustrial_flex": "#6AFFAD",  
    "policy_reg_regain": "#3AAED8",
    "policy_reg_regain_flex": "#99D5EB",
}

data_dict = {
    "Annual system cost [bnEUR/a]": {s: {} for s in scenarios},
    "CO2 Price [EUR/tCO2]": {s: {} for s in scenarios},
}

# === LOAD DATA ===
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0,0]

        data_dict["Annual system cost [bnEUR/a]"][scenario][year] = n.objective / 1e9
        data_dict["CO2 Price [EUR/tCO2]"][scenario][year] = -n.global_constraints.loc['CO2Limit','mu']
        
        
# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

metrics = ["Annual system cost [bnEUR/a]", "CO2 Price [EUR/tCO2]"]
titles = ["Annual System Cost", "Carbon Price"]
ylabels = ["bnEUR/a", "EUR/tCO2"]

bar_width = 0.18
bar_spacing = 0.1
group_spacing = 0.25
x_indices = np.arange(len(years))

flex_pairs = [
    ("policy_reg_deindustrial", "policy_reg_deindustrial_flex"),
    ("policy_reg_regain", "policy_reg_regain_flex"),
]

group_width = 2 * bar_width + bar_spacing      # Width of one pair (base + flex)
group_gap = 0.2                            # Extra gap between the two scenario pairs

for ax, metric, title, ylabel in zip(axes, metrics, titles, ylabels):
    for year_idx, year in enumerate(years):
        tick = x_indices[year_idx]

        # Left pair: Deindustrial
        base = "policy_reg_deindustrial"
        flex = "policy_reg_deindustrial_flex"
        base_val = data_dict[metric][base][year]
        flex_val = data_dict[metric][flex][year]
        pct_change = (flex_val - base_val) / base_val * 100

        base_x = tick - (group_width + group_gap) / 2
        flex_x = base_x + bar_width

        ax.bar(base_x, base_val, width=bar_width,
               label=scenario_labels[base] if year_idx == 0 else "", color=scenario_colors[base])
        bar_flex = ax.bar(flex_x, flex_val, width=bar_width,
                          label=scenario_labels[flex] if year_idx == 0 else "", color=scenario_colors[flex])

        ax.text(flex_x + bar_width/4, flex_val + 0.02 * flex_val,
                f"{pct_change:+.1f}%", ha='center', va='bottom', fontsize=9)

        # Right pair: Reindustrial
        base = "policy_reg_regain"
        flex = "policy_reg_regain_flex"
        base_val = data_dict[metric][base][year]
        flex_val = data_dict[metric][flex][year]
        pct_change = (flex_val - base_val) / base_val * 100

        base_x = tick + group_gap / 2
        flex_x = base_x + bar_width

        ax.bar(base_x, base_val, width=bar_width,
               label=scenario_labels[base] if year_idx == 0 else "", color=scenario_colors[base])
        bar_flex = ax.bar(flex_x, flex_val, width=bar_width,
                          label=scenario_labels[flex] if year_idx == 0 else "", color=scenario_colors[flex])

        ax.text(flex_x + bar_width/4, flex_val + 0.02 * flex_val,
                f"{pct_change:+.1f}%", ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x_indices)
    ax.set_xticklabels(years)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', axis='y')
    ax.set_ylim(bottom=0)


# Add legend once
axes[0].legend(loc='upper left', fontsize=9)

# Add explanation text below
fig.text(0.5, 0.05, "% values indicate variation of the flex scenario relative to the non-flex scenario.",
         ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space for footer
plt.savefig("graphs/costs_co2price_bar_perc.png", dpi=300)
plt.show()
