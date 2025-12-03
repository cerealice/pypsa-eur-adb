# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 10:37:59 2025

@author: Dibella
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:26:37 2025

@author: Dibella
"""


import pypsa
import matplotlib.pyplot as plt
import os
import pandas as pd

# === CONFIGURATION ===
scenarios = [
    "base_reg_deindustrial",
    "policy_reg_deindustrial",
    "base_reg_regain",
    "policy_reg_regain",
]

scenario_colors = {

    "policy_reg_deindustrial": "#00B050",
    "base_reg_deindustrial": "#6AFFAD",  
    "policy_reg_regain": "#3AAED8",
    "base_reg_regain": "#99D5EB",
}

scenario_labels = {
    
    "policy_reg_deindustrial": "Climate Policy\nGradual Deindustr",
    "base_reg_deindustrial": "No Climate Policy\nGradual Deindustr",
    "policy_reg_regain": "Climate Policy\nReindustr Strategy",
    "base_reg_regain": "No Climate Policy\nReindustr Strategy",
}
years = [2030, 2040, 2050]


data_dict = {
    "Annual system cost [b€/yr]": {s: {} for s in scenarios},
    "CO2 Price [€/tCO2]": {s: {} for s in scenarios},
}

# === LOAD DATA ===
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, "results_noloads", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0,0]
        
        data_dict["Annual system cost [b€/yr]"][scenario][year] = n.objective / 1e9
        #data_dict["CO2 emissions [MtCO2/yr]"][scenario][year] = n.stores.loc['co2 atmosphere','e_nom_opt'] / 1e6
        data_dict["CO2 Price [€/tCO2]"][scenario][year] = -n.global_constraints.loc['CO2Limit','mu']

# %%
# === PLOT ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

# Plot: Annual System Cost
label_cost = "Annual system cost [b€/yr]"
for scenario in scenarios:
    ax1.plot(
        years,
        [data_dict[label_cost][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax1.set_ylabel("b€/yr")
ax1.set_title("Annual System Cost", fontsize = 14)
ax1.set_xticks(years)
ax1.set_ylim(bottom=0)  # Set y-min to 0
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=12)

# Plot: CO2 Emissions
label_emissions = "CO2 Price [€/tCO2]"

for scenario in scenarios:
    ax2.plot(
        years,
        [data_dict[label_emissions][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax2.set_ylabel("€/tCO2")
ax2.set_title("Carbon price", fontsize = 14)
ax2.set_xticks(years)
ax2.set_ylim(bottom=0)  # Ensure y-min is 0
ax2.grid(True, linestyle='--')
#ax2.legend(fontsize=12)

# Final layout
plt.tight_layout()
plt.savefig("./graphs/costs_emissions_noloads.png", dpi=300)
plt.show()

# %%

# === PLOT PERCENTAGE VARIATION (Extra Flex vs No Flex) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

flex_pairs = [
    ("policy_reg_deindustrial", "policy_reg_deindustrial_flex"),
    ("policy_reg_regain", "policy_reg_regain_flex"),
]

# Plot: % Variation in Annual System Cost
for base, flex in flex_pairs:
    base_vals = [data_dict[label_cost][base][year] for year in years]
    flex_vals = [data_dict[label_cost][flex][year] for year in years]
    
    variation = [(f - b) / b * 100 for f, b in zip(flex_vals, base_vals)]
    
    label = scenario_labels[flex] + " vs\n" + scenario_labels[base]
    ax1.plot(
        years,
        variation,
        marker="o",
        linestyle="-",
        label=label,
        color=scenario_colors[flex]
    )

ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.set_title("Annual System Cost\n% Variation Extra Flex vs No Flex", fontsize=13)
ax1.set_ylabel("Change [%]")
ax1.set_xticks(years)
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=10)

# Plot: % Variation in CO2 Price
for base, flex in flex_pairs:
    base_vals = [data_dict[label_emissions][base][year] for year in years]
    flex_vals = [data_dict[label_emissions][flex][year] for year in years]
    
    variation = [(f - b) / b * 100 for f, b in zip(flex_vals, base_vals)]
    
    label = scenario_labels[flex] + " vs\n" + scenario_labels[base]
    ax2.plot(
        years,
        variation,
        marker="o",
        linestyle="-",
        label=label,
        color=scenario_colors[flex]
    )

ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title("Carbon Price\n% Variation Extra Flex vs No Flex", fontsize=13)
ax2.set_ylabel("Change [%]")
ax2.set_xticks(years)
ax2.grid(True, linestyle='--')
# ax2.legend(fontsize=10)  # Legend only on ax1

plt.tight_layout()
plt.savefig("./graphs/costs_emissions_flex_percentage_diff_noloads.png", dpi=300)
plt.show()

# %%

# === CONFIGURATION ===
scenarios = [
    "policy_reg_deindustrial",
    "policy_eu_deindustrial",
    "policy_reg_regain",
    "policy_eu_regain",
]

scenario_colors = {

    "policy_reg_deindustrial": "#00B050",
    "policy_eu_deindustrial": "#6AFFAD",  
    "policy_reg_regain": "#3AAED8",
    "policy_eu_regain": "#99D5EB",
}

scenario_labels = {
    
    "policy_reg_deindustrial": "No relocation\nCurrent deindustr trend",
    "policy_eu_deindustrial": "Relocation\nCurrent deindustr trend",
    "policy_reg_regain": "No relocation\nReindustrialize",
    "policy_eu_regain": "Relocation\nReindustrialize",
}

years = [2030, 2040, 2050]


data_dict = {
    "Annual system cost [b€/yr]": {s: {} for s in scenarios},
    "CO2 Price [€/tCO2]": {s: {} for s in scenarios},
}

# === LOAD DATA ===
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, "results_noloads", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0,0]
        
        data_dict["Annual system cost [b€/yr]"][scenario][year] = n.objective / 1e9
        #data_dict["CO2 emissions [MtCO2/yr]"][scenario][year] = n.stores.loc['co2 atmosphere','e_nom_opt'] / 1e6
        data_dict["CO2 Price [€/tCO2]"][scenario][year] = -n.global_constraints.loc['CO2Limit','mu']

# %%
# === PLOT ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True)

# Plot: Annual System Cost
label_cost = "Annual system cost [b€/yr]"
for scenario in scenarios:
    ax1.plot(
        years,
        [data_dict[label_cost][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax1.set_ylabel("b€/yr")
ax1.set_title("Annual System Cost", fontsize = 14)
ax1.set_xticks(years)
ax1.set_ylim(bottom=0)  # Set y-min to 0
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=12)

# Plot: CO2 Emissions
label_emissions = "CO2 Price [€/tCO2]"

for scenario in scenarios:
    ax2.plot(
        years,
        [data_dict[label_emissions][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax2.set_ylabel("€/tCO2")
ax2.set_title("Carbon price", fontsize = 14)
ax2.set_xticks(years)
ax2.set_ylim(bottom=0)  # Ensure y-min is 0
ax2.grid(True, linestyle='--')
#ax2.legend(fontsize=12)

# Final layout
plt.tight_layout()
plt.savefig("./graphs/costs_emissions_reloc_noloads.png", dpi=300)
plt.show()


# === PLOT PERCENTAGE VARIATION (Relocation  vs No) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

reloc_pairs = [
    ("policy_reg_deindustrial", "policy_eu_deindustrial"),
    ("policy_reg_regain", "policy_eu_regain"),
]

# Plot: % Variation in Annual System Cost
for base, reloc in reloc_pairs:
    base_vals = [data_dict[label_cost][base][year] for year in years]
    reloc_vals = [data_dict[label_cost][reloc][year] for year in years]
    
    variation = [(f - b) / b * 100 for f, b in zip(reloc_vals, base_vals)]
    
    label = scenario_labels[reloc] + " vs\n" + scenario_labels[base]
    ax1.plot(
        years,
        variation,
        marker="o",
        linestyle="-",
        label=label,
        color=scenario_colors[reloc]
    )

ax1.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax1.set_title("Annual System Cost\n% Variation Relocation vs No", fontsize=13)
ax1.set_ylabel("Change [%]")
ax1.set_xticks(years)
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=10)

# Plot: % Variation in CO2 Price
for base, reloc in reloc_pairs:
    base_vals = [data_dict[label_emissions][base][year] for year in years]
    reloc_vals = [data_dict[label_emissions][reloc][year] for year in years]
    
    variation = [(f - b) / b * 100 for f, b in zip(reloc_vals, base_vals)]
    
    label = scenario_labels[reloc] + " vs\n" + scenario_labels[base]
    ax2.plot(
        years,
        variation,
        marker="o",
        linestyle="-",
        label=label,
        color=scenario_colors[reloc]
    )

ax2.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax2.set_title("Carbon Price\n% Variation Relocation vs No", fontsize=13)
ax2.set_ylabel("Change [%]")
ax2.set_xticks(years)
ax2.grid(True, linestyle='--')
# ax2.legend(fontsize=10)  # Legend only on ax1

plt.tight_layout()
plt.savefig("./graphs/costs_emissions_reloc_percentage_diff_noloads.png", dpi=300)
plt.show()
