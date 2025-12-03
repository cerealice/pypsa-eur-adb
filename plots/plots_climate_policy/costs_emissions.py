# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 12:29:36 2025

@author: Dibella
"""


import pypsa
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
scenarios = ["base_eu_regain", "policy_eu_regain"]  # <=== DEFINE SCENARIOS HERE
years = [2030, 2040, 2050]

base_colors = {"regain": "#4F5050", "maintain": "#85877C", "deindustrial": "#B0B2A1"}
policy_colors = {"regain": "#5D8850", "maintain": "#95BF74", "deindustrial": "#C5DEB1"}

scenario_colors = {
    "base_eu_regain": "grey",
    "policy_eu_regain": "green",
    "policy_eu_deindustrial": "orange",
    "policy_reg_regain": "purple"
}
scenario_labels = {
    "base_eu_regain": "Baseline",
    "policy_eu_regain": "Climate policy"
}

data_dict = {
    "Annual system cost [b€/yr]": {"base_eu_regain": {}, "policy_eu_regain": {}},
    "CO2 emissions [MtCO2/yr]": {"base_eu_regain": {}, "policy_eu_regain": {}},
}



# %%
# COSTS AND EMISSIONS

# Loop through each year and scenario to fill the data_dict
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(cwd)
        file_path = os.path.join(os.path.dirname(parent_dir), "results_8h_juno", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0, 0]
        
        # Store the data for each metric
        data_dict["Annual system cost [b€/yr]"][scenario][year] = n.objective / 1e9
        data_dict["CO2 emissions [MtCO2/yr]"][scenario][year] = n.stores.loc['co2 atmosphere','e_nom_opt'] / 1e6


# Create the figure with two subplots: one for cost, one for emissions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharex=True)


# Plot Annual System Cost on ax1
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
ax1.set_ylabel(label_cost)
ax1.grid(True, linestyle = '-')
ax1.legend()
ax1.set_title("Annual System Cost")
ax1.set_xticks(years)

# Plot CO2 Emissions on ax2
label_emissions = "CO2 emissions [MtCO2/yr]"
for scenario in scenarios:
    ax2.plot(
        years,
        [data_dict[label_emissions][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )
ax2.set_ylabel(label_emissions)
ax2.grid(True, linestyle = '-')
ax2.legend()
ax2.set_title("CO2 Emissions")
ax2.set_xticks(years)

# Improve layout and save
plt.tight_layout()
plt.savefig("./graphs/costs_emissions.png")
plt.show()