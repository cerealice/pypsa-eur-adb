# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 10:26:37 2025

@author: Dibella
"""


import pypsa
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
scenarios = [
    "base_reg_deindustrial",
    "policy_reg_deindustrial",
    "base_reg_regain",
    "policy_reg_regain"
]
years = [2030, 2040, 2050]

scenario_colors = {
    "base_reg_deindustrial": "#464E47",
    "policy_reg_deindustrial": "#00B050",
    "base_reg_regain": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
    
}

scenario_labels = {
    "base_reg_deindustrial": "No climate policy\nCurrent deindustr trend",
    "policy_reg_deindustrial": "Climate policy\nCurrent deindustr trend",
    "base_reg_regain": "No climate policy\nReindustrialize",
    "policy_reg_regain": "Climate policy\nReindustrialize"
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
        file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
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
plt.savefig("./graphs/costs_emissions.png", dpi=300)
plt.show()
