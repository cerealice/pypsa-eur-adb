# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 14:36:27 2025

@author: Dibella
"""

import matplotlib.pyplot as plt
import pypsa
import pandas as pd

# --- Country and Scenario Setup ---

group_countries = {
    'North-Western Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'Southern Europe': ['ES', 'IT', 'PT', 'GR'],
    'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}

custom_colors = {
    'Southern Europe': '#D8973C',
    'North-Western Europe': '#1B264F',
    'Eastern Europe': '#9B7EDE',
}

scenario_colors = {
    "base_eu_regain": "#464E47",
    "policy_eu_regain": "#00B050",
    "policy_eu_deindustrial": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}
scenario_labels = {
    "base_eu_regain": "NO CLIMATE POLICY\nCompetive industry\nRelocation",
    "policy_eu_regain": "CLIMATE POLICY\nCompetive industry\nRelocation",
    "policy_eu_deindustrial": "Climate policy\nDEINDUSTRIALIZATION\nRelocation",
    "policy_reg_regain": "Climate policy\nCompetive industry\nHISTORICAL HUBS"
}

scenarios = ["policy_eu_regain", "base_eu_regain", "policy_eu_deindustrial", "policy_reg_regain", ]



# --- Function to Compute Weighted Average Electricity Prices ---

def load_networks(scenarios, years, root_dir, res_dir):
    """
    Loads all networks into a dict keyed by (scenario, year).
    """
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks


# 1. Load all networks once

def compute_weighted_avg_prices(networks):
    result = {}

    for (scenario, year), n in networks.items():
        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice_loads = mprice[mprice.columns.intersection(n.loads.index)]
        loads_w_mprice = n.loads_t.p[n.loads_t.p.columns.intersection(mprice_loads.columns)]

        elec_mask = mprice_loads.columns.str.endswith(" 0") 
        mprice_elec = mprice_loads.loc[:, elec_mask]
        loads_elec = loads_w_mprice.loc[:, elec_mask]

        mprice_elec.columns = mprice_elec.columns.str[:2]
        loads_elec.columns = loads_elec.columns.str[:2]

        total_costs = mprice_elec * loads_elec
        grouped_costs = total_costs.T.groupby(level=0).sum().T
        grouped_loads = loads_elec.T.groupby(level=0).sum().T

        weighted_avg = grouped_costs / grouped_loads

        if scenario not in result:
            result[scenario] = {}
        result[scenario][year] = weighted_avg

    return result


def compute_weighted_min_prices(networks):
    result = {}

    for (scenario, year), n in networks.items():
        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice_loads = mprice[mprice.columns.intersection(n.loads.index)]
        loads_w_mprice = n.loads_t.p[n.loads_t.p.columns.intersection(mprice_loads.columns)]

        elec_mask = mprice_loads.columns.str.endswith(" 0")
        mprice_elec = mprice_loads.loc[:, elec_mask]
        loads_elec = loads_w_mprice.loc[:, elec_mask]

        mprice_elec.columns = mprice_elec.columns.str[:2]
        loads_elec.columns = loads_elec.columns.str[:2]

        total_costs = mprice_elec * loads_elec
        grouped_costs = total_costs.T.groupby(level=0).sum().T
        grouped_loads = loads_elec.T.groupby(level=0).sum().T

        weighted_avg = grouped_costs / grouped_loads

        # Calculate min price by country over buses (axis=1 is time)
        min_prices = weighted_avg.min(axis='rows' )

        if scenario not in result:
            result[scenario] = {}
        result[scenario][year] = min_prices.to_frame().T  # keep DataFrame with year as index

    return result

def compute_weighted_max_prices(networks):
    result = {}

    for (scenario, year), n in networks.items():
        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice_loads = mprice[mprice.columns.intersection(n.loads.index)]
        loads_w_mprice = n.loads_t.p[n.loads_t.p.columns.intersection(mprice_loads.columns)]

        elec_mask = mprice_loads.columns.str.endswith(" 0")
        mprice_elec = mprice_loads.loc[:, elec_mask]
        loads_elec = loads_w_mprice.loc[:, elec_mask]

        mprice_elec.columns = mprice_elec.columns.str[:2]
        loads_elec.columns = loads_elec.columns.str[:2]

        total_costs = mprice_elec * loads_elec
        grouped_costs = total_costs.T.groupby(level=0).sum().T
        grouped_loads = loads_elec.T.groupby(level=0).sum().T

        weighted_avg = grouped_costs / grouped_loads

        # Calculate max price by country over buses (axis=1 is time)
        max_prices =  weighted_avg.max(axis='rows' )

        if scenario not in result:
            result[scenario] = {}
        result[scenario][year] = max_prices.to_frame().T

    return result


# --- Main Script (Assumes `networks` is a dict of PyPSA networks per scenario) ---

# STEP 1: Compute data

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
networks = load_networks(scenarios, years, root_dir, res_dir)

weighted_avg_prices = compute_weighted_avg_prices(networks)
weighted_min_prices = compute_weighted_min_prices(networks)
weighted_max_prices = compute_weighted_max_prices(networks)

#%%
# STEP 2: Plot
fig, axes = plt.subplots(1, len(scenario_colors), figsize=(18, 5), sharey=True)

for idx, (scenario, ax) in enumerate(zip(scenario_colors.keys(), axes)):
    avg_data = pd.concat(weighted_avg_prices[scenario], names=["Year"]).groupby("Year").mean()
    min_data = pd.concat(weighted_min_prices[scenario], names=["Year"]).groupby("Year").mean()
    max_data = pd.concat(weighted_max_prices[scenario], names=["Year"]).groupby("Year").mean()

    for region, countries in group_countries.items():
        valid_countries = [c for c in countries if c in avg_data.columns]
        if not valid_countries:
            continue
    
        avg = avg_data[valid_countries].mean(axis=1)
        max_ = max_data[valid_countries].max(axis=1)
    
        ax.plot(avg.index, avg.values, label=region, color=custom_colors[region], linewidth=2)
        ax.fill_between(avg.index, avg.values, max_.values, color=custom_colors[region], alpha=0.3)



    ax.set_title(scenario_labels[scenario], fontsize=10)
    ax.set_ylim(0, 3500)  # Replace 200 with your desired max y-limit
    ax.set_xticks(years)
    if idx == 0:
        ax.set_ylabel("Electricity Price [€/MWh]")

    ax.grid(True, linestyle="--")
    ax.set_ylim(bottom=0)

# Legend outside
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True)


plt.tight_layout(rect=[0, 0, 1, 0.93])  # Make space for legend
plt.savefig("./graphs/electricity_price_bands_by_scenario.png", dpi=300)
plt.show()

# %%

import pandas as pd


# Define scenario colors and years
scenario_colors = {
    "base_eu_regain": "#464E47",
    "policy_eu_regain": "#00B050",
    "policy_eu_deindustrial": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}
years = [2030, 2040, 2050]
timestep = 3  # Set this appropriately


df = pd.DataFrame(0,columns = ['E MAX','GEN TOT'], index = networks.keys())

# Iterate through scenario-year combinations
for scenario, color in scenario_colors.items():
    for year in years:
        network_key = f"{scenario}_{year}"
        print(f"Processing: {network_key}")

        n = networks.get((scenario, year))
        if n is None:
            print(f"  ⚠️ Network {network_key} not found.")
            continue

        # Initialize values
        generation_sum = 0
        e_sum_max = 0

        gen_df = n.generators_t.p
        if 'EU solid biomass' in gen_df.columns:
            generation_sum = gen_df['EU solid biomass'].sum() * timestep


        gen_index = n.generators
        if 'EU solid biomass' in gen_index.index:
            e_sum_max = gen_index.at['EU solid biomass', 'e_sum_max']
            
        df.loc[(scenario,year),'E MAX'] = e_sum_max
        df.loc[(scenario,year),'GEN TOT'] = e_sum_max
            


