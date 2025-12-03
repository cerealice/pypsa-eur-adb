# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 14:34:32 2025

@author: alice
"""

# %% IMPORTS AND CONFIG

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
scenarios = ["base_eu_regain","policy_eu_regain"]  # <=== DEFINE SCENARIOS HERE
years = [2020, 2030, 2040, 2050]

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
    "policy_eu_regain": "Climate policy",    
}

lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t
naphtha_to_hvc = (2.31 * 12.47) * 1000
decay_emis_hvc = 0.2571 * naphtha_to_hvc / 1e3

country_names = {
    "AL": "Albania", "AT": "Austria", "BA": "Bosnia & Herzegovina", "BE": "Belgium", "BG": "Bulgaria",
    "CH": "Switzerland", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "UK", "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "ME": "Montenegro",
    "MK": "North Macedonia", "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "XK": "Kosovo"
}

# %%

# === HISTORICAL VALUES ===
hist_2020_prices = {
    "steel": 415,
    "cement": 93,
    "ammonia": 470,
    "methanol": 326,
    "HVC": 600
}

def weighted_average_marginal_price(n, keyword, exclude_labels=None):
    """
    Compute the weighted average marginal price for loads matching a given keyword,
    optionally excluding columns containing certain labels.
    
    Parameters:
    -----------
    n : pypsa.Network
        The PyPSA network object.
    keyword : str
        String to match in bus/load names (e.g., 'steel', 'cement').
    exclude_labels : list of str or None
        Substrings to exclude from the marginal price columns (e.g., ['process emissions']).

    Returns:
    --------
    float
        Mean weighted average marginal price (excluding NaNs).
    """
    # Filter marginal price columns by keyword
    mprice_cols = n.buses_t.marginal_price.columns[
        n.buses_t.marginal_price.columns.str.contains(keyword)
    ]

    # Exclude unwanted labels if provided
    if exclude_labels:
        for label in exclude_labels:
            mprice_cols = mprice_cols[~mprice_cols.str.contains(label)]

    # Get the filtered marginal prices
    mprice = n.buses_t.marginal_price.loc[:, mprice_cols].where(lambda df: df >= 0, 0)

    # Get the corresponding loads
    relevant_loads = mprice.columns.intersection(n.loads.index)
    mprice_loads = mprice[relevant_loads]
    loads_w_mprice = n.loads_t.p[relevant_loads]

    # Compute total cost and total load per country
    total_costs = mprice_loads * loads_w_mprice
    total_costs.columns = total_costs.columns.str[:2]
    total_costs = total_costs.T.groupby(level=0).sum().T.sum()

    loads_w_mprice.columns = loads_w_mprice.columns.str[:2]
    loads_w_mprice = loads_w_mprice.T.groupby(level=0).sum().T.sum()

    # Weighted average price per country
    weighted_avg = total_costs / loads_w_mprice

    # Drop NaNs and return the mean
    return weighted_avg.dropna().mean()



# %%

# === INIT STORAGE ===
price_data = {commodity: pd.DataFrame(index=scenarios, columns=years) for commodity in hist_2020_prices.keys()}
for commodity, val in hist_2020_prices.items():
    price_data[commodity][2020] = val

# %%

# === LOAD AND COMPUTE PRICES ===
max_value = 0
cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))

for scenario in scenarios:
    for year in years[1:]:  # skip 2020 (already filled)
        file_path = os.path.join(parent_dir, "results_8h_juno", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0, 0]

        price_data["steel"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="steel")/1e3 #€/kt steel -> €/tsteel
   
        
        price_data["cement"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="cement", exclude_labels=["process emissions"]) / 1e3
        price_data["ammonia"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="NH3") * lhv_ammonia

        meth_price = n.buses_t.marginal_price.loc[
            :, n.buses_t.marginal_price.columns.str.contains('methanol') &
               ~n.buses_t.marginal_price.columns.str.contains('industry') &
               ~n.buses_t.marginal_price.columns.str.contains('shipping')
        ].mean().iloc[0] * lhv_methanol
        co2_price = -n.global_constraints.loc["CO2Limit", "mu"]
        extra_methanol_cost = 0.248 * lhv_methanol * co2_price
        price_data["methanol"].loc[scenario, year] = meth_price + extra_methanol_cost

        hvc_price = weighted_average_marginal_price(n, keyword="NH3") / 1e3
        extra_hvc_cost = 0.2571 * 12.47 * co2_price
        price_data["HVC"].loc[scenario, year] = hvc_price + extra_hvc_cost

        max_value = max(max_value, price_data["HVC"].loc[scenario, year])

# %%

# === PLOT ===
commodities = ["steel", "cement", "ammonia", "methanol","HVC"]
fig, axes = plt.subplots(1,len(commodities), figsize=(12, 4), sharex=True, sharey=True)


for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for scenario in scenarios:
        label = scenario_labels[scenario] if idx == 0 else None  # Only label in the last plot
        ax.plot(
            years,
            price_data[commodity].loc[scenario],
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors.get(scenario, 'black')
        )
        
    ax.set_title(f"{commodity.capitalize()} Price")
    ax.set_xticks(years)
    ax.set_ylim(0, max_value)
    if idx == 0:
        ax.set_ylabel("Price (€/t)")
    ax.grid(True, linestyle='--')

axes[0].legend(loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig("./graphs/commodity_prices.png")
plt.show()
