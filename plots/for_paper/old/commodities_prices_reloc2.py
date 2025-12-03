# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:27:22 2025

@author: Dibella
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
scenarios = [
    "policy_reg_deindustrial",
    "policy_eu_deindustrial",
    "policy_reg_regain",
    "policy_eu_regain",
]
years = [2020, 2030, 2040, 2050]

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

lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t
naphtha_to_hvc = (2.31 * 12.47) * 1000
decay_emis_hvc = 0.2571 * naphtha_to_hvc / 1e3
lhv_hydrogen = 33.33 #MWh/t

# === HISTORICAL VALUES ===
hist_2020_prices = {
    "steel": 415,
    "cement": 93,
    "ammonia": 470,
    "methanol": 326,
    "HVC": 600,
    "H2": 1800
}

ft_to_hvc = 2.31 * 12.47
# Data from figure 1 Neumann et al. https://arxiv.org/pdf/2404.03927
max_import_costs = {
    "steel": 501,
    "ammonia": 145.8 * lhv_ammonia ,
    "methanol": 168.5 * lhv_methanol,
    "HVC": 182.4 * ft_to_hvc,
    "H2": 132.1 * lhv_hydrogen
}

min_import_costs = {
    "steel": 417, # https://www.sciencedirect.com/science/article/pii/S0360319923022930
    "ammonia": 87.7 * lhv_ammonia ,
    "methanol": 106.8 * lhv_methanol,
    "HVC": 109.8 * ft_to_hvc,
    "H2": 57.5 * lhv_hydrogen
}


def weighted_average_marginal_price(n, keyword, exclude_labels=None):
    # Time step is both nominator and denominator
    mprice_cols = n.buses_t.marginal_price.columns[
        n.buses_t.marginal_price.columns.str.contains(keyword)
    ]
    if exclude_labels:
        for label in exclude_labels:
            mprice_cols = mprice_cols[~mprice_cols.str.contains(label)]
    mprice = n.buses_t.marginal_price.loc[:, mprice_cols].where(lambda df: df >= 0, 0)
    relevant_loads = mprice.columns.intersection(n.loads.index)
    mprice_loads = mprice[relevant_loads]
    loads_w_mprice = n.loads_t.p[relevant_loads]
    
    if keyword == "H2":
        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads_w_mprice = loads.T.groupby(level=0).sum().T
        mprice.columns = mprice.columns.str[:2]
        mprice_loads = mprice.T.groupby(level=0).sum().T     
        
    total_costs = mprice_loads * loads_w_mprice
    
    total = total_costs.sum().sum() / loads_w_mprice.sum().sum()

    return total

# === INIT STORAGE ===
price_data = {commodity: pd.DataFrame(index=scenarios, columns=years) for commodity in hist_2020_prices.keys()}
for commodity, val in hist_2020_prices.items():
    price_data[commodity][2020] = val

# === LOAD AND COMPUTE PRICES ===
max_value = 0
cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))

for scenario in scenarios:
    for year in years[1:]:
        file_path = os.path.join(parent_dir, "results_july", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0, 0]

        price_data["steel"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="steel") / 1e3
        price_data["cement"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="cement", exclude_labels=["process emissions"]) / 1e3
        price_data["ammonia"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="NH3") * lhv_ammonia

        price_data["methanol"].loc[scenario, year] = weighted_average_marginal_price(n, keyword='industry methanol') * lhv_methanol
        meth_price = n.buses_t.marginal_price.loc[
            :, n.buses_t.marginal_price.columns.str.contains('industry methanol')
              # ~n.buses_t.marginal_price.columns.str.contains('industry') &
              # ~n.buses_t.marginal_price.columns.str.contains('shipping')
        ].mean().iloc[0] * lhv_methanol
        co2_price = -n.global_constraints.loc["CO2Limit", "mu"]
        extra_methanol_cost = 0.248 * lhv_methanol * co2_price
        #price_data["methanol"].loc[scenario, year] = meth_price# + extra_methanol_cost # The models sees this in different demands

        hvc_price = weighted_average_marginal_price(n, keyword="HVC") / 1e3
        extra_hvc_cost = 0.2571 * 12.47 * co2_price
        price_data["HVC"].loc[scenario, year] = hvc_price #- extra_hvc_cost
        price_data["H2"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="H2", exclude_labels=["pipeline"]) * lhv_hydrogen

        max_value = max(max_value, price_data["HVC"].loc[scenario, year])

# %%

# === PLOT ===
commodities = ["steel", "cement", "ammonia", "methanol", "HVC", "H2"]
fig, axes = plt.subplots(1, len(commodities), figsize=(12, 6), sharex=True, sharey=False)

for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for scenario in scenarios:
        label = scenario_labels[scenario] if idx == 0 else None
        ax.plot(
            years,
            price_data[commodity].loc[scenario],
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors.get(scenario, 'black')
        )

    # Skip cement (no import band for it)
    if commodity != "cement":
        # Draw import cost band
        ax.fill_between(
            years,
            min_import_costs[commodity],
            max_import_costs[commodity],
            color="grey",
            alpha=0.3,
            label="Import price range" if idx == 0 else None
        )

    # Titles and layout
    custom_titles = {
        "hvc": "Plastics",
        "h2": "Hydrogen"
    }

    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(f"{title.capitalize()}")
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)
    if idx == 0:
        ax.set_ylabel("Price [€/t]")
        ax.legend(fontsize=8, loc="lower left")  # Legend only on first subplot

    ax.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("./graphs/commodity_prices_with_import_band_labels_reloc.png", dpi=300)
plt.show()


# %%
# === CALCULATE AND PLOT PERCENTAGE CHANGES FOR RELOCATION CASES ===

relocation_pairs = [
    ("policy_reg_deindustrial", "policy_eu_deindustrial"),
    ("policy_reg_regain", "policy_eu_regain"),
]

fig, axes = plt.subplots(1, len(commodities), figsize=(14, 4), sharex=True, sharey=True)

for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for reg_case, eu_case in relocation_pairs:
        # Skip year 2020
        pct_change = (
            (price_data[commodity].loc[eu_case, years[1:]] - price_data[commodity].loc[reg_case, years[1:]])
            / price_data[commodity].loc[reg_case, years[1:]]
        ) * 100

        label = f"{scenario_labels[eu_case].splitlines()[-1]}" if idx == 0 else None

        ax.plot(
            years[1:],
            pct_change,
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors[eu_case]
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(custom_titles.get(commodity.lower(), commodity.title()))
    ax.set_xticks(years[1:])
    ax.set_ylabel("% Change" if idx == 0 else "")
    ax.grid(True, linestyle='--')
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("./graphs/commodity_prices_pct_change_reloc_vs_noreloc.png", dpi=300)
plt.show()
