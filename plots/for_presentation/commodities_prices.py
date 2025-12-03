# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:58:40 2025

@author: Dibella
"""


# %% IMPORTS AND CONFIG

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

# === CONFIGURATION ===
scenarios = [
    "base_eu_regain",
    "policy_eu_regain",
    "policy_eu_deindustrial",
    "policy_reg_regain"
]
years = [2020, 2030, 2040, 2050]

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

def weighted_average_marginal_price(n, keyword, exclude_labels=None):
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
    total_costs.columns = total_costs.columns.str[:2]
    total_costs = total_costs.T.groupby(level=0).sum().T.sum()
    loads_w_mprice.columns = loads_w_mprice.columns.str[:2]
    loads_w_mprice = loads_w_mprice.T.groupby(level=0).sum().T.sum()
    weighted_avg = total_costs / loads_w_mprice
    return weighted_avg.dropna().mean()

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
        file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
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
commodities = ["steel", "cement", "ammonia", "methanol", "HVC","H2"]
fig, axes = plt.subplots(1, len(commodities), figsize=(14, 4), sharex=True, sharey=False)

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
        
    custom_titles = {
        "hvc": "Plastics",
        "h2": "Hydrogen"
    }

    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(f"{title.capitalize()}")
    ax.set_xticks(years)
    #ax.set_ylim(0, max_value)
    ax.set_ylim(bottom=0)
    if idx == 0:
        ax.set_ylabel("Price [€/t]")
    ax.grid(True, linestyle='--')

#axes[0].legend(loc="upper left", frameon=True, fontsize=8, ncol=1)
plt.tight_layout()
plt.savefig("./graphs/commodity_prices_all_scenarios.png", dpi=300)
plt.show()


# Create custom legend handles
legend_elements = [
    Line2D([0], [0], marker='o', color=scenario_colors.get(s), label=scenario_labels[s], linestyle='-')
    for s in scenarios
]

# Create a new figure just for the legend
fig_legend = plt.figure(figsize=(3, len(scenarios) * 0.4))
fig_legend.legend(
    handles=legend_elements,
    loc='center',
    frameon=True,
    fontsize=8
)
fig_legend.tight_layout()

# Save the legend figure
fig_legend.savefig('./graphs/legend_commodity.png', dpi=300, bbox_inches='tight')
plt.close(fig_legend)