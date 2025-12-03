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
    "policy_reg_deindustrial_flex",
    "policy_reg_regain",
    "policy_reg_regain_flex",
]
years = [2020, 2030, 2040, 2050]

scenario_colors = {

    "policy_reg_deindustrial": "#00B050",
    "policy_reg_deindustrial_flex": "#6AFFAD",  
    "policy_reg_regain": "#3AAED8",
    "policy_reg_regain_flex": "#99D5EB",
}

scenario_labels = {
    
    "policy_reg_deindustrial": "Gradual Deindustr",
    "policy_reg_deindustrial_flex": "Gradual Deindustr\nExtra Flexibility",
    "policy_reg_regain": "Reindustr Strategy",
    "policy_reg_regain_flex": "Reindustr Strategy\nExtra Flexibility",
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
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
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
    #ax.set_ylim(bottom=0)
    if idx == 0:
        ax.set_ylabel("Price [€/t]")
        ax.legend(fontsize=8, loc="lower left")  # Legend only on first subplot

    ax.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("./graphs/commodity_prices_flex.png", dpi=300)
plt.show()

# %%
# === PLOT PERCENTAGE VARIATION (excluding 2020) ===
plot_years = [year for year in years if year != 2020]

fig, axes = plt.subplots(1, len(commodities), figsize=(12, 6), sharex=True, sharey=True)

flex_pairs = [
    ("policy_reg_deindustrial", "policy_reg_deindustrial_flex"),
    ("policy_reg_regain", "policy_reg_regain_flex"),
]

for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for base, flex in flex_pairs:
        base_prices = price_data[commodity].loc[base, plot_years]
        flex_prices = price_data[commodity].loc[flex, plot_years]

        percent_change = (flex_prices - base_prices) / base_prices * 100

        label = "Extra Flexibility in\n" + scenario_labels[base] if idx == 0 else None
        ax.plot(
            plot_years,
            percent_change,
            marker="o",
            linestyle="-",
            label=label,
            color=scenario_colors.get(base, 'black')
        )

    title = {
        "hvc": "Plastics",
        "h2": "Hydrogen"
    }.get(commodity.lower(), commodity.title())

    ax.set_title(f"{title}")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(plot_years)

    if idx == 0:
        ax.set_ylabel("Price change [%]")
        ax.legend(fontsize=8, loc="best")

    ax.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("./graphs/commodity_prices_flex_percentage_diff.png", dpi=300)
plt.show()


# %%

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
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0,0]
        
        data_dict["Annual system cost [b€/yr]"][scenario][year] = n.objective / 1e9
        #data_dict["CO2 emissions [MtCO2/yr]"][scenario][year] = n.stores.loc['co2 atmosphere','e_nom_opt'] / 1e6
        data_dict["CO2 Price [€/tCO2]"][scenario][year] = -n.global_constraints.loc['CO2Limit','mu']



# === PLOT PERCENTAGE VARIATION (Extra Flex vs No Flex) ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True)

flex_pairs = [
    ("policy_reg_deindustrial", "policy_reg_deindustrial_flex"),
    ("policy_reg_regain", "policy_reg_regain_flex"),
]
label_cost = "Annual system cost [b€/yr]"
label_emissions = "CO2 Price [€/tCO2]"


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
plt.savefig("./graphs/costs_emissions_flex_percentage_diff.png", dpi=300)
plt.show()

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

