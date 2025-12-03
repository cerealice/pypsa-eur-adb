# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 12:03:02 2025

@author: Dibella
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pypsa
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

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

def compute_green_share_by_country(n, timestep: float, year: int, model: str, scenario: str, region_default: str, countries: list) -> pd.DataFrame:
    """
    Computes share of green electricity (Hydro, Solar, Wind, Biomass) in total electricity by country.

    Parameters:
        n: PyPSA network
        timestep: float (e.g. 1.0 for hourly resolution)
        year: int
        model: str
        scenario: str
        region_default: str (e.g. 'EU')
        countries: list of country codes

    Returns:
        pd.DataFrame in IAMC format with green electricity share [%] per country and Europe
    """
    # Compute generation from different assets
    generation_raw = n.generators_t.p.sum() * timestep * 1e-6  # TWh
    generation_raw = generation_raw[~generation_raw.index.str.contains('|'.join(['EU','thermal']), case=False, na=False)]

    hydro_generation = n.storage_units_t.p.sum() * timestep * 1e-6  # TWh
    hydro_generation = hydro_generation[~hydro_generation.index.str.contains('hydro')]

    tot_prod_links = -n.links_t.p1.sum() * timestep * 1e-6
    elec_links_name = ['OCGT', 'coal-', 'CCGT', 'CHP']
    elec_links = tot_prod_links[tot_prod_links.index.str.contains('|'.join(elec_links_name), case=False, na=False)]

    generation = pd.concat([generation_raw, hydro_generation, elec_links])

    # Parse generation index
    generation = (pd.concat([
        generation.index.str.extract(r'([A-Z][A-Z]\d?)', expand=True),          # Node
        generation.index.str.extract(r'([A-Z][A-Z])', expand=True),            # Country
        generation.index.str.extract(r'[A-Z][A-Z]\d? ?\d? (.+)', expand=True),  # Source type
        generation.reset_index()
    ], ignore_index=True, axis=1)
    .drop(3, axis=1)
    .rename(columns={0: 'Node', 1: 'Country', 2: 'Source type', 4: 'Generation [TWh/yr]'}))

    generation = generation[generation['Generation [TWh/yr]'] > 1e-7]

    generation['Source type'] = generation['Source type'].str.split('-').str[0]
    generation['Source type'] = generation['Source type'].str.strip()
    generation['Source type'] = generation['Source type'].str.title()
    generation['Source type'] = generation['Source type'].replace({'Ccgt': 'CCGT', 'Ocgt': 'OCGT'})

    generation = generation.groupby(['Country', 'Source type'])['Generation [TWh/yr]'].sum().reset_index()

    # Compute green share
    green_sources = ['Hydro', 'Ror', 'Solar', 'Wind', 'Biomass']
    is_green = generation['Source type'].str.contains('|'.join(green_sources), case=False)

    gen_by_country = generation.groupby('Country')['Generation [TWh/yr]'].sum()
    green_gen_by_country = generation[is_green].groupby('Country')['Generation [TWh/yr]'].sum()

    green_share = (green_gen_by_country / gen_by_country).fillna(0)

    # IAMC output format
    rows = []
    for country in countries:
        share = green_share.get(country, 0)
        rows.append({
            'model': model,
            'scenario': scenario,
            'region': country,
            'variable': 'Green Share|Electricity',
            'unit': '%',
            'year': year,
            'value': round(share * 100, 2)
        })

    total_eu = generation['Generation [TWh/yr]'].sum()
    green_eu = generation[is_green]['Generation [TWh/yr]'].sum()
    eu_share = green_eu / total_eu if total_eu > 0 else 0

    rows.append({
        'model': model,
        'scenario': scenario,
        'region': region_default,
        'variable': 'Green Share|Electricity',
        'unit': '%',
        'year': year,
        'value': round(eu_share * 100, 2)
    })

    return pd.DataFrame(rows)




# %%
years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
scenarios = ["policy_eu_regain", "base_eu_regain", "policy_eu_deindustrial", "policy_reg_regain", ]


networks = load_networks(scenarios, years, root_dir, res_dir)

# %%
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
    "base_eu_regain": "No climate policy\nCompetive industry\nRelocation",
    "policy_eu_regain": "Climate policy\nCompetive industry\nRelocation",
    "policy_eu_deindustrial": "Climate policy\nDeindustrialization\nRelocation",
    "policy_reg_regain": "Climate policy\nCompetive industry\nHistorical hubs"
}



weighted_avg_prices = compute_weighted_avg_prices(networks)
weighted_min_prices = compute_weighted_min_prices(networks)
weighted_max_prices = compute_weighted_max_prices(networks)



# --- Compute green share for all scenario/year combinations ---
def get_all_green_shares(networks, group_countries,  region_default):
    result = {}
    for (scenario, year), n in networks.items():
        all_countries = sum(group_countries.values(), [])
        timestep = n.snapshot_weightings.iloc[0,0]
        df = compute_green_share_by_country(
            n=n,
            timestep=timestep,
            year=year,
            model="IndustryModel",
            scenario=scenario,
            region_default=region_default,
            countries=all_countries
        )

        # Convert to DataFrame with region as index
        df = df[df['region'].isin(all_countries)]
        gen_by_country = df.set_index('region')['value']  # This is already % values

        # Group by region
        group_values = {}
        for region, members in group_countries.items():
            vals = gen_by_country[gen_by_country.index.isin(members)]
            if not vals.empty:
                group_values[region] = vals.mean()  # Could be weighted if needed

        if scenario not in result:
            result[scenario] = {}
        result[scenario][year] = group_values
    return result

# Parameters

region_default = 'EU'

# Get green share per scenario-year-region
green_share = get_all_green_shares(networks, group_countries, region_default)

# --- Create custom black-to-green colormap ---
black_to_green = LinearSegmentedColormap.from_list("black_to_green", ["black", "#95C247"])

# Normalize with cutoff at 60%
norm = Normalize(vmin=60, vmax=100, clip=True)

# --- Initialize figure ---
fig, axes = plt.subplots(1, len(scenarios), figsize=(20, 7), sharey=True)

for i, scenario in enumerate(scenarios):
    ax = axes[i]

    # Prepare grid: rows = regions, cols = years
    data = np.zeros((len(group_countries), len(years)))
    text_labels = np.empty_like(data, dtype=object)

    for row_idx, (region, countries) in enumerate(group_countries.items()):
        for col_idx, year in enumerate(years):
            valid = [c for c in countries if c in weighted_avg_prices[scenario][year].columns]
            if not valid:
                continue

            #avg = weighted_avg_prices[scenario][year][valid].mean(axis=1).values[0]
            max_ = weighted_max_prices[scenario][year][valid].max(axis=1).values[0]
            share = green_share[scenario][year].get(region, 0)
            #print(f"Region {region}")

            data[row_idx, col_idx] = share  # Already in %

            text_labels[row_idx, col_idx] = f"{max_:.0f}"

    sns.heatmap(
        data,
        ax=ax,
        annot=text_labels,
        fmt='',
        cmap=black_to_green,
        norm=norm,
        cbar=i == len(scenarios) - 1,
        cbar_kws={'label': ''} if i == len(scenarios) - 1 else None,
        linewidths=0.5,
        linecolor='gray',
        xticklabels=years,
        yticklabels=list(group_countries.keys()) if i == 0 else False,
        annot_kws={"size": 14, "weight": "bold", "va": "center", "ha": "center", "color": "white"}
    )


    if i == 0:
        ax.set_yticks(np.arange(len(group_countries)) + 0.5)
        ax.set_yticklabels(list(group_countries.keys()), rotation=0, fontsize=16, weight="bold")
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
        
    ax.set_title(scenario_labels[scenario], fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    if i == 0:
        ax.tick_params(axis='y', labelsize=9)
    else:
        ax.tick_params(axis='y', left=False)

# --- Global Labels and Legend ---
#fig.suptitle("Electricity Prices and Green Electricity Share", fontsize=14)

legend_text = (
    "Color: Green electricity share [%] (black = ≤60%, green = 100%)\n"
    "Cell text: Max electricity price [€/MWh]"
)
fig.text(0.5, 0.01, legend_text, ha='center', fontsize=14)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("graphs/electricity_price_heatmap_annotated.png", dpi=300)
plt.show()