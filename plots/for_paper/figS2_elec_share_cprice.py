# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 10:13:28 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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



def extract_carbon_prices(networks, scenarios, years):
    """
    Extract CO₂ prices [EUR/tCO₂] from loaded PyPSA networks.
    
    Parameters:
        networks (dict): Nested dict of PyPSA networks [scenario][year].
        scenarios (list): List of scenario names.
        years (list): List of years.
    
    Returns:
        dict: Nested dict with CO₂ prices per scenario and year.
    """
    carbon_prices = {}
    for scenario in scenarios:
        carbon_prices[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]

            try:
                price = -n.global_constraints.loc['CO2Limit', 'mu']
                # Handle missing or zero prices (non-binding constraints)
                if pd.isna(price) or price == 0:
                    carbon_prices[scenario][year] = None  # or "—"
                else:
                    carbon_prices[scenario][year] = price
            except KeyError:
                # In case CO2Limit is missing entirely
                carbon_prices[scenario][year] = None  # or "—"
    
    return carbon_prices


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
    hydro_generation = hydro_generation[~hydro_generation.index.str.contains('PHS')]

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
    green_sources = ['Hydro', 'Ror', 'Solar', 'Wind', 'Biomass',"Nuclear"]
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

# %%

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_october/"
scenarios = [
    "policy_reg_deindustrial",
    "policy_eu_deindustrial",
    "policy_reg_regain",
    "policy_eu_regain",
]

scenarios = [
    "base_reg_maintain",
    "policy_reg_deindustrial",
    "policy_reg_maintain",
    "policy_reg_regain",
]

scenario_colors = {
    "base_reg_deindustrial": "#813746",
    "policy_reg_deindustrial": "#FC814A",
    "base_reg_regain": "#6D8088",
    "policy_reg_regain": "#28C76F",
    "import_policy_reg_deindustrial": "#A6A14E",
    "import_policy_reg_regain": "#6BCDC9",
    "policy_eu_deindustrial": "#AA6DA3",
    "policy_eu_regain": "#3C89CD",
    "policy_reg_maintain": "#deca4b",
    "base_reg_maintain": "#6D8088"
}

scenario_labels = {
    "base_reg_deindustrial": "No Climate Policy\nContinued Decline",
    "base_reg_maintain": "No Climate Policy\nStabilization",
    "policy_reg_maintain": "Climate Policy\nStabilization",
    "policy_reg_deindustrial": "Climate Policy\nContinued Decline",
    "policy_eu_deindustrial": "Relocation\nContinued Decline",
    "policy_reg_regain": "Climate Policy\nReindustrialization",
    "policy_eu_regain": "Relocation\nReindustrialization",
}


region_default = 'EU'

group_countries = {
    'North-Western Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'Southern Europe': ['ES', 'IT', 'PT', 'GR'],
    'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}


networks = load_networks(scenarios, years, root_dir, res_dir)
carbon_prices = extract_carbon_prices(networks, scenarios, years)
weighted_avg_prices = compute_weighted_avg_prices(networks)
green_share = get_all_green_shares(networks, group_countries, region_default)

avg_elec_price_2020 = 230 #EUR/MWh https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_204/default/table?lang=en
avg_green_share = 45.3 #% https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Renewable_energy_statistics
avg_cprice = 64.74 #EUR/tCO2 https://icapcarbonaction.com/en/ets/eu-emissions-trading-system-eu-ets

years_plot = [2024, 2030, 2040, 2050]
years = [2030,2040,2050]

fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=False)

scenario_lines = {}


for scenario in scenarios:
    avg_prices = []
    green_shares_plot = []
    carbon_prices_plot = []
    
    for year in years:
        # Electricity price
        all_countries = sum(group_countries.values(), [])
        valid_countries = [c for c in all_countries if c in weighted_avg_prices[scenario][year].columns]
        if not valid_countries:
            avg_price = np.nan
        else:
            avg_price = weighted_avg_prices[scenario][year][valid_countries].mean(axis=1).values[0]
        avg_prices.append(avg_price)
        
        # Green share
        gs_dict = green_share.get(scenario, {}).get(year, {})
        if gs_dict:
            eu_avg_share = np.mean(list(gs_dict.values()))
        else:
            eu_avg_share = np.nan
        green_shares_plot.append(eu_avg_share)
        
        # Carbon price
        cp = carbon_prices.get(scenario, {}).get(year, np.nan)
        carbon_prices_plot.append(cp)
    
    scenario_lines[scenario] = {
        'avg_prices': avg_prices,
        'green_shares': green_shares_plot,
        'carbon_prices': carbon_prices_plot
    }
    
for scenario in scenarios:
    #scenario_lines[scenario]['avg_prices'] = [avg_elec_price_2020] + scenario_lines[scenario]['avg_prices']
    scenario_lines[scenario]['green_shares'] = [avg_green_share] + scenario_lines[scenario]['green_shares']
    
    # For carbon price, only add exogenous value to policy scenarios
    if 'policy' in scenario:
        scenario_lines[scenario]['carbon_prices'] = [avg_cprice] + scenario_lines[scenario]['carbon_prices']
    else:
        scenario_lines[scenario]['carbon_prices'] = [np.nan] + scenario_lines[scenario]['carbon_prices']


# --- Subplot 1: Electricity price ---
ax1 = axes[0]
for scenario in scenarios:
    ax1.plot(
        years,
        scenario_lines[scenario]['avg_prices'],
        'o-',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario],
        linewidth=2
    )
ax1.set_ylabel("EUR/MWh")
ax1.set_title("Average electricity price")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=9)
ax1.set_ylim(bottom=0)

# --- Subplot 2: Green electricity share ---
ax2 = axes[1]
for scenario in scenarios:
    ax2.plot(
        years_plot,
        scenario_lines[scenario]['green_shares'],
        'o-',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario],
        linewidth=2
    )
ax2.set_ylabel("%")
ax2.set_title("Green electricity share")
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.set_ylim(bottom=0)

# --- Subplot 3: Carbon price (policy scenarios only) ---
ax3 = axes[2]
for scenario in scenarios:
    if "policy" in scenario:
        ax3.plot(
            years_plot,
            scenario_lines[scenario]['carbon_prices'],
            'o-',
            label=scenario_labels[scenario],
            color=scenario_colors[scenario],
            linewidth=2
        )
ax3.set_ylabel("EUR/tCO₂")
ax3.set_title("Carbon price")
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.set_ylim(bottom=0)

# Shared X-axis label
for ax in axes:
    ax.set_xticks(years_plot)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("graphs/electricity_share_cprice.png", dpi=300)
plt.show()

