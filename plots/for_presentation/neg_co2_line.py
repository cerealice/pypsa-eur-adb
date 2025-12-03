# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:19:15 2025

@author: Dibella
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D

# === CONFIGURATION ===
years = [2030, 2040, 2050]


scenarios = [
    #"import_base_reg_deindustrial",
    #"import_base_reg_regain",
    "import_policy_reg_deindustrial",
    "import_policy_reg_regain"
]




title_map = {
    #"base_reg_deindustrial": "Current deindustrialization",
    #"import_base_reg_regain": "No climate policy",
    "import_policy_reg_deindustrial": "Current deindustr trend\nwith intermediate imports",
    "import_policy_reg_regain": "Reindustrialize\nwith intermediate imports"
}




group_countries = {
    'North-West Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL', 'DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE', 'GB', 'IE'],
    'South Europe': ['ES', 'IT', 'PT', 'GR'],
    'East Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI', 'AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}

custom_colors = {
    'South Europe': '#D8973C',
    'North-West Europe': '#1B264F',
    'East Europe': '#9B7EDE',
}

def get_region(code):
    for region, countries in group_countries.items():
        if code in countries:
            return region
    return 'Other'

def get_country_name(code):
    country_map = {
        "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
        "CY": "Cyprus", "CZ": "Czech\nRepublic", "DE": "Germany", "DK": "Denmark",
        "EE": "Estonia", "GR": "Greece", "ES": "Spain", "FI": "Finland",
        "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
        "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
        "MT": "Malta", "NL": "Netherlands", "NO": "Norway", "PL": "Poland",
        "PT": "Portugal", "RO": "Romania", "SE": "Sweden", "SI": "Slovenia",
        "SK": "Slovakia", "UK": "United Kingdom"
    }
    return country_map.get(code, code)

# === PARAMETERS ===
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-industry-imports/"
res_dir = "results/"

# === DATA COLLECTION ===
all_data = []

for scenario in scenarios:
    for year in years:
        fn = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
        n = pypsa.Network(fn)
        timestep = n.snapshot_weightings.iloc[0, 0]

        dac = -n.links_t.p3.loc[:, n.links_t.p3.columns.str.contains('DAC')].sum() * timestep / 1e6
        beccs = -n.links_t.p4.loc[:, n.links_t.p4.columns.str.contains('biomass CHP CC')].sum() * timestep / 1e6

        dac.index = dac.index.str[:2]
        beccs.index = beccs.index.str[:2]

        dac_agg = dac.groupby(dac.index).sum()
        beccs_agg = beccs.groupby(beccs.index).sum()

        df = pd.concat([dac_agg, beccs_agg], axis=1)
        df.columns = ['DAC', 'BECCS']
        df['Scenario'] = scenario
        df['Year'] = year
        df['Country'] = df.index
        df['Region'] = df['Country'].apply(get_region)
        all_data.append(df.reset_index(drop=True))

# === COMBINE AND FILTER ===
data = pd.concat(all_data, ignore_index=True)
data['Total'] = data['DAC'] + data['BECCS']
data = data[data['Total'] >= 1]  # Remove countries with total < 1 Mt

# Group by Region instead of Country
data_grouped = data.groupby(['Scenario', 'Year', 'Region'])[['DAC', 'BECCS']].sum().reset_index()


# %%
# Compute maximum value for y-axis across all scenarios/regions/years
max_val = data_grouped[['DAC', 'BECCS']].values.max()
max_y = math.ceil(max(max_val, 200) / 10.0) * 10  # round up for clarity
# === PLOTTING ===
fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 5), sharey=True)

line_styles = {'DAC': 'solid', 'BECCS': 'dashed'}

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    subset = data_grouped[data_grouped['Scenario'] == scenario]

    regions = subset['Region'].unique()
    techs = ['DAC', 'BECCS']

    for tech in techs:
        for region in regions:
            y_values = []
            for year in years:
                val = subset[(subset['Region'] == region) & (subset['Year'] == year)][tech].sum()
                y_values.append(val)
            ax.plot(years, y_values, label=f"{region} - {tech}", linestyle=line_styles[tech], color=custom_colors.get(region, 'gray'))

    ax.set_title(title_map.get(scenario, scenario))
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticks(years)
    ax.set_ylim(0, max_y)  # Set consistent y-axis limit
    
    ax.axhline(200, color='black', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(years[-1] + 0.5, 200, 'Max CO2\nsequestered', va='center', color='black', fontsize=9)

    if i == 0:
        ax.set_ylabel('CO2 Removed [Mt/yr]')

# === Legends ===


handles_regions = [Line2D([0], [0], color=custom_colors[r], lw=2) for r in group_countries]
labels_regions = list(group_countries.keys())

handles_techs = [Line2D([0], [0], color='black', linestyle=ls, lw=2) for ls in line_styles.values()]
labels_techs = list(line_styles.keys())

legend1 = axes[-1].legend(handles_regions, labels_regions,
                          loc='upper left', bbox_to_anchor=(0.75, 1.1))
axes[-1].add_artist(legend1)

axes[-1].legend(handles_techs, labels_techs,
                loc='lower left', bbox_to_anchor=(1, 0))

plt.tight_layout()
plt.savefig("graphs/neg_co2_lineplot_policy_scenarios.png", bbox_inches='tight', dpi=300)
plt.show()
