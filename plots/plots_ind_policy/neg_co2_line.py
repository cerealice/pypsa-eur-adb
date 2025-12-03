# -*- coding: utf-8 -*-
"""
Created on Mon May 26 16:17:46 2025

@author: Dibella
"""


import pypsa
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
scenarios = ["policy_eu_regain", "policy_eu_deindustrial"]
years = [2030, 2040, 2050]

title_map = {
    "policy_eu_regain": "Regain industrial competitiveness",
    "policy_eu_deindustrial": "Deindustrialization"
}

def get_country_name(code):
    country_map = {
        "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CH": "Switzerland",
        "CY": "Cyprus", "CZ": "Czech\nRepublic", "DE": "Germany", "DK": "Denmark",
        "EE": "Estonia", "EL": "Greece", "ES": "Spain", "FI": "Finland",
        "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
        "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia",
        "MT": "Malta", "NL": "Netherlands", "NO": "Norway", "PL": "Poland",
        "PT": "Portugal", "RO": "Romania", "SE": "Sweden", "SI": "Slovenia",
        "SK": "Slovakia", "UK": "United Kingdom"
    }
    return country_map.get(code, code)  # default to code if not found


# === PARAMETERS ===
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_8h_juno/"

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
        all_data.append(df.reset_index(drop=True))

# === COMBINE AND FILTER ===
data = pd.concat(all_data, ignore_index=True)
data['Total'] = data['DAC'] + data['BECCS']
data = data[data['Total'] >= 1]  # Remove countries with total < 1 Mt

# === PLOTTING ===
fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 6), sharey=True)

colors = plt.cm.tab20.colors  # A color palette with enough distinct colors
line_styles = {'DAC': 'solid', 'BECCS': 'dashed'}

for i, scenario in enumerate(scenarios):
    ax = axes[i]
    subset = data[data['Scenario'] == scenario]

    countries = subset['Country'].unique()
    techs = ['DAC', 'BECCS']
    color_map = {country: colors[j % len(colors)] for j, country in enumerate(countries)}

    for tech in techs:
        for country in countries:
            y_values = []
            for year in years:
                val = subset[(subset['Country'] == country) & (subset['Year'] == year)][tech].sum()
                y_values.append(val)
            ax.plot(years, y_values, label=f"{country}-{tech}", linestyle=line_styles[tech], color=color_map[country])

    ax.set_title(title_map.get(scenario, scenario))
    #ax.set_xlabel('Year')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xticks(years)

    if i == 0:
        ax.set_ylabel('CO2 Removed [Mt/yr]')

# === Legends ===
from matplotlib.lines import Line2D

# Legend for countries (colors)
handles_countries = [Line2D([0], [0], color=color_map[c], lw=2) for c in countries]
#labels_countries = [c for c in countries]
labels_countries = [get_country_name(c) for c in countries]

# Legend for technologies (line styles)
handles_techs = [Line2D([0], [0], color='black', linestyle=ls, lw=2) for ls in line_styles.values()]
labels_techs = list(line_styles.keys())

# Add legends to the right-hand plot
legend1 = axes[-1].legend(handles_countries, labels_countries, title='Countries',
                          loc='upper left', bbox_to_anchor=(1.02, 1))
axes[-1].add_artist(legend1)  # Keep both legends

axes[-1].legend(handles_techs, labels_techs, title='Technologies',
                loc='lower left', bbox_to_anchor=(1.02, 0))


# Place legend only on the right subplot
#axes[-1].legend(handles=handles_countries + handles_techs,
#                labels=labels_countries + labels_techs,
#                title='Country / Technology', loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.savefig("graphs/neg_co2_lineplot.png", bbox_inches='tight')
plt.show()