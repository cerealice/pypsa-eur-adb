# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:49:15 2025

@author: Dibella
"""


import pypsa
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
scenarios = ["policy_eu_regain", "policy_reg_regain"]
years = [2030, 2040, 2050]

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
        all_data.append(df.reset_index().rename(columns={'index': 'Country'}))

# === COMBINE AND FILTER BY 3% THRESHOLD PER SCENARIO & YEAR ===
data = pd.concat(all_data, ignore_index=True)
data['Total'] = data['DAC'] + data['BECCS']

data_filtered = pd.DataFrame()
for (scen, yr), group in data.groupby(['Scenario', 'Year']):
    threshold = 0.03 * group['Total'].max()
    filtered_group = group[group['Total'] >= threshold]
    data_filtered = pd.concat([data_filtered, filtered_group], ignore_index=True)

# === PLOTTING ===
fig, axes = plt.subplots(len(scenarios), len(years), figsize=(18, 10), sharey=True)

for i, scenario in enumerate(scenarios):
    for j, year in enumerate(years):
        ax = axes[i, j]
        subset = data_filtered[(data_filtered['Scenario'] == scenario) & (data_filtered['Year'] == year)]
        subset_sorted = subset.sort_values(by='Total', ascending=False)
        subset_sorted = subset_sorted.set_index(subset_sorted['Link'])
        subset_sorted[['DAC', 'BECCS']].plot(
            kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#9467bd'])

        ax.set_title(f"{scenario}, {year}")
        ax.set_xlabel('')
        if j == 0:
            ax.set_ylabel('CO2 Removed [Mt/yr]')
        else:
            ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("graphs/neg_co2_stacked_bars.png", bbox_inches='tight')
plt.show()