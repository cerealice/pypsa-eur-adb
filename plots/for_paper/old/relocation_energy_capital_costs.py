# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 17:30:33 2025

@author: Dibella
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypsa
import geopandas as gpd
import pycountry
import re
from collections import defaultdict
import matplotlib.patches as mpatches

# Mapping of country codes to names
get_country_name = {
    'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'LU': 'Luxembourg',
    'NL': 'Netherlands', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'LV': 'Latvia', 'LT': 'Lithuania',
    'NO': 'Norway', 'SE': 'Sweden', 'GB': 'United Kingdom', 'IE': 'Ireland', 'ES': 'Spain', 'IT': 'Italy',
    'PT': 'Portugal', 'GR': 'Greece', 'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland',
    'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia', 'AL': 'Albania', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'RS': 'Serbia', 'XK': 'Kosovo'
}


scenario_colors = {
    "policy_eu_regain": "#00B050",
    "policy_reg_regain": "#3AAED8",
}


group_countries = {
    'North-Western Europe': [
        'Austria', 'Belgium', 'Switzerland', 'Germany', 'France', 'Luxembourg', 'Netherlands',
        'Denmark', 'Estonia', 'Finland', 'Latvia', 'Lithuania', 'Norway', 'Sweden',
        'United Kingdom', 'Ireland'
    ],
    'Southern Europe': [
        'Spain', 'Italy', 'Portugal', 'Greece'
    ],
    'Eastern Europe': [
        'Bulgaria', 'Czech Republic', 'Hungary', 'Poland', 'Romania', 'Slovakia', 'Slovenia',
        'Albania', 'Bosnia and Herzegovina', 'Croatia', 'Montenegro', 'North Macedonia',
        'Serbia', 'Kosovo'
    ]
}

nice_scenario_names = {
    'policy_eu_regain': 'Climate policy\nCompetitive industry\nRelocation',
    'policy_reg_regain':  'Climate policy\nCompetitive industry\nHistorical hubs',
}

# Function to get capital cost per country
def get_country_costs(parent_dir, scenario, year):
    file_path = os.path.join(parent_dir, "results_july", scenario, "networks", f"base_s_39___{year}.nc")
    n = pypsa.Network(file_path)

    alinks = n.links.loc[n.links['p_nom_extendable'] == True, 'p_nom_opt'] * \
             n.links.loc[n.links['p_nom_extendable'] == True, 'capital_cost']
    
    alinks = alinks[alinks > 0]
    alinks.index = alinks.index.str[:2]
    alinks = alinks.groupby(alinks.index).sum()/1e9
    alinks.index = alinks.index.to_series().map(lambda x: get_country_name.get(x, "Europe"))
    return alinks

def get_country_name_for_pop(alpha_2_code):
    try:
        if alpha_2_code == "XK":
            return "Kosovo"
        elif alpha_2_code == "HU":
            return "Hungary"
        elif alpha_2_code == "LT":
            return "Lithuania" 
        elif alpha_2_code == "LV":
            return "Latvia"
        country = pycountry.countries.get(alpha_2=alpha_2_code)
        return country.name if country else "Unknown"  # Return 'Unknown' if the code is invalid
    except (ValueError, AttributeError, LookupError):
        return "Unknown"  # Return 'Unknown' if any other error occurs

def normed(s):
    return s / s.sum()



def get_industry_marginal_payments_by_country(parent_dir, scenario, year):

    file_path = os.path.join(parent_dir, "results_july", scenario, "networks", f"base_s_39___{year}.nc")
    n = pypsa.Network(file_path)
    
    industry_keywords = ["steel", "cement", "methanolisation", "ammonia", "HVC"]
    pattern = re.compile("|".join(industry_keywords))
    
    # Select only relevant links (bus1 must contain industry keyword)
    industry_links = n.links[n.links['bus1'].str.contains(pattern)]
    industry_links = industry_links[
        ~industry_links.index.isin(['EU industry methanol']) &
        ~industry_links.index.str.contains('shipping methanol', case=False, na=False)
    ]
    
    payments_by_country = defaultdict(float)
    
    for i in range(5):
        pname = f"p{i}"
        if pname not in n.links_t:
            continue
    
        df_p = n.links_t[pname]
        df_p = df_p[industry_links.index.intersection(df_p.columns)]
    
        for link in df_p.columns:
            bus = n.links.loc[link, f"bus{i}"]
            if bus not in n.buses_t.marginal_price:
                continue
    
            # Country code is usually first 2 characters
            country_code = bus[:2]
    
            power = df_p[link]
            price = n.buses_t.marginal_price[bus]
    
            # Only consider positive power values (consumption)
            payment = (power[power > 0] * price).sum()
            payments_by_country[country_code] += payment
    
    # Convert to Series in b€
    result = pd.Series(payments_by_country) / 1e9
    result.index = result.index.to_series().map(lambda x: get_country_name.get(x, "Europe"))

    return result




# %%


# Collect all data
scenarios = ["policy_eu_regain", "policy_reg_regain"]
years = [2030, 2040, 2050]
cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))


nuts3 = gpd.read_file(parent_dir + '/resources/base_eu_regain/nuts3_shapes.geojson').set_index("index")
nuts3['country'] = nuts3['country'].apply(get_country_name_for_pop)
gdp_by_country = nuts3.groupby('country')['gdp'].sum()
pop_by_country = nuts3.groupby('country')['pop'].sum()
factors = normed(0.6 * normed(gdp_by_country) + 0.4 * normed(pop_by_country))

records = []

for scenario in scenarios:
    for year in years:
        costs = get_industry_marginal_payments_by_country(parent_dir, scenario, year)
        for country, value in costs.items():
            records.append({
                "Country": country,
                "Year": year,
                "Scenario": scenario,
                "Value": value
            })

df_all = pd.DataFrame(records)
# Filter out 'Europe' entries
df_all = df_all[df_all['Country'] != 'Europe']

records_cap = []

for scenario in scenarios:
    for year in years:
        costs = get_country_costs(parent_dir, scenario, year)
        for country, value in costs.items():
            records_cap.append({"Country": country, "Year": year, "Scenario": scenario, "Value": value})
            
df_cap = pd.DataFrame(records_cap)
# Remove 'Europe' entries
df_cap = df_cap[df_cap['Country'] != 'Europe']


# %%


# Map scenario codes to nice names
scenario_labels = [nice_scenario_names.get(s, s) for s in scenario_colors.keys()]
scenario_colors_ordered = [scenario_colors[s] for s in scenario_colors.keys()]

# Prepare legend handles with nice labels
legend_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(scenario_colors_ordered, scenario_labels)
]

country_to_region = {
    country: region
    for region, countries in group_countries.items()
    for country in countries
}

# Filter and map regions
df_all['Region'] = df_all['Country'].map(lambda x: country_to_region.get(x, 'Other'))



# Map countries to regions
df_filtered = df_cap[df_cap['Country'] != 'Europe'].copy()
df_filtered['Region'] = df_filtered['Country'].map(lambda x: country_to_region.get(x, 'Other'))


# Prepare legend handles with nice labels
scenario_labels = [nice_scenario_names.get(s, s) for s in scenario_colors.keys()]
scenario_colors_ordered = [scenario_colors[s] for s in scenario_colors.keys()]
legend_handles = [
    mpatches.Patch(color=color, label=label)
    for color, label in zip(scenario_colors_ordered, scenario_labels)
]

# Prepare regions for both datasets (union of regions for consistent plotting)
regions_energy = [region for region in group_countries.keys() if region in df_all['Region'].unique()]
df_cap = df_cap[df_cap['Country'] != 'Europe'].copy()
df_cap['Region'] = df_cap['Country'].map(lambda x: country_to_region.get(x, 'Other'))

regions_capital = [region for region in group_countries.keys() if region in df_cap['Region'].unique()]
all_regions = sorted(set(regions_energy) | set(regions_capital))
region_order = ['North-Western Europe', 'Southern Europe', 'Eastern Europe']
all_regions = [region for region in region_order if region in all_regions]

ncols = 3
nrows = 2  # top row capital, bottom row energy

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows), sharey='row')

# Flatten axes for easy indexing
axes = axes.flatten()

# --- Plot Capital costs (top row) ---
df_cap_filtered = df_cap[df_cap['Country'] != 'Europe'].copy()
df_cap_filtered['Region'] = df_cap_filtered['Country'].map(lambda x: country_to_region.get(x, 'Other'))
df_cap_grouped = df_cap_filtered.groupby(['Region', 'Year', 'Scenario'], as_index=False)['Value'].sum()

for i, region in enumerate(all_regions):
    ax = axes[i]  # top row index = i (0 to ncols-1)
    data = df_cap_grouped[df_cap_grouped['Region'] == region]
    if data.empty:
        ax.axis('off')
        continue
    sns.barplot(
        data=data,
        x='Year', y='Value', hue='Scenario',
        palette=scenario_colors, dodge=True, ax=ax,
        edgecolor='black'
    )
    ax.set_title(f'{region}')
    ax.set_xlabel('')
    ax.set_ylabel('Capital costs for industry [b€/yr]')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend_.remove()
    ax.set_xticklabels([])



# --- Plot Energy costs (bottom row) ---
df_energy_filtered = df_all[df_all['Country'] != 'Europe'].copy()
df_energy_filtered['Region'] = df_energy_filtered['Country'].map(lambda x: country_to_region.get(x, 'Other'))
df_energy_grouped = df_energy_filtered.groupby(['Region', 'Year', 'Scenario'], as_index=False)['Value'].sum()

for i, region in enumerate(all_regions):
    ax = axes[i + ncols]  # bottom row index = i + ncols
    data = df_energy_grouped[df_energy_grouped['Region'] == region]
    if data.empty:
        ax.axis('off')
        continue
    sns.barplot(
        data=data,
        x='Year', y='Value', hue='Scenario',
        palette=scenario_colors, dodge=True, ax=ax,
        edgecolor='black'
    )
    ax.set_title(f'{region}')
    ax.set_xlabel('')
    ax.set_ylabel('Energy costs for industry [b€/yr]')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend_.remove()


# Remove unused axes (if any)
total_axes = nrows * ncols
for j in range(len(all_regions), ncols):
    axes[j].axis('off')          # top row extra axes
    axes[j + ncols].axis('off')  # bottom row extra axes

# Add shared legend
fig.legend(
    handles=legend_handles,
    loc='upper right',
    bbox_to_anchor=(0.84, 0.89),
    frameon=True,
    title='',
    fontsize='medium',
    title_fontsize='large'
)

plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # leave space on right for legend

# Save figure
plt.savefig('./graphs/industry_costs_combined.png', dpi=300, bbox_inches='tight')

plt.show()
