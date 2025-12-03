# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 16:17:38 2025

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

# Mapping of country codes to names
get_country_name = {
    'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'LU': 'Luxembourg',
    'NL': 'Netherlands', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'LV': 'Latvia', 'LT': 'Lithuania',
    'NO': 'Norway', 'SE': 'Sweden', 'GB': 'United Kingdom', 'IE': 'Ireland', 'ES': 'Spain', 'IT': 'Italy',
    'PT': 'Portugal', 'GR': 'Greece', 'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland',
    'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia', 'AL': 'Albania', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'RS': 'Serbia', 'XK': 'Kosovo'
}


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


# %%


def get_industry_marginal_payments_by_country(parent_dir, scenario, year):

    file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
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
    
    # Convert to Series in M€
    result = pd.Series(payments_by_country) / 1e6
    result.index = result.index.to_series().map(lambda x: get_country_name.get(x, "Europe"))

    return result

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
df_filtered = df_all[df_all['Country'] != 'Europe']

unique_countries = sorted(df_filtered['Country'].unique())
ncols = 4
nrows = (len(unique_countries) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharey=False)
axes = axes.flatten()

for i, country in enumerate(unique_countries):
    ax = axes[i]
    data = df_filtered[df_filtered['Country'] == country]
    sns.barplot(
        data=data,
        x='Year', y='Value', hue='Scenario',
        palette='Set2', dodge=True, ax=ax
    )
    ax.set_title(country)
    ax.set_xlabel('')
    ax.set_ylabel('Cost per Capita')
    ax.legend_.remove()  # remove individual legend to keep one shared legend

# Remove extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# One shared legend bottom-right
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', title='Scenario')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %%


scenario_colors = {
    "base_eu_regain": "#464E47",
    "policy_eu_regain": "#00B050",
    "policy_eu_deindustrial": "#FF92D4",
    "policy_reg_regain": "#3AAED8",
    "base_reg_regain": "red"
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

country_to_region = {
    country: region
    for region, countries in group_countries.items()
    for country in countries
}

# Filter out 'Europe' entries
df_filtered = df_all[df_all['Country'] != 'Europe']

# Map countries to their regions
df_filtered['Region'] = df_filtered['Country'].map(lambda x: country_to_region.get(x, 'Other'))

# Group by Region, Year, Scenario and sum cost
df_grouped = df_filtered.groupby(['Region', 'Year', 'Scenario'], as_index=False)['Value'].sum()

unique_regions = sorted(df_grouped['Region'].unique())
ncols = 3
nrows = (len(unique_regions) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), sharey=True)
axes = axes.flatten()

for i, region in enumerate(unique_regions):
    ax = axes[i]
    data = df_grouped[df_grouped['Region'] == region]
    sns.barplot(
        data=data,
        x='Year', y='Value', hue='Scenario',
        palette=scenario_colors, dodge=True, ax=ax
    )
    ax.set_title(region)
    ax.set_xlabel('')
    ax.set_ylabel('Total Energy Cost (€)')
    ax.legend_.remove()

# Remove unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', title='Scenario')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%

# Filter out 'Europe' entries
df_filtered = df_all[df_all['Country'] != 'Europe']

# Map countries to their regions
df_filtered['Region'] = df_filtered['Country'].map(lambda x: country_to_region.get(x, 'Other'))

# Group cost by Region, Year, Scenario
cost_by_region = df_filtered.groupby(['Region', 'Year', 'Scenario'], as_index=False)['Value'].sum()

# Compute total population per region (in thousands)
pop_by_region = pop_by_country.groupby(
    pop_by_country.index.map(lambda x: country_to_region.get(x, 'Other'))
).sum()

# Merge population with cost and compute per capita cost (M€/capita)
cost_by_region['Population'] = cost_by_region['Region'].map(pop_by_region)
cost_by_region['€/capita'] = cost_by_region['Value'] / cost_by_region['Population'] * 1000  # €/person

unique_regions = sorted(cost_by_region['Region'].unique())
ncols = 2
nrows = (len(unique_regions) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), sharey=True)
axes = axes.flatten()

for i, region in enumerate(unique_regions):
    ax = axes[i]
    data = cost_by_region[cost_by_region['Region'] == region]
    sns.barplot(
        data=data,
        x='Year', y='€/capita', hue='Scenario',
        palette=scenario_colors, dodge=True, ax=ax
    )
    ax.set_title(region)
    ax.set_xlabel('')
    ax.set_ylabel('€/capita')
    ax.legend_.remove()

# Remove extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', title='Scenario')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

