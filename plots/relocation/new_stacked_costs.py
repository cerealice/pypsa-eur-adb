import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypsa
import geopandas as gpd
import pycountry

# Mapping of country codes to names
get_country_name = {
    'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'LU': 'Luxembourg',
    'NL': 'Netherlands', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'LV': 'Latvia', 'LT': 'Lithuania',
    'NO': 'Norway', 'SE': 'Sweden', 'GB': 'United Kingdom', 'IE': 'Ireland', 'ES': 'Spain', 'IT': 'Italy',
    'PT': 'Portugal', 'GR': 'Greece', 'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland',
    'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia', 'AL': 'Albania', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'RS': 'Serbia', 'XK': 'Kosovo'
}

# Function to get capital cost per country
def get_country_costs(parent_dir, scenario, year):
    file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
    n = pypsa.Network(file_path)

    alinks = n.links.loc[n.links['p_nom_extendable'] == True, 'p_nom_opt'] * \
             n.links.loc[n.links['p_nom_extendable'] == True, 'capital_cost']
    
    alinks = alinks[alinks > 0]
    alinks.index = alinks.index.str[:2]
    alinks = alinks.groupby(alinks.index).sum()
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
        costs = get_country_costs(parent_dir, scenario, year)
        for country, value in costs.items():
            records.append({"Country": country, "Year": year, "Scenario": scenario, "Value": value})
            
df_all = pd.DataFrame(records)

# Remove 'Europe' entries
df_all = df_all[df_all['Country'] != 'Europe']

# Use pop_by_country (in thousands) to compute cost per capita
df_all['Cost_per_Capita'] = df_all.apply(
    lambda row: row['Value'] / pop_by_country.get(row['Country'], float('nan')) * 1000,
    axis=1
)

# %%
# Create DataFrame

unique_countries = sorted(df_all['Country'].unique())
ncols = 4
nrows = (len(unique_countries) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharey=False)
axes = axes.flatten()

for i, country in enumerate(unique_countries):
    ax = axes[i]
    data = df_all[df_all['Country'] == country]
    sns.barplot(
        data=data,
        x='Year', y='Value', hue='Scenario',
        palette='Set2', dodge=True, ax=ax
    )
    ax.set_title(country)
    ax.set_xlabel('')
    ax.set_ylabel('Cost per Capita')

# Remove extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# One shared legend bottom-right
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', title='Scenario')

plt.tight_layout(rect=[0,0,1,0.95])
plt.show()


# %%


# Plotting
unique_countries = sorted(df_all['Country'].unique())
ncols = 4
nrows = (len(unique_countries) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), sharey=False)
axes = axes.flatten()

for i, country in enumerate(unique_countries):
    ax = axes[i]
    data = df_all[df_all['Country'] == country]
    sns.barplot(
        data=data,
        x='Year', y='Cost_per_Capita', hue='Scenario',
        palette='Set2', dodge=True, ax=ax
    )
    ax.set_title(country)
    ax.set_xlabel('')
    ax.set_ylabel('€/capita')

# Remove unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Shared legend in bottom-right
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


# 1. Reverse mapping: country code → region
country_to_region = {
    country: region
    for region, countries in group_countries.items()
    for country in countries
}

# 2. Replace 'Country' codes/names with Region names
df_all = df_all[df_all['Country'] != 'Europe']  # if needed
df_all['Region'] = df_all['Country'].map(lambda x: country_to_region.get(x, 'Other'))

# 3. Group by Region, Year, Scenario and sum cost
df_grouped = df_all.groupby(['Region', 'Year', 'Scenario'], as_index=False)['Value'].sum()


unique_regions = sorted(df_grouped['Region'].unique())
ncols = 2
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
    ax.set_ylabel('Total Cost (€)')
    ax.legend_.remove()

# Remove unused subplots
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', title='Scenario')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# 1. Map countries to their regions
country_to_region = {
    country: region
    for region, countries in group_countries.items()
    for country in countries
}

# 2. Assign region to each row
df_all = df_all[df_all['Country'] != 'Europe']  # optional
df_all['Region'] = df_all['Country'].map(lambda x: country_to_region.get(x, 'Other'))

# 3. Group cost by Region, Year, Scenario
cost_by_region = df_all.groupby(['Region', 'Year', 'Scenario'], as_index=False)['Value'].sum()

# 4. Compute total population per region (in thousands)
pop_by_region = pop_by_country.groupby(
    pop_by_country.index.map(lambda x: country_to_region.get(x, 'Other'))
).sum()

# 5. Merge population with cost and compute per capita
cost_by_region['Population'] = cost_by_region['Region'].map(pop_by_region)
cost_by_region['€/capita'] = cost_by_region['Value'] / 1e6 / cost_by_region['Population'] * 1000  # M€/person


unique_regions = sorted(cost_by_region['Region'].unique())
ncols = 2
nrows = (len(unique_regions) + ncols - 1) // ncols

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6*ncols, 4*nrows), sharey=False)
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
    ax.set_ylabel('M€/capita')
    ax.legend_.remove()

# Remove extra axes
for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right', title='Scenario')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()