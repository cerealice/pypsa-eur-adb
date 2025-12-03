# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 17:58:49 2025

@author: Dibella
"""


import os
import pypsa
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import yaml
import re
from collections import defaultdict
from matplotlib.patches import Patch
from PIL import Image, ImageDraw, ImageFont
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, Normalize
import numpy as np

# Mapping of country codes to names
get_country_name = {
    'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'DE': 'Germany', 'FR': 'France', 'LU': 'Luxembourg',
    'NL': 'Netherlands', 'DK': 'Denmark', 'EE': 'Estonia', 'FI': 'Finland', 'LV': 'Latvia', 'LT': 'Lithuania',
    'NO': 'Norway', 'SE': 'Sweden', 'GB': 'United Kingdom', 'IE': 'Ireland', 'ES': 'Spain', 'IT': 'Italy',
    'PT': 'Portugal', 'GR': 'Greece', 'BG': 'Bulgaria', 'CZ': 'Czech Republic', 'HU': 'Hungary', 'PL': 'Poland',
    'RO': 'Romania', 'SK': 'Slovakia', 'SI': 'Slovenia', 'AL': 'Albania', 'BA': 'Bosnia and Herzegovina',
    'HR': 'Croatia', 'ME': 'Montenegro', 'MK': 'North Macedonia', 'RS': 'Serbia', 'XK': 'Kosovo'
}

# === LOADER ===
def load_networks(scenarios, years, root_dir, res_dir):
    """
    Load all networks once into a dict keyed by (scenario, year).
    """
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks


def get_industry_marginal_payments_by_country(n):
    
    industry_keywords = ["steel", "cement", "methanol", "NH3", "HVC"]
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
        pname = "p0"
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

def get_industry_marginal_payments_total(n):
    """
    Compute total industry marginal payments across all countries (in bEUR).
    """
    timestep = n.snapshot_weightings.iloc[0,0]
    industry_keywords = ["steel", "cement", "methanol", "NH3", "HVC"]
    pattern = re.compile("|".join(industry_keywords))

    # Select industry links (exclude shipping/misc)
    industry_links = n.links[n.links['bus1'].str.contains(pattern)]
    industry_links = industry_links[
        ~industry_links.index.isin(['EU industry methanol']) &
        ~industry_links.index.str.contains('shipping methanol', case=False, na=False)
    ]

    total_payment = 0.0

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

            power = df_p[link]
            price = n.buses_t.marginal_price[bus]

            # Only positive power values (consumption)
            payment = (power[power > 0] * price).sum() 
            total_payment += payment

    # Convert to bEUR
    return total_payment / 1e9


def get_country_industry_costs_total(n):
    """
    Compute total industry-related capital costs across all countries (in bEUR).
    Only includes extendable industry links.
    """
    timestep = n.snapshot_weightings.iloc[0,0]
    industry_keywords = ["EAF", "BOF", "cement", "methanolisation", "Haber-Bosch", "naphtha steam", "Electrolysis", "Fischer"]
    pattern = re.compile("|".join(industry_keywords))
    
    industry_links = n.links[
        # (n.links['p_nom_extendable'] == True) &
        (n.links['carrier'].str.contains(pattern, na=False)) &
        (~n.links.index.str.contains("EAF-2020", na=False))
    ]


    cap_costs = (industry_links['p_nom_opt'] * industry_links['capital_cost']).sum() 

    return cap_costs / 1e9


# Function to get capital cost per country
def get_country_costs(n):

    alinks = n.links.loc[n.links['p_nom_extendable'] == True, 'p_nom_opt'] * \
             n.links.loc[n.links['p_nom_extendable'] == True, 'capital_cost']
    
    alinks = alinks[alinks > 0]
    alinks.index = alinks.index.str[:2]
    alinks = alinks.groupby(alinks.index).sum()/1e9
    alinks.index = alinks.index.to_series().map(lambda x: get_country_name.get(x, "Europe"))
    return alinks

def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)

def assign_country(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["country"] = c.df.index.str[:2]
        
# %%
# === AUTOMATED INDUSTRIAL PRODUCTION MAP ===


# === CONFIGURATION ===
scenarios = [ "policy_reg_deindustrial", "policy_eu_deindustrial", "policy_reg_regain", "policy_eu_regain",]
years = [2030, 2040, 2050]

commodity_tech_dict = {
    "steel": {
        "Green H2 EAF": "EAF",
        "Grey H2 EAF": "EAF",
        "CH4 EAF": "EAF",
        "BOF": "BOF"
    },
    "cement": {
        "Electric kiln": "Electric Kiln",
        "Oxyfuel kiln": "Oxyfuel",
        "Conventional kiln": "Kiln"
    },
    "ammonia": {
        "Green Ammonia": "H2 Electrolysis",
        "SMR Ammonia": "SMR",
        "SMR CC Ammonia": "SMR CC"
    }
}

scenario_labels = {
    "policy_reg_deindustrial": "No relocation",
    "policy_eu_deindustrial": "Relocation",
    "policy_reg_regain": "No relocation",
    "policy_eu_regain": "Relocation"
}



root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_september_new/"
scenario = "base_eu_regain"
regions_fn = root_dir + "resources/" + scenario + "/regions_onshore_base_s_39.geojson"

networks = load_networks(scenarios, years, root_dir, res_dir)

with open(root_dir + res_dir + "base_reg_regain/configs/config.base_s_39___2030.yaml") as config_file:
    config = yaml.safe_load(config_file)

regions = gpd.read_file(regions_fn).set_index("name")
regions["country"] = regions.index.str[:2]
regions = regions.dissolve(by="country")
config["plotting"]["projection"]["name"] = "EqualEarth"
proj = load_projection(config["plotting"])

# %%
# === Calculate max total industrial production ===
max_total_prod = 0
results = {}

for (scenario, year), n in networks.items():

    assign_country(n)  # if needed for mapping
    timestep = n.snapshot_weightings.iloc[0, 0]
    total_prod = pd.Series(0, index=regions.index)

    all_commodity_prods = {}

    for commodity, techs in commodity_tech_dict.items():
        # Filter links for this commodity
        links = n.links[n.links['bus1'].str.contains(commodity, case=False, na=False)].copy()
        links["country"] = links.index.str[:2]

        # Compute production per link (convert to Gt)
        prod = -n.links_t.p1[links.index].sum() * timestep
        prod.index = prod.index.str[:2]
        prod = prod.groupby(prod.index).sum() / 1e6  

        # Add to totals
        total_prod = total_prod.add(prod, fill_value=0)
        all_commodity_prods[commodity] = prod

    # Track maximum production across all scenarios/years
    max_total_prod = max(max_total_prod, total_prod.max())

    # Store results
    results[(scenario, year)] = {
        "total_prod": total_prod,
        "commodity_prods": all_commodity_prods
    }


# %%
# === TOTAL INDUSTRIAL PRODUCTION MAPS (SUM ALL COMMODITIES) ===
commodities = ["steel", "NH3", "industry methanol", "cement", "hvc"]  #, "H2"]
commodity_search_terms = {
    "steel": "steel",
    "NH3": "NH3",
    "industry methanol": "industry methanol",
    "cement": "cement",
    "hvc": "HVC",
    # "H2": "H2",
}

max_total_prod = 0
total_prod_results = {}

for (scenario, year), n in networks.items():

    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]
    total_prod = pd.Series(dtype=float)

    for commodity, search_term in commodity_search_terms.items():
        # Find links matching this commodity
        links = n.links[n.links['bus1'].str.contains(search_term, case=False, na=False)].copy()
        if links.empty:
            continue

        if commodity == "cement":
            # Exclude cement "process emissions" links
            links = links[~links.index.str.contains("process emissions", case=False, na=False)]

        # Assign countries and exclude EU aggregate
        links["country"] = links.index.str[:2]
        prod = -n.links_t.p1[links.index].sum() * timestep
        prod.index = prod.index.str[:2]
        prod = prod.groupby(prod.index).sum() / 1e6  # Gt
        prod = prod[prod.index != "EU"]

        # Accumulate into total production
        total_prod = total_prod.add(prod, fill_value=0)

    # Track global maximum
    max_total_prod = max(max_total_prod, total_prod.max())

    # Save per-scenario/year results
    total_prod_results[(scenario, year)] = total_prod



# %%

# EXTRA PART FOR 2024

# --- CONFIGURATION ---
csv_path = '../plots_general/capacities_s_39.csv'  # Replace with your CSV file path


# Load full dataframe (if not already loaded)
df_full = pd.read_csv(csv_path, index_col=0)

# Trim index to first 2 letters (country codes)
df_full.index = df_full.index.str[:2]

# Group by country code (sum all plants in same country)
df_grouped_all = df_full.groupby(df_full.index).sum()

df_by_tec = df_grouped_all.sum()/1e3
# Sum across all technologies and all countries for total 2024 production (kt)
total_prod_kt = df_grouped_all.sum(axis=1)


# Convert to Gt
total_prod_gt = total_prod_kt / 1e6

# For each scenario, sum total production across all countries & technologies for 2024
for scenario in scenarios:
    
    # Store in dictionary with a special key, e.g.:
    total_prod_results[(scenario, 2024)] = total_prod_gt


# %%

#from mpl_toolkits.axes_grid1 import make_axes_locatable

years = [2024,2030,2040,2050]

year_max_total_prod = {}

for year in years:
    max_prod = max(
        total_prod_results[(scenario, year)].max()
        for scenario in scenarios
    )
    year_max_total_prod[year] = max_prod

scenario_max_total_prod = {}

for scenario in scenarios:
    max_prod = max(
        total_prod_results[(scenario, year)].max()
        for year in years
    )
    scenario_max_total_prod[scenario] = max_prod
    


vmin = 0
v1 = 0.1
v1_plus = 0.35
v2 = scenario_max_total_prod['policy_eu_deindustrial']
v3 = scenario_max_total_prod['policy_reg_deindustrial']
v4 = scenario_max_total_prod['policy_reg_regain']
vmax = max_total_prod

# Normalize your points between 0 and 1 for colormap creation
points = np.array([vmin, v1, v2, v3, v4, vmax])
normalized_points = (points - vmin) / (vmax - vmin)

# Define your colors at each point (can adjust colors as desired)
colors = ["blue", "lightblue", "white", "pink", "red", "darkred"]

# Create colormap
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    list(zip(normalized_points, colors))
)

# Use Normalize for continuous scaling
norm = Normalize(vmin=vmin, vmax=vmax)
boundaries = [vmin,v1, v1_plus, v2, v3, v4, vmax]
norm = BoundaryNorm(boundaries, ncolors=plt.get_cmap('bwr').N, clip=True)
    


fig2024, ax2024 = plt.subplots(1, 1, figsize=(4, 4), subplot_kw={"projection": proj})

# Pick any scenario for 2024 since it’s the same across all
scenario_for_2024 = scenarios[0]
total_prod_2024 = total_prod_results[(scenario_for_2024, 2024)]
regions["total_industrial_prod"] = total_prod_2024
regions_2024 = regions.to_crs(proj.proj4_init)

regions_2024.plot(
    ax=ax2024,
    column="total_industrial_prod",
    cmap="RdYlGn",
    linewidth=0.3,
    edgecolor="black",
    norm=norm,
)

ax2024.set_title("2024", fontsize=16)
ax2024.set_facecolor("white")
ax2024.set_axis_off()

plt.tight_layout()
plt.savefig("graphs/total_industrial_production_2024_single_map.png", bbox_inches='tight')
plt.show()

years_grid = [2030, 2040, 2050]

fig, axes = plt.subplots(
    len(scenarios), 
    len(years_grid),
    figsize=(3.5 * len(years_grid), 3.5 * len(scenarios)),
    subplot_kw={"projection": proj}
)

fig.subplots_adjust(right=0.85)

for i, scenario in enumerate(scenarios):
    for j, year in enumerate(years_grid):
        ax = axes[i, j]
        total_prod = total_prod_results[(scenario, year)]
        regions["total_industrial_prod"] = total_prod
        regions_proj = regions.to_crs(proj.proj4_init)

        regions_proj.plot(
            ax=ax,
            column="total_industrial_prod",
            cmap="RdYlGn",
            linewidths=0.3,
            legend=False,
            norm=norm,
            edgecolor="black",
        )
        ax.set_facecolor("white")
        if i == 0:
            ax.set_title(year, fontsize=16, loc='center')

        if j == 0:
            ax.annotate(
                scenario_labels.get(scenario, scenario),
                xy=(-0.1, 0.5), xycoords='axes fraction',
                fontsize=14, ha='center', va='center', rotation=90,
                fontweight='bold'
            )
            
        ax.set_axis_off()

# Add colorbar only once on the right
cax = fig.add_axes([1.02, 0.15, 0.02, 0.7]) 
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=norm)
sm._A = []
cbar = fig.colorbar(sm, cax=cax)
cbar.set_label("Total Industrial Production [Gt/yr]", fontsize=16, labelpad=20)
cbar.ax.tick_params(labelsize=14)

# Add rotated group labels left of y-axis labels (adjust as needed)
left_text_x = -0.02
mid_row_1_2 = 0.75
mid_row_3_4 = 0.25
fig.text(left_text_x, mid_row_1_2, "Deindustrialization", rotation=90,
         fontsize=16, fontweight='bold', va='center', ha='center')
fig.text(left_text_x, mid_row_3_4, "Reindustrialization", rotation=90,
         fontsize=16, fontweight='bold', va='center', ha='center')

# Horizontal line between rows 2 and 3
n_rows = len(scenarios)
step = 1 / n_rows
row2_center = 1 - (1 + 0.5) * step
row3_center = 1 - (2 + 0.5) * step
line_y = (row2_center + row3_center) / 2 - (row2_center + row3_center) / 100
fig.lines.append(plt.Line2D([0, 1], [line_y, line_y], transform=fig.transFigure, color='black', linewidth=1.5))

plt.tight_layout()
plt.savefig("graphs/total_industrial_production_per_country_2030_2040_2050.png", bbox_inches='tight')
plt.show()


# %%


# === ADD COSTS ===
records = []
records_cap = []

for (scenario, year), n in networks.items():

    # Industry marginal payments
    costs_ind = get_industry_marginal_payments_by_country(n)
    for country, value in costs_ind.items():
        if country != "EU":
            records.append({
                "Country": country,
                "Year": year,
                "Scenario": scenario,
                "Value": value
            })

    # Capital costs
    costs_cap = get_country_costs(n)
    for country, value in costs_cap.items():
        if country != "EU":
            records_cap.append({
                "Country": country,
                "Year": year,
                "Scenario": scenario,
                "Value": value
            })

df_all = pd.DataFrame(records)
df_cap = pd.DataFrame(records_cap)


# %%
scenario_labels = {
    "policy_reg_deindustrial": "No relocation",
    "policy_eu_deindustrial": "Relocation",
    "policy_reg_regain": "No relocation",
    "policy_eu_regain": "Relocation"
}

scenario_colors = {
    "base_reg_deindustrial": "#813746",
    "policy_reg_deindustrial": "#FC814A",
    "base_reg_regain": "#6D8088",
    "policy_reg_regain": "#28C76F",
    "import_policy_reg_deindustrial": "#A6A14E",
    "import_policy_reg_regain": "#6BCDC9",
    "policy_eu_deindustrial": "#AA6DA3",
    "policy_eu_regain": "#3C89CD"
}



# Filter df_all and df_cap for only these scenarios
df_all_filtered = df_all[df_all['Scenario'].isin(scenarios)].copy()
df_cap_filtered = df_cap[df_cap['Scenario'].isin(scenarios)].copy()

# Sum over all countries (Europe)
df_all_sum = df_all[df_all['Scenario'].isin(scenarios)].groupby(['Year', 'Scenario'], as_index=False)['Value'].sum()
df_cap_sum = df_cap[df_cap['Scenario'].isin(scenarios)].groupby(['Year', 'Scenario'], as_index=False)['Value'].sum()

# Map scenario labels
df_all_sum['Scenario_label'] = df_all_sum['Scenario'].map(scenario_labels)
df_cap_sum['Scenario_label'] = df_cap_sum['Scenario'].map(scenario_labels)

# Merge datasets to align energy & capital costs for stacking
df_combined = pd.merge(
    df_cap_sum,
    df_all_sum,
    on=['Year', 'Scenario', 'Scenario_label'],
    suffixes=('_capital', '_energy')
)


# %%

# Example inputs
years = [2030, 2040, 2050]  # or whatever years you are using
scenarios = [
    "policy_reg_deindustrial",
    "policy_eu_deindustrial",
    "policy_reg_regain",
    "policy_eu_regain"
]

# Initialize data storage
data = []

for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
        
        n = pypsa.Network(file_path)
        
        annual_cost = n.objective / 1e9  # Convert to billion euros per year
        
        data.append({
            'Year': year,
            'Scenario': scenario,
            'Value': annual_cost
        })

# Convert to DataFrame
df_total_cost = pd.DataFrame(data)

# Map scenario labels (same as in your plotting code)
df_total_cost['Scenario_label'] = df_total_cost['Scenario'].map(scenario_labels)

# Assuming df_total_cost has columns: Year, Scenario, Value
df_total_cost_filtered = df_total_cost[df_total_cost['Scenario'].isin(scenarios)].copy()

# Merge for plotting dots later
df_total_cost_filtered['Scenario_label'] = df_total_cost_filtered['Scenario'].map(scenario_labels)



# %%

scenario_labels = {
    "policy_reg_deindustrial": "Deindustrialization\nNo relocation",
    "policy_eu_deindustrial": "Deindustrialization\nRelocation",
    "policy_reg_regain": "Reindustrialization\nNo relocation",
    "policy_eu_regain": "Reindustrialization\nRelocation"
}

fig, ax = plt.subplots(figsize=(7, 6))

bar_width = 0.2
years = sorted(df_combined['Year'].unique())
x = np.arange(len(years))

offsets = {
    scenario: i * bar_width - (len(scenarios) - 1) * bar_width / 2
    for i, scenario in enumerate(scenarios)
}

# Bar plots: Capital + Energy stacked bars
for scenario in scenarios:
    data = df_combined[df_combined['Scenario'] == scenario]
    xpos = x + offsets[scenario]
    
    ax.bar(
        xpos, data['Value_capital'],
        width=bar_width,
        color=scenario_colors[scenario],
        edgecolor='black'
    )
    
    ax.bar(
        xpos, data['Value_energy'],
        width=bar_width,
        bottom=data['Value_capital'],
        color=scenario_colors[scenario],
        hatch='///',
        edgecolor='black'
    )

# Labels & Title
ax.set_title('Industry costs (bars) and total system costs (dots) in bnEUR/a')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)


# Combine all legend items
combined_handles = [
    # Current Deindustrialization
    Patch(facecolor=scenario_colors['policy_reg_deindustrial'], edgecolor='black', label=scenario_labels['policy_reg_deindustrial']),
    Patch(facecolor=scenario_colors['policy_eu_deindustrial'], edgecolor='black', label=scenario_labels['policy_eu_deindustrial']),

    # Reindustrialization
    Patch(facecolor=scenario_colors['policy_reg_regain'], edgecolor='black', label=scenario_labels['policy_reg_regain']),
    Patch(facecolor=scenario_colors['policy_eu_regain'], edgecolor='black', label=scenario_labels['policy_eu_regain']),

    # Cost components
    Patch(facecolor='gray', hatch='', label='Capital costs'),
    Patch(facecolor='gray', hatch='///', label='Energy costs'),

    # Total cost explanation
    Patch(facecolor='none', edgecolor='none', label='● = Total system\n        cost [bnEUR/a]')
]

# Show combined legend inside the plot area (upper left corner)
ax.legend(
    handles=combined_handles,
    loc='upper right',
    fontsize=9,
    frameon=True,
    ncol=1,  # Optional: set to 2 if you want it more compact
    borderpad=0.8
)

# ==== Total System Costs (Dots + Numbers) ====
for scenario in scenarios:
    data = df_total_cost_filtered[df_total_cost_filtered['Scenario'] == scenario]
    xpos = x + offsets[scenario]
    # Dots
    ax.plot(
        xpos, data['Value'], 
        marker='o', 
        linestyle='None', 
        color=scenario_colors[scenario],
        label=f'Total cost - {scenario_labels[scenario]}'
    )
    # Numbers (black, horizontal)
    for xi, yi in zip(xpos, data['Value']):
        ax.text(
            xi, yi + ax.get_ylim()[1]*0.02,  # small offset above dot
            f"{yi:.1f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black',
            rotation=0
        )

# 4. Legend for dots
#dummy_patch = Patch(facecolor='none', edgecolor='none', label='Dots: Total system cost [b€/yr]')
#legend4 = ax.legend(handles=[dummy_patch], loc='upper left', bbox_to_anchor=(1.0, 0.55), frameon=False)
#ax.add_artist(legend4)

plt.tight_layout()
plt.savefig('./graphs/industry_costs_europe_stacked_with_dots_and_numbers.png', dpi=300, bbox_inches='tight')
plt.show()

# %% ALTERNATIVE

# === ADD COSTS with new total functions ===
records = []

for (scenario, year), n in networks.items():

    # Industry marginal payments (energy costs)
    energy_cost = get_industry_marginal_payments_total(n)

    # Industry capital costs
    capital_cost = get_country_industry_costs_total(n)

    records.append({
        "Year": year,
        "Scenario": scenario,
        "Value_energy": energy_cost,
        "Value_capital": capital_cost
    })

# Convert to DataFrame
df_combined = pd.DataFrame(records)

# Add scenario labels
scenario_labels = {
    "policy_reg_deindustrial": "Deindustrialization\nNo relocation",
    "policy_eu_deindustrial": "Deindustrialization\nRelocation",
    "policy_reg_regain": "Reindustrialization\nNo relocation",
    "policy_eu_regain": "Reindustrialization\nRelocation"
}
df_combined["Scenario_label"] = df_combined["Scenario"].map(scenario_labels)


# === Load total system costs (same logic as before) ===
data = []
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
        
        n = pypsa.Network(file_path)
        annual_cost = n.objective / 1e9  # bnEUR/a
        
        data.append({
            "Year": year,
            "Scenario": scenario,
            "Value": annual_cost
        })

df_total_cost = pd.DataFrame(data)
df_total_cost["Scenario_label"] = df_total_cost["Scenario"].map(scenario_labels)


# === Plot stacked bars and dots ===
fig, ax = plt.subplots(figsize=(7, 6))

bar_width = 0.2
years = sorted(df_combined['Year'].unique())
x = np.arange(len(years))

offsets = {
    scenario: i * bar_width - (len(scenarios) - 1) * bar_width / 2
    for i, scenario in enumerate(scenarios)
}

# Bar plots: Capital + Energy stacked bars
for scenario in scenarios:
    data = df_combined[df_combined['Scenario'] == scenario]
    xpos = x + offsets[scenario]
    
    ax.bar(
        xpos, data['Value_capital'],
        width=bar_width,
        color=scenario_colors[scenario],
        edgecolor='black'
    )
    
    ax.bar(
        xpos, data['Value_energy'],
        width=bar_width,
        bottom=data['Value_capital'],
        color=scenario_colors[scenario],
        hatch='///',
        edgecolor='black'
    )

# Labels & Title
ax.set_title('Industry costs (bars) and total system costs (dots) in bnEUR/a')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Combine all legend items
combined_handles = [
    Patch(facecolor=scenario_colors['policy_reg_deindustrial'], edgecolor='black', label=scenario_labels['policy_reg_deindustrial']),
    Patch(facecolor=scenario_colors['policy_eu_deindustrial'], edgecolor='black', label=scenario_labels['policy_eu_deindustrial']),
    Patch(facecolor=scenario_colors['policy_reg_regain'], edgecolor='black', label=scenario_labels['policy_reg_regain']),
    Patch(facecolor=scenario_colors['policy_eu_regain'], edgecolor='black', label=scenario_labels['policy_eu_regain']),
    Patch(facecolor='gray', hatch='', label='Capital costs'),
    Patch(facecolor='gray', hatch='///', label='Energy costs'),
    Patch(facecolor='none', edgecolor='none', label='● = Total system\n        cost [bnEUR/a]')
]
ax.legend(
    handles=combined_handles,
    loc='upper right',
    fontsize=9,
    frameon=True,
    borderpad=0.8
)

# ==== Total System Costs (Dots + Numbers) ====
for scenario in scenarios:
    data = df_total_cost[df_total_cost['Scenario'] == scenario]
    xpos = x + offsets[scenario]
    # Dots
    ax.plot(
        xpos, data['Value'], 
        marker='o', 
        linestyle='None', 
        color=scenario_colors[scenario]
    )
    # Numbers above dots
    for xi, yi in zip(xpos, data['Value']):
        ax.text(
            xi, yi + ax.get_ylim()[1]*0.02,
            f"{yi:.1f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )

plt.tight_layout()
plt.savefig('./graphs/industry_costs_europe_totals.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 6))

bar_width = 0.2
years = sorted(df_combined['Year'].unique())
x = np.arange(len(years))

offsets = {
    scenario: i * bar_width - (len(scenarios) - 1) * bar_width / 2
    for i, scenario in enumerate(scenarios)
}

# ------------------------------
# Bar plots: Capital + Energy stacked
# ------------------------------
for scenario in scenarios:
    data = df_combined[df_combined['Scenario'] == scenario]
    xpos = x + offsets[scenario]
    
    ax.bar(
        xpos, data['Value_capital'],
        width=bar_width,
        color=scenario_colors[scenario],
        edgecolor='black'
    )
    
    ax.bar(
        xpos, data['Value_energy'],
        width=bar_width,
        bottom=data['Value_capital'],
        color=scenario_colors[scenario],
        hatch='///',
        edgecolor='black'
    )

# ------------------------------
# Labels & Title
# ------------------------------
ax.set_title('Industry costs (bars) and total system costs (dots) in bnEUR/a')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# ------------------------------
# Legend
# ------------------------------
combined_handles = [
    Patch(facecolor=scenario_colors['policy_reg_deindustrial'], edgecolor='black',
          label=scenario_labels['policy_reg_deindustrial']),
    Patch(facecolor=scenario_colors['policy_eu_deindustrial'], edgecolor='black',
          label=scenario_labels['policy_eu_deindustrial']),

    Patch(facecolor=scenario_colors['policy_reg_regain'], edgecolor='black',
          label=scenario_labels['policy_reg_regain']),
    Patch(facecolor=scenario_colors['policy_eu_regain'], edgecolor='black',
          label=scenario_labels['policy_eu_regain']),

    Patch(facecolor='gray', hatch='', label='Capital costs'),
    Patch(facecolor='gray', hatch='///', label='Energy costs'),

    Patch(facecolor='none', edgecolor='none', label='● = Total system\n        cost [bnEUR/a]')
]

ax.legend(
    handles=combined_handles,
    loc='upper right',
    fontsize=9,
    frameon=True,
    ncol=1,
    borderpad=0.8
)

# ------------------------------
# Total System Costs (Dots + Numbers)
# ------------------------------
for scenario in scenarios:
    data = df_total_cost_filtered[df_total_cost_filtered['Scenario'] == scenario]
    xpos = x + offsets[scenario]
    ax.plot(
        xpos, data['Value'],
        marker='o',
        linestyle='None',
        color=scenario_colors[scenario]
    )
    for xi, yi in zip(xpos, data['Value']):
        ax.text(
            xi, yi + ax.get_ylim()[1]*0.02,
            f"{yi:.1f}",
            ha='center',
            va='bottom',
            fontsize=10,
            color='black'
        )

# ------------------------------
# Difference lines between dots
# ------------------------------
def add_difference_line(ax, x, y_no_reloc, y_reloc, abs_diff, pct_diff):
    """Draw vertical line with ticks and place absolute and % difference in a small box below the line."""
    # Draw vertical line
    ax.plot([x, x], [y_no_reloc, y_reloc], color="black", linewidth=1.2)
    
    # Draw horizontal ticks
    ax.plot([x - 0.05, x + 0.05], [y_no_reloc, y_no_reloc], color="black", linewidth=1.2)
    ax.plot([x - 0.05, x + 0.05], [y_reloc, y_reloc], color="black", linewidth=1.2)
    
    # Place text just below the lower tick, inside a small box
    ymin, ymax = sorted([y_no_reloc, y_reloc])
    ax.text(
        x, ymin - (ax.get_ylim()[1] * 0.02),  # small offset below
        f"{abs_diff:+.1f} bnEUR\n({pct_diff:+.1f}%)",
        va="top", ha="center", fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )



for i, year in enumerate(years):
    xpos_base = x[i]

    # -------------------
    # Deindustrialization pair
    # -------------------
    y_no_reloc = df_total_cost_filtered[
        (df_total_cost_filtered['Scenario'] == 'policy_reg_deindustrial') &
        (df_total_cost_filtered['Year'] == year)
    ]['Value'].values[0]
    y_reloc = df_total_cost_filtered[
        (df_total_cost_filtered['Scenario'] == 'policy_eu_deindustrial') &
        (df_total_cost_filtered['Year'] == year)
    ]['Value'].values[0]
    
    abs_diff_deind = y_reloc - y_no_reloc
    pct_diff_deind = 100 * abs_diff_deind / y_no_reloc
    
    xpos_deind = xpos_base + (offsets['policy_reg_deindustrial'] + offsets['policy_eu_deindustrial']) / 2
    add_difference_line(ax, xpos_deind, y_no_reloc, y_reloc, abs_diff_deind, pct_diff_deind)

    # -------------------
    # Reindustrialization pair
    # -------------------
    y_no_reloc = df_total_cost_filtered[
        (df_total_cost_filtered['Scenario'] == 'policy_reg_regain') &
        (df_total_cost_filtered['Year'] == year)
    ]['Value'].values[0]
    y_reloc = df_total_cost_filtered[
        (df_total_cost_filtered['Scenario'] == 'policy_eu_regain') &
        (df_total_cost_filtered['Year'] == year)
    ]['Value'].values[0]
    
    abs_diff_reind = y_reloc - y_no_reloc
    pct_diff_reind = 100 * abs_diff_reind / y_no_reloc
    
    xpos_reind = xpos_base + (offsets['policy_reg_regain'] + offsets['policy_eu_regain']) / 2
    add_difference_line(ax, xpos_reind, y_no_reloc, y_reloc, abs_diff_reind, pct_diff_reind)


# ------------------------------
# Save & Show
# ------------------------------
plt.tight_layout()
plt.savefig('./graphs/industry_costs_europe_stacked_with_dots_and_pctdiffs.png', dpi=300, bbox_inches='tight')
plt.show()


# %%

# Paths
img_path_2024 = "graphs/total_industrial_production_2024_single_map.png"
img_path_grid = "graphs/total_industrial_production_per_country_2030_2040_2050.png"
img_path_costs = "graphs/industry_costs_europe_stacked_with_dots_and_pctdiffs.png"

# Open images
img_2024 = Image.open(img_path_2024)
img_grid = Image.open(img_path_grid)
img_costs = Image.open(img_path_costs)

# Scale factor for first image
scale_factor = 1.45

# Resize first image
w1_orig, h1_orig = img_2024.size
w1 = int(w1_orig * scale_factor)
h1 = int(h1_orig * scale_factor)
img_2024_resized = img_2024.resize((w1, h1), Image.Resampling.LANCZOS)

# Second image remains the same
w2, h2 = img_grid.size

# Resize third image to exactly same size as resized first image
img_costs_resized = img_costs.resize((w1, h1), Image.Resampling.LANCZOS)

# Label function
def add_label(image, label, y_offset=0):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10 + y_offset), label, fill="black", font=font)
    return image


# Add labels
img_2024_labeled = add_label(img_2024_resized.copy(), "a)")
img_grid_labeled = add_label(img_grid.copy(), "b)")  # unchanged
img_costs_labeled = add_label(img_costs_resized.copy(), "c)", y_offset=-20)


# Sizes after resizing
w3, h3 = img_costs_labeled.size

# Calculate combined canvas size
combined_width = w1 + w2
combined_height = max(h2, h1 + h3)

# Create blank white canvas
combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

# Paste images
combined_img.paste(img_2024_labeled, (0, 0))       # top-left, scaled
combined_img.paste(img_grid_labeled, (w1, 0))      # top-right, original size
combined_img.paste(img_costs_labeled, (0, h1))     # below first image, same size as first

# Save combined image
combined_img.save("graphs/combined_layout_2rows_equal_size.png")

# %%
# Paths
img_path_2024 = "graphs/total_industrial_production_2024_single_map.png"
img_path_grid = "graphs/total_industrial_production_per_country_2030_2040_2050.png"
img_path_costs = "graphs/industry_costs_europe_stacked_with_dots_and_pctdiffs.png"

# Open images
img_2024 = Image.open(img_path_2024)
img_grid = Image.open(img_path_grid)
img_costs = Image.open(img_path_costs)

# Scale factor for first image
scale_factor = 1.45

# Resize first image
w1_orig, h1_orig = img_2024.size
w1 = int(w1_orig * scale_factor)
h1 = int(h1_orig * scale_factor)
img_2024_resized = img_2024.resize((w1, h1), Image.Resampling.LANCZOS)

# Second image remains the same
w2, h2 = img_grid.size

# Resize third image to same width/height as a)
img_costs_resized = img_costs.resize((w1, h1), Image.Resampling.LANCZOS)

# Label function
def add_label(image, label, y_offset=0):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10 + y_offset), label, fill="black", font=font)
    return image

# Add labels
img_2024_labeled = add_label(img_2024_resized.copy(), "a)")
img_grid_labeled = add_label(img_grid.copy(), "b)")
img_costs_labeled = add_label(img_costs_resized.copy(), "c)", y_offset=-20)

# Sizes after resizing
w3, h3 = img_costs_labeled.size

# Calculate combined canvas size
combined_width = w1 + w2
combined_height = max(h2, h1 + h3)   # ensure enough space for taller side

# Create blank white canvas
combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

# Paste images
combined_img.paste(img_2024_labeled, (0, 0))        # top-left (a)
combined_img.paste(img_costs_labeled, (0, h1))      # directly below a (c)
combined_img.paste(img_grid_labeled, (w1, 0))       # right side (b)

# Save combined image
combined_img.save("graphs/combined_layout_clean.png")

# %%

years = [2030, 2040, 2050]
data_dict = {
    "Annual system cost [bnEUR/a]": {s: {} for s in scenarios},
    "CO2 Price [EUR/tCO2]": {s: {} for s in scenarios},
}

# === LOAD DATA ===
for year in years:
    for scenario in scenarios:
        cwd = os.getcwd()
        parent_dir = os.path.dirname(os.path.dirname(cwd))
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0,0]
        
        data_dict["Annual system cost [bnEUR/a]"][scenario][year] = n.objective / 1e9
        #data_dict["CO2 emissions [MtCO2/yr]"][scenario][year] = n.stores.loc['co2 atmosphere','e_nom_opt'] / 1e6
        data_dict["CO2 Price [EUR/tCO2]"][scenario][year] = -n.global_constraints.loc['CO2Limit','mu']


# === AVERAGE SYSTEM COST OVER 2030, 2040, 2050 ===
average_costs = {
    scenario: sum(data_dict["Annual system cost [bnEUR/a]"][scenario][year] for year in years) / len(years)
    for scenario in scenarios
}

# %%
# === PLOT ===
fig, ax1 = plt.subplots(1,1, figsize=(12, 8), sharex=True)

# ------------------------
# Plot: Annual System Cost
# ------------------------
label_cost = "Annual system cost [bnEUR/a]"
for scenario in scenarios:
    ax1.plot(
        years,
        [data_dict[label_cost][scenario][year] for year in years],
        linestyle="-",
        marker='o',
        label=scenario_labels[scenario],
        color=scenario_colors[scenario]
    )

ax1.set_ylabel("bnEUR/a")
ax1.set_title("Annual System Cost", fontsize=14)
ax1.set_xticks(years)
ax1.set_ylim(bottom=0)
ax1.grid(True, linestyle='--')
ax1.legend(fontsize=11)

def add_difference_line(ax, x, y1, y2, abs_diff, pct_diff):
    """
    Draw vertical line with horizontal ticks and add text annotation
    showing absolute and percentage difference inside a small box.
    """
    # vertical line
    ax.plot([x, x], [y1, y2], color="black", linewidth=1.2)
    # horizontal ticks
    ax.plot([x - 0.2, x + 0.2], [y1, y1], color="black", linewidth=1.2)
    ax.plot([x - 0.2, x + 0.2], [y2, y2], color="black", linewidth=1.2)
    
    # text annotation with box
    ax.text(
        x + 0.4, (y1 + y2) / 2,
        f"{abs_diff:+.1f} bnEUR\n({pct_diff:+.1f}%)",
        va="center", ha="left", fontsize=10, color="black",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )


# --- Differences for each year ---
for x in [y for y in years if y != 2030]:
    # Deindustrialization vs Imports
    y_deind = data_dict[label_cost]["policy_reg_deindustrial"][x]
    y_deind_rel = data_dict[label_cost]["policy_eu_deindustrial"][x]
    diff_abs = - (y_deind - y_deind_rel)
    diff_pct = 100 * diff_abs / y_deind   # % relative to no imports
    add_difference_line(ax1, x + 0.15, y_deind, y_deind_rel, diff_abs, diff_pct)

    # Reindustrialization vs Imports
    y_reind = data_dict[label_cost]["policy_reg_regain"][x]
    y_reind_rel = data_dict[label_cost]["policy_eu_regain"][x]
    diff_abs = - (y_reind - y_reind_rel)
    diff_pct = 100 * diff_abs / y_reind   # % relative to no imports
    add_difference_line(ax1, x + 0.15, y_reind, y_reind_rel, diff_abs, diff_pct)

# Final layout
plt.tight_layout()
plt.savefig("./graphs/costs_reloc_withdiff_all.png", dpi=300)
plt.show()

# === PLOT ===
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8), sharex=True)

label_cost = "Annual system cost [bnEUR/a]"
bar_width = 0.35
x = np.arange(len(years))

# --- Helper to plot bars ---
def plot_bars(base_vals, diff_vals, xpos, color_base, color_diff, label_base, label_diff):
    # Baseline (relocation)
    ax1.bar(
        xpos, base_vals,
        width=bar_width,
        color=color_base,
        edgecolor="black",
        label=label_base,
        zorder=2
    )
    # Overlay difference (no relocation – relocation)
    ax1.bar(
        xpos, diff_vals,
        width=bar_width,
        bottom=base_vals,
        color=color_diff,
        edgecolor="black",
        #hatch="///",
        label=label_diff,
        zorder=3
    )

# --- Values ---
deind_no = [data_dict[label_cost]["policy_reg_deindustrial"][y] for y in years]
deind_rel = [data_dict[label_cost]["policy_eu_deindustrial"][y] for y in years]
reind_no = [data_dict[label_cost]["policy_reg_regain"][y] for y in years]
reind_rel = [data_dict[label_cost]["policy_eu_regain"][y] for y in years]

# Differences
deind_diff = [no - rel for no, rel in zip(deind_no, deind_rel)]
reind_diff = [no - rel for no, rel in zip(reind_no, reind_rel)]

# --- Plot both groups ---
plot_bars(
    deind_rel, deind_diff, x - bar_width/2,
    scenario_colors["policy_eu_deindustrial"], scenario_colors["policy_reg_deindustrial"],
    "Deindustrialization & Relocation", "Δ vs No relocation"
)
plot_bars(
    reind_rel, reind_diff, x + bar_width/2,
    scenario_colors["policy_eu_regain"], scenario_colors["policy_reg_regain"],
    "Reindustrialization & Relocation", "Δ vs No relocation"
)

# Labels & formatting
ax1.set_ylabel("bnEUR/a")
ax1.set_title("Annual System Cost", fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(years)
ax1.set_ylim(bottom=200)
ax1.grid(True, axis="y", linestyle="--", alpha=0.6)

# Legend
ax1.legend(fontsize=11, ncol=1, frameon=True)

# --- Difference annotations ---
def add_difference_line(ax, xpos, y_no, y_rel, abs_diff, pct_diff):
    #ax.plot([xpos, xpos], [y_no, y_rel], color="black", linewidth=1.2)
    #ax.plot([xpos - 0.1, xpos + 0.1], [y_no, y_no], color="black", linewidth=1.2)
    #ax.plot([xpos - 0.1, xpos + 0.1], [y_rel, y_rel], color="black", linewidth=1.2)
    ax.text(
        xpos + 0.2, (y_no + y_rel) / 2,
        f"{abs_diff:+.1f} bnEUR\n({pct_diff:+.1f}%)",
        va="center", ha="left", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.8)
    )

for i, year in enumerate(years):

    # Deindustrialization
    add_difference_line(ax1, x[i] - bar_width/2,
                        deind_no[i], deind_rel[i],
                        deind_no[i] - deind_rel[i],
                        100 * (deind_no[i] - deind_rel[i]) / deind_no[i])
    # Reindustrialization
    add_difference_line(ax1, x[i] + bar_width/2,
                        reind_no[i], reind_rel[i],
                        reind_no[i] - reind_rel[i],
                        100 * (reind_no[i] - reind_rel[i]) / reind_no[i])

plt.tight_layout()
plt.savefig("./graphs/costs_reloc_withdiff_stacked.png", dpi=300)
plt.show()

            
# %%

# Paths
img_path_2024 = "graphs/total_industrial_production_2024_single_map.png"
img_path_grid = "graphs/total_industrial_production_per_country_2030_2040_2050.png"
img_path_costs = "graphs/costs_reloc_withdiff_stacked.png"

# Open images
img_2024 = Image.open(img_path_2024)
img_grid = Image.open(img_path_grid)
img_costs = Image.open(img_path_costs)

# Scale factor for first image
scale_factor = 1.45

# Resize first image
w1_orig, h1_orig = img_2024.size
w1 = int(w1_orig * scale_factor)
h1 = int(h1_orig * scale_factor)
img_2024_resized = img_2024.resize((w1, h1), Image.Resampling.LANCZOS)
# Sizes after resizing
w1, h1 = img_2024_resized.size
w3, h3 = img_costs_resized.size

# Target size for image 2 (square)
target_side = h1 + h3

# Resize image 2 to a square
img_grid_square = img_grid.resize((target_side, target_side), Image.Resampling.LANCZOS)

# Relabel after resize
img_grid_labeled = add_label(img_grid_square.copy(), "b)")

# Recalculate combined canvas size
combined_width = w1 + target_side
combined_height = max(target_side, h1 + h3)

# Create blank white canvas
combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

# Paste images
combined_img.paste(img_2024_labeled, (0, 0))            # top-left (a)
combined_img.paste(img_costs_labeled, (0, h1))          # below a (c)
combined_img.paste(img_grid_labeled, (w1, 0))           # right side (b, now square)

# Save combined image
combined_img.save("graphs/combined_layout_totsyscost.png")
