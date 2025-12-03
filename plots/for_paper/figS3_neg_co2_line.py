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
import geopandas as gpd
import yaml
import cartopy.crs as ccrs


# === CONFIGURATION ===
years = [2030, 2040, 2050]

scenarios = [
    "policy_reg_deindustrial",
    "policy_reg_regain",
    "policy_eu_deindustrial",
    "policy_eu_regain"
]

title_map = {
    "policy_reg_deindustrial": "Deindustrialization",
    "policy_reg_regain": "Reindustrialization",
    "policy_eu_deindustrial": "Deindustrialization",
    "policy_eu_regain": "Rendustrialization"
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


# === PARAMETERS ===
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_october/"

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
    ax.text(years[-1] -1.4, 215, 'Max CO$_2$\nremoval', va='center', color='black', fontsize=9)

    if i == 0:
        ax.set_ylabel('CO2 Removed [Mt/yr]')

# === Legends ===


handles_regions = [Line2D([0], [0], color=custom_colors[r], lw=2) for r in group_countries]
labels_regions = list(group_countries.keys())

handles_techs = [Line2D([0], [0], color='black', linestyle=ls, lw=2) for ls in line_styles.values()]
labels_techs = list(line_styles.keys())

legend1 = axes[-1].legend(handles_regions, labels_regions,
                          loc='upper left', bbox_to_anchor=(0.7, 0.96), fontsize=9)
axes[-1].add_artist(legend1)

axes[-1].legend(handles_techs, labels_techs,
                loc='lower left', bbox_to_anchor=(1, 0))

plt.tight_layout()

# === COLUMN GROUP TITLES AND SEPARATOR LINE ===
n_cols = len(scenarios)

if n_cols >= 4:
    # Add vertical line between columns 2 and 3 (i.e., between scenario 2 and 3)
    ax2 = axes[1]  # Column 2
    ax3 = axes[2]  # Column 3
    x2 = ax2.get_position().x1
    x3 = ax3.get_position().x0
    x_line = (x2 + x3) / 2

    fig.lines.append(
        plt.Line2D([x_line, x_line], [0, 1], transform=fig.transFigure, color='black', linewidth=1)
    )

    # Add group titles above the two groups
    x_first_group = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    x_second_group = (axes[2].get_position().x0 + axes[3].get_position().x1) / 2

    fig.text(x_first_group, 0.985, "No Relocation", ha='center', fontsize=14)
    fig.text(x_second_group, 0.985, "Relocation within Europe", ha='center', fontsize=14)


plt.savefig("graphs/neg_co2_lineplot_policy_scenarios.png", bbox_inches='tight', dpi=300)
plt.show()

# %%

# European map


# Define input paths
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
scenario_map = "base_eu_regain/"
res_dir = "results_october/"
regions_fn = root_dir + "resources/" + scenario_map + "regions_onshore_base_s_39.geojson"
config_path = root_dir + res_dir + "base_reg_regain/configs/config.base_s_39___2030.yaml"

# Define country groupings and colors
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

# Load configuration
with open(config_path) as config_file:
    config = yaml.safe_load(config_file)

# Load and preprocess regions
regions = gpd.read_file(regions_fn).set_index("name")
regions["country"] = regions.index.str[:2]
regions = regions.dissolve(by="country")

# Assign group based on country code
def assign_group(country_code):
    for group, countries in group_countries.items():
        if country_code in countries:
            return group
    return "Other"

def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)

regions["group"] = regions.index.map(assign_group)
regions["color"] = regions["group"].map(custom_colors)

# Set up map boundaries and projection
map_opts = config["plotting"]["map"]
if map_opts["boundaries"] is None:
    map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

config["plotting"]["projection"]["name"] = "EqualEarth"
proj = load_projection(config["plotting"])

# Plot map
fig, ax = plt.subplots(figsize=(5, 3), subplot_kw={"projection": proj})

regions.plot(ax=ax, color=regions["color"], edgecolor="black", linewidth=0.5)

#ax.set_title("European Countries by Region Grouping", fontsize=16)
ax.set_xlim(map_opts["boundaries"][:2])
ax.set_ylim(map_opts["boundaries"][2:])

# Create custom legend
import matplotlib.patches as mpatches
legend_patches = [mpatches.Patch(color=color, label=region) for region, color in custom_colors.items()]
#ax.legend(handles=legend_patches, loc='lower left', frameon=True)

plt.tight_layout()
plt.savefig("graphs/europe_map.png", bbox_inches='tight')
plt.show()


# %%
from PIL import Image

# Load images
img_path_lineplot = "graphs/neg_co2_lineplot_policy_scenarios.png"
img_path_map = "graphs/europe_map.png"

img_lineplot = Image.open(img_path_lineplot)
img_map = Image.open(img_path_map)

# === Resize map smaller ===
map_target_width = int(img_lineplot.width * 0.2)  # ~20% width of line plot
map_aspect = img_map.height / img_map.width
map_target_height = int(map_target_width * map_aspect)
img_map_resized = img_map.resize((map_target_width, map_target_height), Image.Resampling.LANCZOS)

# === Paste map over lineplot ===
img_combined = img_lineplot.copy()

# Position the map on the left, vertically centered
x_offset = 50 # int(img_lineplot.width * 10)  # Move it more to the left (3% from left)
y_offset = (img_lineplot.height - img_map_resized.height) // 2

# Paste the map over the line plot (in front)
img_combined.paste(img_map_resized, (x_offset, y_offset))

# Save result
img_combined.save("graphs/neg_co2_lineplot_with_map_overlay.png")

# %%
from PIL import Image

# === Load images ===
img_lineplot = Image.open("graphs/neg_co2_lineplot_policy_scenarios.png")
img_map = Image.open("graphs/europe_map.png")

# === Resize the map to be much smaller ===
map_target_width = int(img_lineplot.width * 0.12)  # 12% of line plot width
map_aspect = img_map.height / img_map.width
map_target_height = int(map_target_width * map_aspect)

img_map_resized = img_map.resize((map_target_width, map_target_height), Image.Resampling.LANCZOS)

# === Create wider canvas by adding right margin ===
right_margin = int(img_lineplot.width * 0.12)  # Space for the small map
new_width = img_lineplot.width + right_margin
new_height = img_lineplot.height

# White background canvas
img_combined = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

# === Paste the lineplot to the left ===
img_combined.paste(img_lineplot, (0, 0))

# === Paste the map into the right margin, vertically centered ===
x_map = img_lineplot.width + int((right_margin - map_target_width) / 2)
y_map = int((new_height - map_target_height) / 2)

img_combined.paste(img_map_resized, (x_map, y_map))

# === Save the final result ===
img_combined.save("graphs/neg_co2_lineplot_with_small_map_on_right.png")
