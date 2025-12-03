# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 12:17:11 2025

@author: Dibella
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import yaml
import cartopy.crs as ccrs

# Define input paths
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
scenario_map = "base_eu_regain/"
res_dir = "results_3h_juno/"
regions_fn = root_dir + "resources/" + scenario_map + "regions_onshore_base_s_39.geojson"
config_path = root_dir + res_dir + "base_eu_regain/configs/config.base_s_39___2030.yaml"

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
