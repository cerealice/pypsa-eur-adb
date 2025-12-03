# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 08:15:43 2025
@author: Dibella
"""

import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml

# === CONFIGURATION ===
scenarios = ["base_eu_regain", "policy_eu_regain"]  # <=== DEFINE SCENARIOS HERE
years = [2030, 2040, 2050]

base_colors = {"regain": "#4F5050", "maintain": "#85877C", "deindustrial": "#B0B2A1"}
policy_colors = {"regain": "#5D8850", "maintain": "#95BF74", "deindustrial": "#C5DEB1"}

scenario_colors = {
    "base_eu_regain": "grey",
    "policy_eu_regain": "green",
    "policy_eu_deindustrial": "orange",
    "policy_reg_regain": "purple"
}
scenario_labels = {
    "base_eu_regain": "Baseline",
    "policy_eu_regain": "Climate policy"
}
lhv_methanol = 5.528  # MWh / t
# %%

def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)

### MODIFIED: Assign country based on first two characters of component index
def assign_country(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["country"] = c.df.index.str[:2]  # Assumes DE0, FR0, etc.
        
def share_green_h2(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    h2_clean = n.links.loc[n.links.index.str.contains('H2 Electrolysis|SMR CC', regex=True, na=False), :].index
    h2_dirty = n.links.loc[n.links.index.str.contains('SMR(?! CC)', regex=True, na=False), :].index

    h2_clean_df = -n.links_t.p1.loc[:, h2_clean].sum() * timestep
    h2_dirty_df = -n.links_t.p1.loc[:, h2_dirty].sum() * timestep

    h2_clean_df.index = h2_clean_df.index.str[:2]
    h2_dirty_df.index = h2_dirty_df.index.str[:2]

    h2_clean_df = h2_clean_df.groupby(h2_clean_df.index).sum()
    h2_dirty_df = h2_dirty_df.groupby(h2_dirty_df.index).sum()

    share_green = round(h2_clean_df / (h2_clean_df + h2_dirty_df), 2)
    return share_green



def plot_methanol_map(n, regions, year, i,j, max_methanol,ax=None):
    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Extract methanol 
    methanol_links = n.links[(n.links.index.str.contains('methanolisation', case=False, na=False))].copy()
    methanol_links["country"] = methanol_links.index.str[:2]

    methanol_prod = -n.links_t.p1.loc[:, methanol_links.index].sum().div(1e6) * timestep / lhv_methanol  # Mt methanol
    methanol_prod.index = methanol_prod.index.str[:2]
    methanol_prod_df = methanol_prod.groupby(methanol_prod.index).sum().to_frame(name='methanol_prod')
    
    # Filter: Keep only countries with >3% of max production
    methanol_prod_auxiliary = methanol_prod_df.copy()
    threshold = 0.03 * methanol_prod_auxiliary['methanol_prod'].max()
    methanol_prod_auxiliary = methanol_prod_auxiliary[methanol_prod_auxiliary['methanol_prod'] > threshold]
    
    valid_countries_for_pies = methanol_prod_auxiliary[methanol_prod_auxiliary['methanol_prod'] > 0.5].index.tolist()

    # Calculate H2 share
    share_green = share_green_h2(n)

    # Align indexes: make sure only matching countries are used
    common_countries = methanol_prod_auxiliary.index.intersection(share_green.index)
    
    share_green = share_green.loc[common_countries]
    
    # Match columns so broadcasting works
    prod_methanol_grey = methanol_prod_auxiliary.mul((1 - share_green), axis=0)
    prod_methanol_green = methanol_prod_auxiliary.mul(share_green, axis=0)

    prod_methanol_green = prod_methanol_green.dropna()
    prod_methanol_grey = prod_methanol_grey.dropna()

    methanol_prod_tech = pd.concat([
        prod_methanol_green, prod_methanol_grey
    ], axis=1)
    methanol_prod_tech.columns = ['Green H2', 'Grey H2']
    
    row_sums = methanol_prod_tech.sum(axis=1)
    methanol_prod_shares = methanol_prod_tech.div(row_sums, axis=0).fillna(0)

    ### MODIFIED: Assign steel production to regions GeoDataFrame using country index
    regions["methanol"] = methanol_prod_df.methanol_prod #Mt
    regions = regions.to_crs(proj.proj4_init)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    show_legend = 1
    #show_legend = (i == len(years) - 1 and j == len(scenarios) - 1)

    regions.plot(
        ax=ax,
        column="methanol",
        cmap="Blues",
        linewidths=0.3,
        legend=show_legend,
        vmax=max_methanol,
        vmin=0,
        edgecolor="black",
        legend_kwds={
            "label": "MeOH prod. [Mt MeOH/yr]",
            "shrink": 0.7,
            "extend": "max",
        } if show_legend else {},
    )

    piecolors = ['green', 'grey']

    for idx, region in regions.iterrows():
        centroid = region['geometry'].centroid
        if idx in valid_countries_for_pies: #steel_prod_shares.index:
            shares = methanol_prod_shares.loc[idx]
            pie_values = shares.tolist()
            if sum(pie_values) == 0:
                continue
            size = 150

            # Full circle if 100% of one tech
            max_share = max(pie_values)
            if max_share == 1:
                ax.scatter(centroid.x, centroid.y, s=size, color=piecolors[pie_values.index(max_share)],
                           edgecolor='black', linewidth=0.5)
                continue

            previous = 0
            markers = []
            for color, ratio in zip(piecolors, pie_values):
                if ratio == 0:
                    continue
                this = 2 * np.pi * ratio + previous
                x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
                y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
                xy = np.column_stack([x, y])
                previous = this
                markers.append({'marker': xy, 's': size, 'facecolor': color, 'edgecolor': 'black', 'linewidth': 0.5})
            for marker in markers:
                ax.scatter(centroid.x, centroid.y, **marker)
                


    if year == 2050 and scenario == scenarios[0]:
        legend_elements = [
            Patch(facecolor=piecolors[0], edgecolor='black', label='Green H2'),
            Patch(facecolor=piecolors[1], edgecolor='black', label='Grey H2'),
        ]
        ax.legend(
            handles=legend_elements,
            title="Methanol with",
            loc='center left',
            bbox_to_anchor=(1.55, 0.5),
            frameon=False
        )
    ax.set_facecolor("white")
    ax.set_title(year, fontsize=16, loc="center")


#%% Load and plot

root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
scenario_map = "base_eu_regain/"
res_dir = "results_3h_juno/"
regions_fn = root_dir + "resources/" + scenario_map + "regions_onshore_base_s_39.geojson"

with open(root_dir + res_dir + "base_eu_regain/configs/config.base_s_39___2030.yaml") as config_file:
    config = yaml.safe_load(config_file)

regions = gpd.read_file(regions_fn).set_index("name")
regions["country"] = regions.index.str[:2]  # Get country code from region name
regions = regions.dissolve(by="country")    # Collapse regions into countries

map_opts = config["plotting"]["map"]
if map_opts["boundaries"] is None:
    map_opts["boundaries"] = regions.total_bounds[[0, 2, 1, 3]] + [-1, 1, -1, 1]

config["plotting"]["projection"]["name"] = "EqualEarth"
proj = load_projection(config["plotting"])

# Calculate the maximum value for the methanol map
max_methanol = 0

for scenario in scenarios:
    for year in years:
        fn = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
        n = pypsa.Network(fn)

        timestep = n.snapshot_weightings.iloc[0, 0]

        # Find methanol production links
        methanol_prod_index = n.links[
            n.links.index.str.contains('methanolisation', case=False, na=False)].index

        methanol_prod = -n.links_t.p1.loc[:, methanol_prod_index].sum()
        methanol_prod.index = methanol_prod.index.str.split(' 0 ').str[0] + ' 0'
        methanol_prod_df = methanol_prod.to_frame(name='methanol_prod')

        methanol_by_region = (
            methanol_prod_df
            .methanol_prod.groupby(level=0)
            .sum()
            .div(1e6) * timestep /lhv_methanol # convert to Mt MeOH/yr
        )

        current_max = methanol_by_region.max()
        if current_max > max_methanol:
            max_methanol = current_max


fig, axes = plt.subplots(len(scenarios), len(years),
                         figsize=(3*len(years), 3*len(scenarios)),
                         subplot_kw={"projection": proj})

for i, year in enumerate(years):
    for j, scenario in enumerate(scenarios):
        fn = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
        ax = axes[j, i]
        n = pypsa.Network(fn)
        plot_methanol_map(n, regions.copy(), year, i,j, max_methanol, ax=ax)
        
        # === Add row title on the left side for each scenario (once per row) ===
        if i == 0:
            label = scenario_labels.get(scenario, scenario)
            ax.annotate(label,
                        xy=(-0.15, 0.5),
                        xycoords='axes fraction',
                        fontsize=14,
                        ha='right', va='center',
                        rotation=90,
                        fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(hspace=0.01) 
plt.savefig("graphs/methanol_map_pie.png", bbox_inches='tight')
plt.show()
