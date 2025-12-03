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

# %%

def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)

### MODIFIED: Assign country based on first two characters of component index
def assign_country(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["country"] = c.df.index.str[:2]  # Assumes DE0, FR0, etc.

    
def plot_cement_map(n, regions, year, i,j, max_steel,ax=None):
    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Extract cement 
    cement_links = n.links[(n.links['bus1'].str.contains('cement', case=False, na=False)) &
                          (~n.links['bus1'].str.contains('process emissions', case=False, na=False))].copy()
    cement_links["country"] = cement_links.index.str[:2]

    cement_prod = -n.links_t.p1.loc[:, cement_links.index].sum() * timestep
    cement_prod.index = cement_prod.index.str[:2]
    cement_prod_df = cement_prod.groupby(cement_prod.index).sum().to_frame(name='cement_prod')
    
    # Filter: Keep only countries with >3% of max production
    cement_prod_auxiliary = cement_prod_df.copy()
    threshold = 0.04 * cement_prod_auxiliary['cement_prod'].max()
    cement_prod_auxiliary = cement_prod_auxiliary[cement_prod_auxiliary['cement_prod'] > threshold]
    
    valid_countries_for_pies = cement_prod_auxiliary[cement_prod_auxiliary['cement_prod'] > 0.5].index.tolist()

    # Extract emissions data
    cement_not_captured = -n.links_t.p1.filter(like='cement process emis to atmosphere', axis=1).sum() * timestep
    cement_ccs_not_cap = -n.links_t.p1.filter(like='cement TGR', axis=1).sum() * timestep
    cement_ccs = -n.links_t.p2.filter(like='cement TGR', axis=1).sum() * timestep


    cement_not_captured.index = cement_not_captured.index.str[:2]
    cement_not_captured = cement_not_captured.groupby(cement_not_captured.index).sum()
    cement_ccs.index = cement_ccs.index.str[:2]
    cement_ccs = cement_ccs.groupby(cement_ccs.index).sum()

    # Calculate CCS share
    share_ccs = round(cement_ccs / (cement_ccs + cement_not_captured), 2)

    # Align indexes: make sure only matching countries are used
    common_countries = cement_prod_df.index.intersection(share_ccs.index)
    
    # Subset both to common countries
    cement_prod_auxiliary = cement_prod_df.copy()
    cement_prod_auxiliary = cement_prod_auxiliary.loc[common_countries]
    share_ccs = share_ccs.loc[common_countries]
    
    # Match columns so broadcasting works
    summed_prod_cement_not_captured = cement_prod_auxiliary.mul((1 - share_ccs), axis=0)
    summed_prod_cement_captured = cement_prod_auxiliary.mul(share_ccs, axis=0)

    summed_prod_cement_not_captured = summed_prod_cement_not_captured.dropna()
    summed_prod_cement_captured = summed_prod_cement_captured.dropna()

    cement_prod_tech = pd.concat([
        summed_prod_cement_not_captured, summed_prod_cement_captured
    ], axis=1)
    cement_prod_tech.columns = ['CO2 not captured', 'CO2 captured']
    
    row_sums = cement_prod_tech.sum(axis=1)
    cement_prod_shares = cement_prod_tech.div(row_sums, axis=0).fillna(0)

    ### MODIFIED: Assign steel production to regions GeoDataFrame using country index
    regions["cement"] = cement_prod_df.cement_prod / 1e3  # Convert to Mt
    regions = regions.to_crs(proj.proj4_init)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    show_legend = 1
    #show_legend = (i == len(years) - 1 and j == len(scenarios) - 1)

    regions.plot(
        ax=ax,
        column="cement",
        cmap="Blues",
        linewidths=0.3,
        legend=show_legend,
        vmax=max_steel,
        vmin=0,
        edgecolor="black",
        legend_kwds={
            "label": "Cement prod. [Mt cement/yr]",
            "shrink": 0.7,
            "extend": "max",
        } if show_legend else {},
    )

    piecolors = ['gray', 'green']

    for idx, region in regions.iterrows():
        centroid = region['geometry'].centroid
        if idx in valid_countries_for_pies: #steel_prod_shares.index:
            shares = cement_prod_shares.loc[idx]
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
            Patch(facecolor=piecolors[0], edgecolor='black', label='CO2 not captured'),
            Patch(facecolor=piecolors[1], edgecolor='black', label='CO2 captured'),
        ]
        ax.legend(
            handles=legend_elements,
            title="Process emissions",
            loc='center left',
            bbox_to_anchor=(1.45, 0.5),
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

# Calculate the maximum value for the cement map
max_cement = 0

for scenario in scenarios:
    for year in years:
        fn = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
        n = pypsa.Network(fn)

        timestep = n.snapshot_weightings.iloc[0, 0]

        # Find cement production links
        cement_prod_index = n.links[
            n.links['bus1'].str.contains('cement', case=False, na=False) &
            ~n.links['bus1'].str.contains('process emissions', case=False, na=False)
        ].index

        cement_prod = -n.links_t.p1.loc[:, cement_prod_index].sum()
        cement_prod.index = cement_prod.index.str.split(' 0 ').str[0] + ' 0'
        cement_prod_df = cement_prod.to_frame(name='cement_prod')

        cement_by_region = (
            cement_prod_df
            .cement_prod.groupby(level=0)
            .sum()
            .div(1e3) * timestep  # convert to Mt cement/yr
        )

        current_max = cement_by_region.max()
        if current_max > max_cement:
            print(f"Scenario {scenario} and year {year}")
            max_cement = current_max


fig, axes = plt.subplots(len(scenarios), len(years),
                         figsize=(3*len(years), 3*len(scenarios)),
                         subplot_kw={"projection": proj})

for i, year in enumerate(years):
    for j, scenario in enumerate(scenarios):
        fn = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
        ax = axes[j, i]
        n = pypsa.Network(fn)
        plot_cement_map(n, regions.copy(), year, i,j, max_cement, ax=ax)
        
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
plt.savefig("graphs/cement_map_pie_ccs.png", bbox_inches='tight')
plt.show()
