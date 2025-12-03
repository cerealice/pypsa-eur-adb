# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:14:54 2025

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
scenarios = ["policy_eu_regain","policy_eu_deindustrial"]  # <=== DEFINE SCENARIOS HERE
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
    "policy_eu_regain": "Regain",
    "policy_eu_deindustrial": "Deindustrial"
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
        
        
def normed(s):
    return s / s.sum()


def extract_and_aggregate_cc_tech(n, port: str, pattern: str, timestep: float) -> pd.DataFrame:
    """
    Extracts and aggregates CC-related time series by 2-letter country codes.

    Parameters:
        n: PyPSA network object.
        port (str): Port attribute name from `n.links_t` (e.g., 'p2', 'p3', etc.).
        pattern (str): Substring to filter technology names.
        timestep (float): Timestep multiplier for time aggregation.

    Returns:
        pd.DataFrame: Aggregated CO₂ capture data by 2-letter country code.
    """
    # Filter time series data
    ts_data = -getattr(n.links_t, port).filter(like=pattern, axis=1).sum() * timestep / 1e6

    # Convert to DataFrame (if it's a Series), transpose if needed
    ts_df = ts_data.to_frame() if isinstance(ts_data, pd.Series) else ts_data

    # Assign proper index if missing
    if ts_df.index.name is None:
        ts_df.index.name = 'location'

    # Extract and aggregate by 2-letter country code
    ts_df.index = ts_df.index.str[:2]
    aggregated = ts_df.groupby(ts_df.index).sum()

    return aggregated


    
def plot_negco2_map(n, regions, year, i,j, max_steel,ax=None):
    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Extract negative co2
    
    co2_neg3 = -n.links_t.p3.loc[:, n.links_t.p3.columns.str.contains('DAC')].sum() * timestep / 1e6
    co2_neg4 = -n.links_t.p4.loc[:, n.links_t.p3.columns.str.contains('biomass CHP CC')].sum() * timestep / 1e6
    
    co2_neg = pd.concat([co2_neg3,co2_neg4])
    
    co2_neg.index = co2_neg.index.str[:2]
    
    co2_neg_df = co2_neg.groupby(co2_neg.index).sum().to_frame(name='co2_neg')
    
    # Filter: Keep only countries with >3% of max production
    co2_neg_auxiliary = co2_neg_df.copy()
    # Calculate threshold based on that
    threshold = 0.03 * co2_neg_auxiliary['co2_neg'].max()
    co2_neg_auxiliary = co2_neg_auxiliary[co2_neg_auxiliary['co2_neg'] > threshold]
    
    valid_countries_for_pies = co2_neg_auxiliary[co2_neg_auxiliary['co2_neg'] > 0.5].index.tolist()

    # Extract technology data
    dac           = extract_and_aggregate_cc_tech(n, 'p3', 'DAC', timestep)
    beccs         = extract_and_aggregate_cc_tech(n, 'p4', 'biomass CHP CC', timestep)
    

    cc_tech = pd.concat([
        dac,
        beccs,
    ], axis=1).fillna(0)    
    
    cc_tech.columns = ['DAC','BECCS']#,'Methanolisation']
    
    row_sums = cc_tech.sum(axis=1)
    cc_tech_shares = cc_tech.div(row_sums, axis=0).fillna(0)

    ### MODIFIED: Assign steel production to regions GeoDataFrame using country index
    regions["co2_neg"] = co2_neg_df.co2_neg  # Convert to Mt
    regions = regions.to_crs(proj.proj4_init)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    show_legend = 1
    #show_legend = (i == len(years) - 1 and j == len(scenarios) - 1)

    regions.plot(
        ax=ax,
        column="co2_neg",
        cmap="Blues",
        linewidths=0.3,
        legend=show_legend,
        vmax=max_co2,
        vmin=0,
        edgecolor="black",
        legend_kwds={
            "label": "CO2 removed [MtCO2/yr]",
            "shrink": 0.7,
            "extend": "max",
        } if show_legend else {},
    )

    piecolors = {
        'DAC': '#1f77b4',
        'BECCS': '#9467bd',
        #'Methanolisation': '#17becf'
    }



    for idx, region in regions.iterrows():
        centroid = region['geometry'].centroid
        if idx in valid_countries_for_pies: #steel_prod_shares.index:
            shares = cc_tech_shares.loc[idx]
            pie_values = shares.tolist()
            if sum(pie_values) == 0:
                continue
            size = 150

            # Full circle if 100% of one tech
            max_share = max(pie_values)
            if max_share == 1:
                max_label = pie_values.idxmax() 
                max_color = piecolors[max_label]
                ax.scatter(centroid.x, centroid.y, s=size, color=max_color,
                           edgecolor='black', linewidth=0.5)
                #ax.scatter(centroid.x, centroid.y, s=size, color=piecolors[pie_values.index(max_share)],
                #           edgecolor='black', linewidth=0.5)
                continue

            previous = 0
            markers = []

            for label, ratio in zip(cc_tech_shares.columns, pie_values):
                if ratio == 0:
                    continue
                color = piecolors[label]  # ✅ Get actual hex code
                this = 2 * np.pi * ratio + previous
                x = [0] + np.cos(np.linspace(previous, this, 10)).tolist() + [0]
                y = [0] + np.sin(np.linspace(previous, this, 10)).tolist() + [0]
                xy = np.column_stack([x, y])
                previous = this
                markers.append({
                    'marker': xy,
                    's': size,
                    'facecolor': color,
                    'edgecolor': 'black',
                    'linewidth': 0.5
                })

            for marker in markers:
                ax.scatter(centroid.x, centroid.y, **marker)



    if year == 2050 and scenario == scenarios[0]:
        legend_elements = [
            Patch(facecolor=piecolor, edgecolor='black', label=label)
            for label, piecolor in piecolors.items()
        ]
        ax.legend(
            handles=legend_elements,
            title="CO2 removed",
            loc='center left',
            bbox_to_anchor=(1.45, 0.5),
            frameon=False
        )
    ax.set_facecolor("white")
    ax.set_title(year, fontsize=16, loc="center")


#%% Load and plot

root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
scenario_map = "base_eu_regain/"
res_dir = "results_8h_juno/"
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
max_co2 = 0

for scenario in scenarios:
    for year in years:
        fn = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
        n = pypsa.Network(fn)
        timestep = n.snapshot_weightings.iloc[0, 0]
    
        co2_neg3 = -n.links_t.p3.loc[:, n.links_t.p3.columns.str.contains('DAC')].sum() * timestep / 1e6
        co2_neg4 = -n.links_t.p4.loc[:, n.links_t.p3.columns.str.contains('biomass CHP CC')].sum() * timestep / 1e6
        
        co2_neg = pd.concat([co2_neg3,co2_neg4])
        
        co2_neg.index = co2_neg.index.str[:2]
        
        co2_neg_df = co2_neg.groupby(co2_neg.index).sum()

        # Find max CO2 stored in this snapshot and update max_co2 globally
        current_max = co2_neg_df.max()
        if current_max > max_co2:
            max_co2 = current_max

fig, axes = plt.subplots(len(scenarios), len(years),
                         figsize=(3*len(years), 3*len(scenarios)),
                         subplot_kw={"projection": proj})

for i, year in enumerate(years):
    for j, scenario in enumerate(scenarios):
        fn = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
        ax = axes[j, i]
        n = pypsa.Network(fn)
        plot_negco2_map(n, regions.copy(), year, i,j, max_co2, ax=ax)
        
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
plt.savefig("graphs/neg_co2_map_pie_ccs.png", bbox_inches='tight')
plt.show()
