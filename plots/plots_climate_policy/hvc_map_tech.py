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

def normed(s):
    return s / s.sum()

### MODIFIED: Assign country based on first two characters of component index
def assign_country(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["country"] = c.df.index.str[:2]  # Assumes DE0, FR0, etc.
        
def share_bio_naphtha(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Get relevant links
    oil_ref = n.links.index[n.links.index.str.contains('oil refining', regex=True, na=False)]
    oil_biomass = n.links.index[n.links.index.str.contains('biomass to liquid', regex=True, na=False)]
    oil_ft = n.links.index[n.links.index.str.contains('Fischer-Tropsch', regex=True, na=False)]

    # Calculate total flows
    oil_bio_total = -n.links_t.p1[oil_biomass].sum().sum() * timestep
    oil_ref_total = -n.links_t.p1[oil_ref].sum().sum() * timestep

    # FT flows with country info
    oil_ft_df = -n.links_t.p1[oil_ft].sum(axis=0) * timestep
    oil_ft_df.index = oil_ft_df.index.str[:2]  # extract country code
    oil_ft_by_country = oil_ft_df.groupby(oil_ft_df.index).sum()
    
    # Distribute based on population and GDP
    nuts3 = gpd.read_file('../../resources/base_eu_regain/nuts3_shapes.geojson').set_index("index")
    #nuts3['country'] = nuts3['country'].apply(get_country_name)
    gdp_by_country = nuts3.groupby('country')['gdp'].sum()
    pop_by_country = nuts3.groupby('country')['pop'].sum()
    factors = normed(0.6 * normed(gdp_by_country) + 0.4 * normed(pop_by_country))
    
    oil_bio_by_country = oil_bio_total * factors
    oil_ref_by_country = oil_ref_total * factors
        
    # Compute share
    share_bio = round(oil_bio_by_country / (oil_bio_by_country + oil_ref_by_country + oil_ft_by_country), 2)
    share_ft = round(oil_ft_by_country / (oil_bio_by_country + oil_ref_by_country + oil_ft_by_country), 2)
    return share_bio, share_ft


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


def plot_hvc_map(n, regions, year, i,j, max_hvc,ax=None):
    
    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]

    hvc_links = n.links[(n.links['bus1'].str.contains('HVC', case=False, na=False))].copy()
    hvc_links["country"] = hvc_links.index.str[:2]

    hvc_prod = -n.links_t.p1.loc[:, hvc_links.index].sum() * timestep # kt HVC
    hvc_prod.index = hvc_prod.index.str[:2]
    hvc_prod_df = hvc_prod.groupby(hvc_prod.index).sum().to_frame(name='hvc_prod')
    
    ### ✅ APPLY THRESHOLD
    hvc_prod_auxiliary = hvc_prod_df.copy()
    # Calculate threshold based on that
    threshold = 0.03 * hvc_prod_auxiliary['hvc_prod'].max()
    hvc_prod_auxiliary = hvc_prod_auxiliary[hvc_prod_auxiliary['hvc_prod'] > threshold]
    valid_countries_for_pies = hvc_prod_auxiliary[hvc_prod_auxiliary['hvc_prod'] > 0.5].index.tolist()

    hvc_prod_tech = -n.links_t.p1.loc[:, hvc_links.index].sum()

    methanol_links = hvc_links[hvc_links.index.str.contains('methanol')].index
    naphtha_links = hvc_links[hvc_links.index.str.contains('naphtha')].index

    hvc_methanol = -n.links_t.p1.loc[:, methanol_links].sum() * timestep
    hvc_naphtha = -n.links_t.p1.loc[:, naphtha_links].sum() * timestep

    hvc_methanol.index = hvc_methanol.index.str[:2]
    hvc_naphtha.index = hvc_naphtha.index.str[:2]

    hvc_methanol = hvc_methanol.groupby(hvc_methanol.index).sum().div(1e3)
    hvc_naphtha = hvc_naphtha.groupby(hvc_naphtha.index).sum().div(1e3)

    share_bio, share_ft = share_bio_naphtha(n)

    hvc_bio_naphtha = hvc_naphtha * share_bio
    hvc_ft_naphtha = hvc_naphtha * share_ft
    hvc_fossil_naphtha = hvc_naphtha * (1- share_bio - share_ft)
    
    hvc_prod_tech = pd.concat([
        hvc_fossil_naphtha, hvc_bio_naphtha, hvc_ft_naphtha,
        hvc_methanol
    ], axis=1)
    hvc_prod_tech.columns = ['Fossil naphtha', 'Bio naphtha', 'Fischer-Tropsch','Methanol']
    
    row_sums = hvc_prod_tech.sum(axis=1)
    hvc_prod_shares = hvc_prod_tech.div(row_sums, axis=0).fillna(0)

    ### MODIFIED: Assign hvc production to regions GeoDataFrame using country index
    regions["hvc"] = hvc_prod_df.hvc_prod / 1e3  # Convert to Mt
    regions = regions.to_crs(proj.proj4_init)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    show_legend = 1
    #show_legend = (i == len(years) - 1 and j == len(scenarios) - 1)

    regions.plot(
        ax=ax,
        column="hvc",
        cmap="Blues",
        linewidths=0.3,
        legend=show_legend,
        vmax=max_hvc,
        vmin=0,
        edgecolor="black",
        legend_kwds={
            "label": "HVC production [Mt/yr]",
            "shrink": 0.7,
            "extend": "max",
        } if show_legend else {},
    )


    piecolors = ['black', 'grey', 'orange','green']

    for idx, region in regions.iterrows():
        centroid = region['geometry'].centroid
        if idx in valid_countries_for_pies: #hvc_prod_shares.index:
            shares = hvc_prod_shares.loc[idx]
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
            Patch(facecolor=piecolors[0], edgecolor='black', label='Fossil naphtha'),
            Patch(facecolor=piecolors[1], edgecolor='black', label='Bio naphtha'),
            Patch(facecolor=piecolors[2], edgecolor='black', label='Fischer-Tropsch'),
            Patch(facecolor=piecolors[3], edgecolor='black', label='Methanol'),
        ]
        ax.legend(
            handles=legend_elements,
            title="HVC with",
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

# Calculate the maximum value for the hvc map
max_hvc = 0

for scenario in scenarios:
    for year in years:
        fn = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
        n = pypsa.Network(fn)

        timestep = n.snapshot_weightings.iloc[0, 0]

        # Find hvc production links
        hvc_prod_index = n.links[
            n.links['bus1'].str.contains('HVC', case=False, na=False)].index

        hvc_prod = -n.links_t.p1.loc[:, hvc_prod_index].sum()
        hvc_prod.index = hvc_prod.index.str.split(' 0 ').str[0] + ' 0'
        hvc_prod_df = hvc_prod.to_frame(name='hvc_prod')

        hvc_by_region = (
            hvc_prod_df
            .hvc_prod.groupby(level=0)
            .sum()
            .div(1e3) * timestep # convert to Mt hvc/yr
        )

        current_max = hvc_by_region.max()
        if current_max > max_hvc:
            max_hvc = current_max


fig, axes = plt.subplots(len(scenarios), len(years),
                         figsize=(3*len(years), 3*len(scenarios)),
                         subplot_kw={"projection": proj})

for i, year in enumerate(years):
    for j, scenario in enumerate(scenarios):
        fn = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
        ax = axes[j, i]
        n = pypsa.Network(fn)
        plot_hvc_map(n, regions.copy(), year, i,j, max_hvc, ax=ax)
        
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
plt.savefig("graphs/hvc_map_pie.png", bbox_inches='tight')
plt.show()
