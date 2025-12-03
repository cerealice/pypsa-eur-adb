# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:25:23 2025

@author: Dibella
"""

import pypsa
import pandas as pd
import numpy as np
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# === CONFIGURATION ===
scenarios = [
    "base_eu_regain",
    "policy_eu_regain",
    "policy_eu_deindustrial",
    "policy_reg_regain"
]
years = [2030, 2040, 2050]

scenario_colors = {
    "base_eu_regain": "#464E47",
    "policy_eu_regain": "#00B050",
    "policy_eu_deindustrial": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}

scenario_labels = {
    "base_eu_regain": "NO CLIMATE POLICY\nCompetive industry\nRelocation",
    "policy_eu_regain": "CLIMATE POLICY\nCompetive industry\nRelocation",
    "policy_eu_deindustrial": "Climate policy\nDEINDUSTRIALIZATION\nRelocation",
    "policy_reg_regain": "Climate policy\nCompetive industry\nHISTORICAL HUBS"
}


# Load projection and assign country function (reuse yours)
def load_projection(plotting_params):
    proj_kwargs = plotting_params.get("projection", dict(name="EqualEarth"))
    proj_func = getattr(ccrs, proj_kwargs.pop("name"))
    return proj_func(**proj_kwargs)

def assign_country(n):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        c.df["country"] = c.df.index.str[:2]

def plot_h2_map(n, regions, year, i, j, max_h2, ax=None):
    assign_country(n)
    timestep = n.snapshot_weightings.iloc[0, 0]

    # --- Extract hydrogen production links ---
    # Assuming hydrogen production links contain 'H2 production' or similar identifiers in bus1
    # Adjust the filter based on your dataset structure

    h2_prod_links = n.links[
        n.links['bus1'].str.contains(' H2', case=False, na=False) &
        ~n.links.index.str.contains('pipeline')
    ].copy()

    if h2_prod_links.empty:
        print(f"No H2 production links for scenario/year {year}")
        return

    h2_prod_links["country"] = h2_prod_links.index.str[:2]

    # Total production per snapshot (negate to convert flow to production)
    h2_prod = -n.links_t.p1.loc[:, h2_prod_links.index] * timestep
    # Sum production over snapshots for each link
    h2_prod_sum = h2_prod.sum(axis=0)
    h2_prod_sum.index = h2_prod_sum.index.str[:2]
    h2_prod_df = h2_prod_sum.groupby(h2_prod_sum.index).sum().to_frame(name='h2_prod')

    # Filter countries with >3% of max production for pies (optional)
    threshold = 0.03 * h2_prod_df['h2_prod'].max()
    valid_countries_for_pies = h2_prod_df[h2_prod_df['h2_prod'] > threshold].index.tolist()

    # --- Extract technology-specific production ---
    # This depends on how your technologies are tagged.
    tech_names = ['Electrolysis', 'SMR CC', 'SMR']

    tech_productions = {}
    for tech in tech_names:
        tech_links = n.links[
            n.links.index.str.contains(tech, case=False, na=False)
        ].index
        
        if tech == 'SMR':
            tech_links = tech_links[~tech_links.str.contains("CC")]

        if len(tech_links) == 0:
            # No production of this tech in this network
            continue

        tech_prod = -n.links_t.p1.loc[:, tech_links] * timestep
        tech_sum = tech_prod.sum(axis=0)
        tech_sum.index = tech_sum.index.str[:2]
        tech_sum_grouped = tech_sum.groupby(tech_sum.index).sum()
        tech_productions[tech] = tech_sum_grouped

    # Combine all tech production into one DataFrame, fill missing with 0
    tech_prod_df = pd.DataFrame(tech_productions).fillna(0)

    # Align with total production countries
    tech_prod_df = tech_prod_df.reindex(h2_prod_df.index).fillna(0)

    # Calculate shares for pies
    tech_shares = tech_prod_df.div(tech_prod_df.sum(axis=1), axis=0).fillna(0)

    # Add total H2 production to regions GeoDataFrame (convert to Mt)
    regions["h2_prod"] = h2_prod_df["h2_prod"] / lhv_hydrogen / 1e6 # convert to TWh/yr
    regions = regions.to_crs(proj.proj4_init)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    # Plot chloropleth
    show_legend = 1
    regions.plot(
        ax=ax,
        column="h2_prod",
        cmap="Purples",
        linewidth=0.3,
        legend=show_legend,
        vmax=max_h2,
        vmin=0,
        edgecolor="black",
        legend_kwds={
            "label": "H₂ Production [Mt H₂/yr]",
            "shrink": 0.7,
            "extend": "max",
        } if show_legend else {},
    )

    # Pie chart colors for techs (adjust or add as needed)
    piecolors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blues, orange, green, red

    for idx, region in regions.iterrows():
        centroid = region['geometry'].centroid
        if idx in valid_countries_for_pies:
            shares = tech_shares.loc[idx]
            pie_values = shares.tolist()
            if sum(pie_values) == 0:
                continue
            size = 150

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

    if year == 2050 and i == 0:
        legend_elements = [
            Patch(facecolor=piecolors[i], edgecolor='black', label=tech.capitalize()) for i, tech in enumerate(tech_names)
        ]
        ax.legend(
            handles=legend_elements,
            loc='center left',
            bbox_to_anchor=(1.4, 0.5),
            frameon=False
        )

    ax.set_facecolor("white")
    ax.set_title(f"{year}", fontsize=16, loc="center")
    ax.spines['geo'].set_visible(False)


#%% Load regions and projection

root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
regions_fn = root_dir + "resources/base_eu_regain/regions_onshore_base_s_39.geojson"

regions = gpd.read_file(regions_fn).set_index("name")
regions["country"] = regions.index.str[:2]
regions = regions.dissolve(by="country")

lhv_hydrogen = 33.3  #MWh/t
    
config = {"plotting": {"projection": {"name": "EqualEarth"}}}
proj = load_projection(config["plotting"])

# Compute max H2 production across all scenarios and years for consistent color scale
max_h2 = 0

for scenario in scenarios:
    for year in years:
        fn = root_dir + f"{res_dir}/{scenario}/networks/base_s_39___{year}.nc"
        n = pypsa.Network(fn)

        assign_country(n)
        timestep = n.snapshot_weightings.iloc[0, 0]

        # Identify hydrogen production links (exclude pipelines)
        h2_prod_links = n.links[
            n.links['bus1'].str.contains(' H2', case=False, na=False) &
            ~n.links.index.str.contains('pipeline')
        ]

        if h2_prod_links.empty:
            continue

        # Get production: sum all link power outputs over time, convert to energy
        h2_prod = -n.links_t.p1.loc[:, h2_prod_links.index].sum() * timestep

        # Assign country from link index
        h2_prod.index = h2_prod.index.str[:2]  # e.g., "DE", "FR"
        h2_prod_df = h2_prod.groupby(h2_prod.index).sum().to_frame(name='h2_prod')

        # Convert to Mt H2/yr: (MWh / LHV) / 1e6
        h2_by_country_mt = (h2_prod_df['h2_prod'] / lhv_hydrogen) / 1e6

        current_max = h2_by_country_mt.max()
        if current_max > max_h2:
            max_h2 = current_max



#%% Plotting all

fig, axes = plt.subplots(len(scenarios), len(years),
                         figsize=(3 * len(years), 3 * len(scenarios)),
                         subplot_kw={"projection": proj})

for i, scenario in enumerate(scenarios):
    for j, year in enumerate(years):
        fn = root_dir + f"{res_dir}/{scenario}/networks/base_s_39___{year}.nc"
        ax = axes[i, j] if len(scenarios) > 1 else axes[j]
        n = pypsa.Network(fn)
        plot_h2_map(n, regions.copy(), year, i, j, max_h2, ax=ax)

        # === Add row title on the left side for each scenario (once per row) ===
        if j == 0:
            label = scenario_labels.get(scenario, scenario)
            ax.annotate(label,
                        xy=(-0.15, 0.5),
                        xycoords='axes fraction',
                        fontsize=14,
                        ha='right', va='center',
                        rotation=90,
                        fontweight='bold')

plt.tight_layout()
plt.savefig("graphs/hydrogen_prod_per_country_pie_chart.png", bbox_inches='tight')
plt.show()

