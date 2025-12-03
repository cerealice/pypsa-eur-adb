# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 12:42:01 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CONFIG
years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_october/"

scenarios = [
    "base_reg_maintain",
    "policy_reg_deindustrial",
    "policy_reg_regain",
    "import_policy_reg_deindustrial",
    "import_policy_reg_regain"
]

nice_scenario_names = {
    "policy_reg_deindustrial": "Continued Decline",
    "base_reg_maintain": "Stabilization",
    "policy_reg_regain": "Reindustrialization",
    "import_policy_reg_deindustrial": "Continued Decline",
    "import_policy_reg_regain": "Reindustrialization"
}


# ------------------------------
# Load networks
# ------------------------------
def load_networks(scenarios, years, root_dir, res_dir):
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks



def compute_eu_green_share(n):
    """
    Computes share of green electricity (Hydro, Solar, Wind, Biomass) in total electricity by country.

    Parameters:
        n: PyPSA network
        timestep: float (e.g. 1.0 for hourly resolution)
        year: int
        model: str
        scenario: str
        region_default: str (e.g. 'EU')
        countries: list of country codes

    Returns:
        pd.DataFrame in IAMC format with green electricity share [%] per country and Europe
    """
    # Compute generation from different assets
    timestep = n.snapshot_weightings.iloc[0,0]
    generation_raw = n.generators_t.p.sum() * timestep * 1e-6  # TWh
    generation_raw = generation_raw[~generation_raw.index.str.contains('|'.join(['EU','thermal']), case=False, na=False)]

    hydro_generation = n.storage_units_t.p.sum() * timestep * 1e-6  # TWh
    hydro_generation = hydro_generation[~hydro_generation.index.str.contains('PHS')]

    tot_prod_links = -n.links_t.p1.sum() * timestep * 1e-6
    elec_links_name = ['OCGT', 'coal-', 'CCGT', 'CHP']
    elec_links = tot_prod_links[tot_prod_links.index.str.contains('|'.join(elec_links_name), case=False, na=False)]

    generation = pd.concat([generation_raw, hydro_generation, elec_links])

    # Parse generation index
    generation = (pd.concat([
        generation.index.str.extract(r'([A-Z][A-Z]\d?)', expand=True),          # Node
        generation.index.str.extract(r'([A-Z][A-Z])', expand=True),            # Country
        generation.index.str.extract(r'[A-Z][A-Z]\d? ?\d? (.+)', expand=True),  # Source type
        generation.reset_index()
    ], ignore_index=True, axis=1)
    .drop(3, axis=1)
    .rename(columns={0: 'Node', 1: 'Country', 2: 'Source type', 4: 'Generation [TWh/yr]'}))

    generation = generation[generation['Generation [TWh/yr]'] > 1e-7]

    generation['Source type'] = generation['Source type'].str.split('-').str[0]
    generation['Source type'] = generation['Source type'].str.strip()
    generation['Source type'] = generation['Source type'].str.title()
    generation['Source type'] = generation['Source type'].replace({'Ccgt': 'CCGT', 'Ocgt': 'OCGT'})

    generation = generation.groupby(['Country', 'Source type'])['Generation [TWh/yr]'].sum().reset_index()

    # Compute green share
    green_sources = ['Hydro', 'Ror', 'Solar', 'Wind', 'Biomass',"Nuclear"]
    is_green = generation['Source type'].str.contains('|'.join(green_sources), case=False)

    #gen_by_country = generation.groupby('Country')['Generation [TWh/yr]'].sum()
    #green_gen_by_country = generation[is_green].groupby('Country')['Generation [TWh/yr]'].sum()

    #green_share = (green_gen_by_country / gen_by_country).fillna(0)

    total_eu = generation['Generation [TWh/yr]'].sum()
    green_eu = generation[is_green]['Generation [TWh/yr]'].sum()
    eu_share = green_eu / total_eu if total_eu > 0 else 0
    
    return eu_share


def get_h2_prod_split(n, year, scenario):
    """
    Splits H2 production into Electrolysis (green/grey), SMR CC, and SMR.

    Returns: pd.Series in Mt/yr
    """
    timestep = n.snapshot_weightings.iloc[0, 0]
    lhv_hydrogen = 33.33  # MWh/t H2
    
    # Get total electrolysis production [Mt/yr]
    elec_links = n.links[n.links.index.str.contains("H2 Electrolysis")]
    elec_prod = -n.links_t.p1[elec_links.index].sum().sum() * timestep  # MWh
    elec_prod_mt = elec_prod / lhv_hydrogen / 1e6 #Mt

    # Get SMR CC [Mt/yr]
    smrcc_links = n.links[n.links.index.str.contains("SMR CC")]
    smrcc_prod = -n.links_t.p1[smrcc_links.index].sum().sum() * timestep
    smrcc_prod_mt = smrcc_prod / lhv_hydrogen / 1e6

    # Get SMR [Mt/yr]
    smr_links = n.links[n.links.index.str.contains("SMR(?! CC)", regex=True)]
    smr_prod = -n.links_t.p1[smr_links.index].sum().sum() * timestep
    smr_prod_mt = smr_prod / lhv_hydrogen / 1e6

    # Get green electricity share
    green_share = compute_eu_green_share(n)

    # Split electrolysis
    elec_green = elec_prod_mt * green_share
    elec_grey = elec_prod_mt * (1 - green_share)

    return pd.Series({
        "Electrolysis\n(green)": elec_green,
        "Electrolysis\n(grey)": elec_grey,
        "SMR CC": smrcc_prod_mt,
        "SMR": smr_prod_mt
    })


# Define consistent custom colors
h2_colors = {
    "Electrolysis\n(green)": "#2ca02c",   # green
    "Electrolysis\n(grey)": "#8c564b",    # brown/grey
    "SMR CC": "#1f77b4",                 # blue
    "SMR": "black"                     # grey
}


def plot_hydrogen_pies(networks, scenarios, years, save_path="graphs/hydrogen_pies.png"):
    n_rows, n_cols = len(years), len(scenarios)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))

    if n_rows == 1: axes = np.expand_dims(axes, axis=0)
    if n_cols == 1: axes = np.expand_dims(axes, axis=1)

    # find max H2 production to scale pie radii
    max_total = 0
    for year in years:
        for scenario in scenarios:
            net = networks[(scenario, year)]
            prod = get_h2_prod_split(net, year, scenario)
            max_total = max(max_total, prod.sum())
            max_total = 100

    for i, year in enumerate(years):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            net = networks[(scenario, year)]

            prod = get_h2_prod_split(net, year, scenario)
            values = prod.values
            labels = prod.index
            colors = [h2_colors[l] for l in labels]

            total = values.sum()
            if total > 0:
                # radius proportional to total
                radius = 0.3 + 0.7 * (total / max_total)  # min radius 0.3, max 1.0
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=None,
                    autopct=lambda pct: f"{pct:.0f}%" if pct > 3 else "",
                    startangle=90,
                    colors=colors,
                    radius=radius,
                    textprops={'fontsize': 8}
                )
                for t in autotexts: t.set_fontsize(8)
            
                # total H₂ above the pie
                ax.text(
                    0, radius + 0.05, f"{total:.1f} Mt/yr",
                    ha="center", va="bottom",
                    fontsize=9, weight="bold"
                )

            else:
                ax.text(0.5, 0.5, "No H₂", ha="center", va="center", fontsize=10)
                ax.axis("off")
                continue

            if i == 0:
                ax.set_title(nice_scenario_names[scenario], fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{year}", fontsize=12, rotation=0, labelpad=25, ha="right", va="center")

    # Add group dividers and titles
    if n_cols >= 3:
        # First vertical line between col 0 and col 1
        x1 = (axes[0, 0].get_position().x1 + axes[0, 1].get_position().x0) / 2
        # Second vertical line between col 2 and col 3
        x2 = (axes[0, 2].get_position().x1 + axes[0, 3].get_position().x0) / 2
        
        y_top = axes[0, 0].get_position().y1 * 1.05
        y_bottom = axes[-1, 0].get_position().y0 * 0.95

        fig.lines.append(
            plt.Line2D([x1, x1], [y_bottom, y_top], transform=fig.transFigure,
                       color='black', linewidth=1)
        )
        fig.lines.append(
            plt.Line2D([x2, x2], [y_bottom, y_top], transform=fig.transFigure,
                       color='black', linewidth=1)
        )

        # Titles above each group
        x_group1 = (axes[0, 0].get_position().x0 + axes[0, 0].get_position().x1) / 2
        x_group2 = (axes[0, 1].get_position().x0 + axes[0, 2].get_position().x1) / 2
        x_group3 = (axes[0, 3].get_position().x0 + axes[0, -1].get_position().x1) / 2
        y_group_title = axes[0, 0].get_position().y1 + 0.04

        fig.text(x_group1, y_group_title, "No Climate Policy",
                 ha='center', va='bottom', fontsize=14, fontweight="bold")
        fig.text(x_group2, y_group_title, "Climate Policy\nNo Interm. Imports",
                 ha='center', va='bottom', fontsize=14, fontweight="bold")
        fig.text(x_group3, y_group_title, "Climate Policy\nIntermediate Imports",
                 ha='center', va='bottom', fontsize=14, fontweight="bold")


    # Legend
    fig.legend(
        handles=[plt.matplotlib.patches.Patch(color=color, label=label) for label, color in h2_colors.items()],
        loc="center right",
        bbox_to_anchor=(0.15, 0.65),
        fontsize=10
    )

    #plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()



# %%

# ------------------------------
# Run
# ------------------------------
networks = load_networks(scenarios, years, root_dir, res_dir)

# %%

plot_hydrogen_pies(
    networks=networks,
    scenarios=scenarios,
    years=years,
    save_path="graphs/hydrogen_pies.png"
)

# %% 
from PIL import Image, ImageDraw, ImageFont

# Paths
img_path_stacked = "graphs/european_production_stacked.png"
img_path_h2 = "graphs/hydrogen_pies.png"

# Open images
img_stacked = Image.open(img_path_stacked)
img_h2 = Image.open(img_path_h2)

# DPI and cm to pixels
dpi = 300  # assuming images saved at 300 dpi
cm_to_px = int(dpi / 2.54)

# Crop 1 cm from bottom of first image
w1, h1 = img_stacked.size
img_stacked_cropped = img_stacked.crop((0, cm_to_px, w1, h1 - 2*cm_to_px))
h1_cropped = img_stacked_cropped.size[1]

# Crop 0.5 cm from top of second image
w2, h2 = img_h2.size
img_h2_cropped = img_h2.crop((0, int(0.5*cm_to_px), w2, h2))
h2_cropped = img_h2_cropped.size[1]

# Resize second image width to match first (keep full height)
img_h2_resized = img_h2_cropped#.resize((w1, h2_cropped), Image.Resampling.LANCZOS)

# Combine vertically
combined_height = h1_cropped + h2_cropped
combined_img = Image.new("RGB", (w1, combined_height), color=(255, 255, 255))
combined_img.paste(img_stacked_cropped, (0, 0))
combined_img.paste(img_h2_resized, (90, h1_cropped))

# Crop 1 cm from bottom of combined image
combined_img = combined_img.crop((0, 0, w1, combined_height - 2*cm_to_px))

# Add big labels
def add_label(image, label, position, fontsize=72):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", fontsize)
    except:
        font = ImageFont.load_default()
    draw.text(position, label, fill="black", font=font)
    return image

img_labeled = combined_img.copy()
img_labeled = add_label(img_labeled, "a)", position=(40, 40), fontsize=72)
img_labeled = add_label(img_labeled, "b)", position=(40, h1_cropped + 20), fontsize=72)

# Save
img_labeled.save("graphs/european_production_plus_h2_cropped.png")
