# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 16:41:05 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# CONFIG
years = [2030, 2040, 2050]
technologies = ['steel', 'cement', 'NH3', 'methanol', 'HVC',"H2"]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_august/"
lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t

feedstock_colors = {
    "Green H2": "#95C247",
    "Grey H2": "#8B9697",
    "Fossil naphtha": "black",
    "Bio naphtha": "#896700",
    "Fischer-Tropsch": "#8A0067",
    #"From methanol": "#FA7E61",
    "With CCS": "#FF9EE7",
    "Without CCS": "black",
    "Green H2 EAF": "#95C247",     
    "Grey H2 EAF": "#8B9697",      
    "CH4 EAF": "#2F52E0",         
    "BOF": "black",  
    "BOF + TGR": "red",
    "SMR": "black",
    "SMR CC": "red",
    "Electrolysis": "green"
}


group_countries = {
    'North-Western Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'Southern Europe': ['ES', 'IT', 'PT', 'GR'],
    'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}

country_to_group = {
    country: group for group, countries in group_countries.items() for country in countries
}

def normed(s):
    return s / s.sum()

def get_production(network, tech, target_region):
    timestep = network.snapshot_weightings.iloc[0, 0]
    links = network.links
    p1 = network.links_t.p1

    # Filter links by technology in bus1
    is_tech = links['bus1'].str.contains(tech, case=False, na=False)
    selected_links = links[is_tech].copy()
    if tech == 'methanol':
        selected_links = selected_links.loc[selected_links.index.str.contains('methanolisation'),:]

    if tech == 'H2':
        selected_links = selected_links.loc[~selected_links.index.str.contains('pipeline'),:]

    if selected_links.empty:
        return 0.0

    # Map link index (country code) to region
    country_codes = selected_links.index.str[:2]
    selected_links['region'] = country_codes.map(country_to_group)

    # Filter for links in the target region
    selected_links = selected_links[selected_links['region'] == target_region]
    if selected_links.empty:
        return 0.0

    # Get production from p1 and sum (MWh)
    total_mwh = -p1[selected_links.index].sum().sum() * timestep 

    # Conversion constants
    lhv_ammonia = 5.166   # MWh / t
    lhv_methanol = 5.528  # MWh / t
    lhv_hydrogen = 33.33 # MWh/ t

    # Convert depending on technology
    if tech.lower() in ['steel', 'cement', 'hvc']:
        # Divide by 1e3 to convert from GWh to TWh or adjust scale as needed
        # You mentioned "directly" Mt, assuming 1e3 converts to Mt here
        return total_mwh / 1e3  
    elif tech.lower() == 'nh3' or tech.lower() == 'ammonia':
        # Convert MWh to tons using LHV, then to Mt
        return total_mwh / lhv_ammonia / 1e6
    elif tech.lower() == 'methanol':
        return total_mwh / lhv_methanol / 1e6
    elif tech.lower() == 'h2':
        return total_mwh / lhv_hydrogen / 1e6
    else:
        # If tech unknown, just return MWh (or you can return zero)
        return total_mwh


def get_production_percentages(network, tech):
    """
    Returns a dict {region: production} and total production sum for a given tech.
    """
    total = 0.0
    region_prod = {}

    for region in group_countries.keys():
        prod = get_production(network, tech, region)
        region_prod[region] = prod
        total += prod

    if total == 0:
        # Avoid division by zero
        return {region: 0.0 for region in region_prod}, 0.0

    # Convert to percentage
    region_pct = {region: (prod / total) * 100 for region, prod in region_prod.items()}
    return region_pct, total

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

def share_bio_naphtha(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Get relevant links
    oil_ref = n.generators.index[n.generators.index.str.contains('oil', regex=True, na=False)]
    oil_biomass = n.links.index[n.links.index.str.contains('biomass to liquid', regex=True, na=False)]
    oil_ft = n.links.index[n.links.index.str.contains('Fischer-Tropsch', regex=True, na=False)]

    # Calculate total flows
    oil_bio_total = -n.links_t.p1[oil_biomass].sum().sum() * timestep
    oil_ref_total = n.generators_t.p[oil_ref].sum().sum() * timestep

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


def get_steel_prod_eu(network):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Filter steel links excluding heat
    steel_links = network.links[
        (network.links['bus1'].str.contains('steel', case=False, na=False)) &
        (~network.links['bus1'].str.contains('heat', case=False, na=False))
    ].copy()
    steel_links["country"] = steel_links.index.str[:2]

    # Only consider European countries (assuming group_countries['EU'] or similar)
    #european_countries = [c for c in steel_links["country"].unique() if c in group_countries['EU']]
    #steel_links = steel_links[steel_links["country"].isin(european_countries)]

    if steel_links.empty:
        return pd.Series({'Green H2 EAF': 0, 'Grey H2 EAF': 0, 'CH4 EAF': 0, 'BOF': 0, "BOF + TGR": 0})

    # Sum steel production per link
    steel_prod = -network.links_t.p1.loc[:, steel_links.index].sum() * timestep
    steel_prod.index = steel_prod.index.str[:2]

    # Get EAF and BOF indices filtered to Europe
    eaf_links = steel_links[steel_links.index.str.contains('EAF')].index
    bof_links = steel_links[steel_links.index.str.contains('BOF')].index

    steel_eaf = -network.links_t.p1.loc[:, eaf_links].sum() * timestep
    steel_bof = -network.links_t.p1.loc[:, bof_links].sum() * timestep

    # Reindex by country code
    steel_eaf.index = steel_eaf.index.str[:2]
    steel_bof.index = steel_bof.index.str[:2]

    # Group sums by country and sum over Europe
    steel_eaf = steel_eaf.groupby(steel_eaf.index).sum()
    steel_bof = steel_bof.groupby(steel_bof.index).sum()
    
    # Emissions not captured and captured by CCS
    uncaptured = -network.links_t.p1.filter(like='steel BOF process emis to atmosphere', axis=1).sum().sum() * timestep
    captured = -network.links_t.p2.filter(like='steel BOF CC', axis=1).sum().sum() * timestep

    total_emissions = captured + uncaptured

    # Share by emissions capture
    share_captured = captured / total_emissions
    share_uncaptured = 1 - share_captured
        
    # Green hydrogen shares by country
    share_green = share_green_h2(network)

    # DRI shares (H2 vs CH4)
    dri_ch4 = -network.links_t.p1.filter(like='CH4 to syn gas DRI', axis=1).sum() * timestep
    dri_h2 = -network.links_t.p1.filter(like='H2 to syn gas DRI', axis=1).sum() * timestep
    share_h2 = dri_h2.sum() / (dri_h2.sum() + dri_ch4.sum()) if (dri_h2.sum() + dri_ch4.sum()) > 0 else 0

    # Calculate production categories per country
    steel_prod_green_h2_eaf = (steel_eaf * share_h2 * share_green).sum() / 1e3  # Mt
    steel_prod_grey_h2_eaf = (steel_eaf * share_h2 * (1 - share_green)).sum() / 1e3
    steel_prod_ch4_eaf = (steel_eaf * (1 - share_h2)).sum() / 1e3
    steel_prod_bof_uncapt = steel_bof.sum() * share_uncaptured  / 1e3
    steel_prod_bof_capt = steel_bof.sum() * share_captured  / 1e3

    return pd.Series({
        'Green H2 EAF': steel_prod_green_h2_eaf,
        'Grey H2 EAF': steel_prod_grey_h2_eaf,
        'CH4 EAF': steel_prod_ch4_eaf,
        'BOF': steel_prod_bof_uncapt ,
        'BOF + TGR': steel_prod_bof_capt
    })


def get_cement_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Filter cement links (exclude process emissions)
    cement_links = n.links[
        (n.links['bus1'].str.contains('cement', case=False, na=False)) &
        (~n.links['bus1'].str.contains('process emissions', case=False, na=False))
    ]
    if cement_links.empty:
        return pd.Series({'With CCS': 0.0, 'Without CCS': 0.0})

    # Total cement production
    total_cement_mwh = -n.links_t.p1[cement_links.index].sum().sum() * timestep

    # Emissions not captured and captured by CCS
    uncaptured = -n.links_t.p1.filter(like='cement process emis to atmosphere', axis=1).sum().sum() * timestep
    captured = -n.links_t.p2.filter(like='cement TGR', axis=1).sum().sum() * timestep

    total_emissions = captured + uncaptured
    if total_emissions == 0:
        return pd.Series({'With CCS': 0.0, 'Without CCS': 0.0})

    # Share by emissions capture
    share_captured = captured / total_emissions
    share_uncaptured = 1 - share_captured

    # Convert total production from MWh to Mt if needed (optional scaling here)
    cement_total_mt = total_cement_mwh / 1e3

    return pd.Series({
        'With CCS': cement_total_mt * share_captured,
        'Without CCS': cement_total_mt * share_uncaptured
    })



def get_ammonia_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    links = n.links
    p1 = n.links_t.p1

    ammonia_links = links[links['bus1'].str.contains("NH3", case=False, na=False)]
    if ammonia_links.empty:
        return pd.Series({"Green H2": 0.0, "Grey H2": 0.0})

    total_p1 = -p1[ammonia_links.index].sum(axis=0) * timestep
    total_mwh = total_p1.sum()

    share_green = share_green_h2(n)
    green_ratio = share_green.mean() if not share_green.empty else 0.0
    grey_ratio = 1 - green_ratio

    lhv_ammonia = 5.166  # MWh/t
    total_mt = total_mwh / lhv_ammonia / 1e6

    return pd.Series({
        "Green H2": total_mt * green_ratio,
        "Grey H2": total_mt * grey_ratio
    })


def get_methanol_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    links = n.links
    p1 = n.links_t.p1

    # Identify production links
    methanol_links = links[links['bus1'].str.contains("methanol", case=False, na=False)]
    methanol_links = methanol_links.loc[methanol_links.index.str.contains('methanolisation'),:]
    if methanol_links.empty:
        return pd.Series({"Green H2": 0.0, "Grey H2": 0.0})

    total_p1 = -p1[methanol_links.index].sum(axis=0) * timestep
    total_mwh = total_p1.sum()

    # Estimate shares from hydrogen origin
    share_green = share_green_h2(n)
    green_ratio = share_green.mean() if not share_green.empty else 0.0
    grey_ratio = 1 - green_ratio

    lhv_methanol = 5.528  # MWh/t
    total_mt = total_mwh / lhv_methanol / 1e6

    return pd.Series({
        "Green H2": total_mt * green_ratio,
        "Grey H2": total_mt * grey_ratio
    })


def get_hvc_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]

    # Select HVC-related links
    hvc_links = n.links[n.links['bus1'].str.contains('HVC', case=False, na=False)]
    if hvc_links.empty:
        return pd.Series({k: 0.0 for k in ["Fossil naphtha", "Bio naphtha", "Fischer-Tropsch", "Methanol"]})

    #hvc_methanol = -n.links_t.p1[hvc_links.index[hvc_links.index.str.contains('methanol')]].sum().sum() * timestep
    hvc_naphtha = -n.links_t.p1[hvc_links.index[hvc_links.index.str.contains('naphtha')]].sum().sum() * timestep

    share_bio, share_ft = share_bio_naphtha(n)
    avg_share_bio = share_bio.mean() if not share_bio.empty else 0.0
    avg_share_ft = share_ft.mean() if not share_ft.empty else 0.0
    share_fossil = max(0.0, 1 - avg_share_bio - avg_share_ft)

    # Convert from kt to Mt (using 1000 divisor)
    #hvc_methanol_mt = hvc_methanol / 1e3
    hvc_naphtha_mt = hvc_naphtha / 1e3

    return pd.Series({
        "Fossil naphtha": hvc_naphtha_mt * share_fossil,
        "Bio naphtha": hvc_naphtha_mt * avg_share_bio,
        "Fischer-Tropsch": hvc_naphtha_mt * avg_share_ft,
        #"From methanol": hvc_methanol_mt
    })

def get_h2_prod_eu(n):
    timestep = n.snapshot_weightings.iloc[0, 0]
    lhv_hydrogen = 33.33  # MWh/tH2 (lower heating value)

    tech_names = ['Electrolysis', 'SMR CC', 'SMR']
    tech_productions = {}

    for tech in tech_names:
        if tech == 'SMR':
            # Match "SMR" but exclude "SMR CC"
            tech_links = n.links[
                n.links.index.str.contains('SMR', case=False, na=False) &
                ~n.links.index.str.contains('SMR CC', case=False, na=False)
            ].index
        else:
            # Match normally
            tech_links = n.links[
                n.links.index.str.contains(tech, case=False, na=False)
            ].index

        if tech_links.empty:
            tech_productions[tech] = 0.0
            continue

        prod = -n.links_t.p1[tech_links].sum().sum() * timestep  # in MWh
        fuel_cells = n.links_t.p0.loc[:, n.links_t.p0.columns.str.contains('Fuel Cell')].sum().sum() * timestep

        tech_productions[tech] = (prod - fuel_cells) / lhv_hydrogen / 1e6  # convert to Mt H2

    return pd.Series(tech_productions)



def get_subtypes_for_tech(tech):
    tech = tech.lower()
    if tech == "methanol":
        return ["Green H2", "Grey H2"]
    elif tech == "hvc":
        return ["Fossil naphtha", "Bio naphtha", "Fischer-Tropsch"]#, "From methanol"]
    elif tech == "nh3" or tech == "ammonia":
        return ["Green H2", "Grey H2"]
    elif tech == "steel":
        return ['Green H2 EAF', 'Grey H2 EAF', 'CH4 EAF', 'BOF', 'BOF + TGR']
    elif tech == "cement":
        return ["Without CCS", "With CCS"]
    elif tech == "h2":
        return ["SMR", "SMR CC", "Electrolysis"]
    else:
        return ["Total"]

def get_tech_production_components(n, tech):
    # Dispatch to the correct internal logic for tech-specific subtype breakdowns
    if tech.lower() == "methanol":
        return get_methanol_prod_eu(n)
    elif tech.lower() == "steel":
        return get_steel_prod_eu(n)
    elif tech.lower() == 'cement':
        return get_cement_prod_eu(n)
    elif tech.lower() == "hvc":
        return get_hvc_prod_eu(n)
    elif tech.lower() == "nh3":
        return get_ammonia_prod_eu(n)
    elif tech.lower() == "h2":
        return get_h2_prod_eu(n)

def load_networks(scenarios, years, root_dir, res_dir):
    """
    Loads all networks into a dict keyed by (scenario, year).
    """
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks


def plot_total_eu_production_by_tech_reversed(
    networks,
    scenarios,
    nice_scenario_names,
    technologies,
    years,
    save_path="graphs/european_production_stacked.png"
):
    n_rows = len(technologies)
    n_cols = len(scenarios)
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(3.5 * n_cols, 2.5 * n_rows),
        sharex=True,
        #gridspec_kw={"wspace": 0.5, "width_ratios": [1, 1] + [0.9] * (n_cols - 2)}
    )
    
    if n_cols >= 3:
        ax2 = axes[0, 1]  # Column 2
        ax3 = axes[0, 2]  # Column 3
        x2 = ax2.get_position().x1
        x3 = ax3.get_position().x0
        x_line = (x2 + x3) / 2
    
        fig.lines.append(
            plt.Line2D([x_line, x_line], [0, 1], transform=fig.transFigure, color='black', linewidth=1)
        )
    
    # Add group titles above columns
    if n_cols >= 4:
        x_first_group = (axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2
        x_second_group = (axes[0, 2].get_position().x0 + axes[0, 3].get_position().x1) / 2
    
        fig.text(x_first_group, 0.95, "Gradual Deindustr", ha='center', fontsize=14)
        fig.text(x_second_group, 0.95, "Reindustr Strategy", ha='center', fontsize=14)


    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    bar_width = 7  # full-width bars, no space between years

    for i, tech in enumerate(technologies):  # tech index = row
        for j, scenario in enumerate(scenarios):  # scenario index = col
            ax = axes[i, j]
            tech_data = {subtype: [] for subtype in get_subtypes_for_tech(tech)}

            for year in years:
                net = networks.get((scenario, year))
                if net is None:
                    for subtype in tech_data:
                        tech_data[subtype].append(0.0)
                    continue

                prod_df = get_tech_production_components(net, tech)
                for subtype in tech_data:
                    tech_data[subtype].append(prod_df.get(subtype, 0.0))

            # Stack bars by subtype
            bottom = np.zeros(len(years))
            for subtype in tech_data:
                values = np.array(tech_data[subtype])
                ax.bar(
                    years,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=subtype,
                    color=feedstock_colors.get(subtype, 'gray'),
                    edgecolor='black',
                    linewidth=0.3,
                    align='center'
                )
                bottom += values

            if j == 0:
                custom_titles = {
                    "hvc": "Plastics",
                    "nh3": "Ammonia",
                    "h2": "Hydrogen"
                }
                title = custom_titles.get(tech.lower(), tech.title())
                ax.set_ylabel(f"{title} [Mt/yr]", fontsize=12)
            else:
                ax.set_yticklabels([])  # Remove y-axis tick labels
                ax.tick_params(axis='y', which='both', length=0)  # Optional: remove y-ticks


            # Column titles for each scenario
            if i == 0:
                ax.set_title(nice_scenario_names[scenario], fontsize=14)

            ax.set_xticks(years)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ymax = tech_ymax.get(tech.lower(), None)
            if ymax:
                ax.set_ylim(0, ymax)

            ax.grid(True, linestyle="--", alpha=0.3)

    # Define subtypes by technology for grouped legend
    legend_tech_groups = {
        "Steel": [
            "Green H2 EAF", "Grey H2 EAF", "CH4 EAF", "BOF", "BOF + TGR"
        ],
        "Cement": [
            "With CCS","Without CCS"
        ],
        "Ammonia": [
            "Green H2", "Grey H2"
            ],
        "Methanol":  [
            "Green H2", "Grey H2"
            ],
        
        "HVC": [
            "Bio naphtha","Fischer-Tropsch","Fossil naphtha" # "From methanol"
            ],
        "Hydrogen": [
            "Electrolysis","SMR CC", "SMR",
            ],

    }
    
    # Build custom legend handles with headers
    legend_handles = []
    for tech, subtypes in legend_tech_groups.items():
        # Add fake patch as section title
        legend_handles.append(mpatches.Patch(color='none', label=tech))
        for subtype in subtypes:
            if subtype in feedstock_colors:
                patch = mpatches.Patch(color=feedstock_colors[subtype], label=f"{subtype}")
                legend_handles.append(patch)
    
    # Add legend
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(1.03, 0.5),
        fontsize=11,
        #title="Feedstock by Sector",
        frameon=True,
        handlelength=1.5
    )
    

    #plt.suptitle("Total European Industrial Production by Technology and Feedstock", fontsize=15)
    #plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


scenarios = [
    "policy_eu_deindustrial",
    "policy_eu_deindustrial_flex",
    "policy_eu_regain",
    "policy_eu_regain_flex"
]

nice_scenario_names = {
    "policy_eu_deindustrial": "Standard Flex",
    "policy_eu_deindustrial_flex": "Extra Flex",
    "policy_eu_regain": "Standard Flex",
    "policy_eu_regain_flex": "Extra Flex"
}


tech_ymax = {
    "steel": 210,        
    "cement": 260,
    "nh3": 20,
    "methanol": 50,
    "hvc": 80,
    "h2": 150
}

# %%
# 1. Load all networks once
networks = load_networks(scenarios, years, root_dir, res_dir)

# %%
plot_total_eu_production_by_tech_reversed(
    networks=networks,
    scenarios=scenarios,
    nice_scenario_names=nice_scenario_names,
    technologies=technologies,
    years=years,
    save_path="graphs/european_production_stacked_flex.png"
)
