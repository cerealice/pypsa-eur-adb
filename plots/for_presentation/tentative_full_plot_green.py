# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:06:39 2025

@author: Dibella
"""


import pypsa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
import geopandas as gpd

# CONFIG
years = [2030, 2040, 2050]
technologies = ['steel', 'cement', 'NH3', 'methanol', 'HVC']
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t

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
    nuts3 = gpd.read_file('../resources/base_eu_regain/nuts3_shapes.geojson').set_index("index")
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


def get_green_share_steel(network, region):
    timestep = network.snapshot_weightings.iloc[0, 0]
    
    # Get all steel-related links excluding 'heat'
    steel_links = network.links[
        (network.links['bus1'].str.contains('steel', case=False, na=False)) &
        (~network.links['bus1'].str.contains('heat', case=False, na=False))
    ].copy()
    steel_links["country"] = steel_links.index.str[:2]

    # Total steel production by country
    steel_prod = -network.links_t.p1.loc[:, steel_links.index].sum() * timestep
    steel_prod.index = steel_prod.index.str[:2]
    steel_prod = steel_prod.groupby(steel_prod.index).sum()

    # Filter countries in the region
    countries_in_region = [c for c in steel_prod.index if c in group_countries[region]]
    steel_prod = steel_prod.loc[countries_in_region]

    if steel_prod.sum() == 0:
        return 0.0  # Avoid division by zero

    # EAF / BOF link masks
    eaf_links = steel_links[steel_links.index.str.contains('EAF')].index
    bof_links = steel_links[steel_links.index.str.contains('BOF')].index

    # EAF and BOF by country
    steel_eaf = -network.links_t.p1.loc[:, eaf_links].sum() * timestep
    steel_bof = -network.links_t.p1.loc[:, bof_links].sum() * timestep

    steel_eaf.index = steel_eaf.index.str[:2]
    steel_bof.index = steel_bof.index.str[:2]

    steel_eaf = steel_eaf.groupby(steel_eaf.index).sum()
    steel_bof = steel_bof.groupby(steel_bof.index).sum()

    # Share of H2 in DRI
    dri_ch4 = -network.links_t.p1.filter(like='CH4 to syn gas DRI', axis=1).sum() * timestep
    dri_h2 = -network.links_t.p1.filter(like='H2 to syn gas DRI', axis=1).sum() * timestep
    share_h2 = dri_h2.sum() / (dri_ch4.sum() + dri_h2.sum()) if (dri_ch4.sum() + dri_h2.sum()) > 0 else 0

    # Share of green H2 by country
    share_green = share_green_h2(network)

    # Per-country green steel estimate
    green_steel_by_country = {}
    for country in countries_in_region:
        eaf = steel_eaf.get(country, 0)
        green_h2_share = share_green.get(country, 0)
        green_steel = eaf * share_h2 * green_h2_share
        green_steel_by_country[country] = green_steel

    # Weighted average
    total_green = sum(green_steel_by_country[c] for c in countries_in_region)
    total_prod = steel_prod.sum()

    return float(total_green / total_prod) if total_prod > 0 else 0.0

def get_green_share_cement(network, region):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Get cement production links excluding process emissions
    cement_links = network.links[
        (network.links['bus1'].str.contains('cement', case=False, na=False)) &
        (~network.links['bus1'].str.contains('process emissions', case=False, na=False))
    ].copy()
    cement_links["country"] = cement_links.index.str[:2]

    # Total cement production
    cement_prod = -network.links_t.p1.loc[:, cement_links.index].sum() * timestep
    cement_prod.index = cement_prod.index.str[:2]
    cement_prod = cement_prod.groupby(cement_prod.index).sum()

    countries_in_region = [c for c in cement_prod.index if c in group_countries[region]]
    cement_prod = cement_prod.loc[countries_in_region]

    if cement_prod.sum() == 0:
        return 0.0

    # Emissions and CCS flows
    cement_not_captured = -network.links_t.p1.filter(
        like='cement process emis to atmosphere', axis=1).sum() * timestep
    cement_ccs = -network.links_t.p2.filter(like='cement TGR', axis=1).sum() * timestep

    cement_not_captured.index = cement_not_captured.index.str[:2]
    cement_ccs.index = cement_ccs.index.str[:2]

    cement_not_captured = cement_not_captured.groupby(cement_not_captured.index).sum()
    cement_ccs = cement_ccs.groupby(cement_ccs.index).sum()

    # Filter to regional countries
    cement_not_captured = cement_not_captured.loc[countries_in_region].fillna(0)
    cement_ccs = cement_ccs.loc[countries_in_region].fillna(0)

    # Calculate green share: CCS / (CCS + uncaptured)
    green_cement_by_country = cement_ccs / (cement_ccs + cement_not_captured)
    green_cement_by_country = green_cement_by_country.fillna(0)

    # Weighted average over regional cement production
    weighted_sum = sum(green_cement_by_country[c] * cement_prod[c] for c in countries_in_region)
    total_prod = cement_prod.sum()

    return float(weighted_sum / total_prod) if total_prod > 0 else 0.0


def get_green_share_ammonia(network, region):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Identify ammonia production links
    ammonia_links = network.links[network.links['bus1'].str.contains('NH3', case=False, na=False)].copy()
    ammonia_links["country"] = ammonia_links.index.str[:2]

    # Total ammonia production (Mt)
    ammonia_prod = -network.links_t.p1.loc[:, ammonia_links.index].sum() * timestep / lhv_ammonia / 1e6
    ammonia_prod.index = ammonia_prod.index.str[:2]
    ammonia_prod = ammonia_prod.groupby(ammonia_prod.index).sum()

    # Restrict to countries in the selected region
    countries_in_region = [c for c in ammonia_prod.index if c in group_countries[region]]
    ammonia_prod = ammonia_prod.loc[countries_in_region]

    if ammonia_prod.sum() == 0:
        return 0.0

    # Get green H2 share per country
    share_green = share_green_h2(network).fillna(0)
    share_green = share_green.loc[countries_in_region]

    # Compute weighted average green share
    weighted_sum = sum(share_green[c] * ammonia_prod[c] for c in countries_in_region)
    total_prod = ammonia_prod.sum()

    return float(weighted_sum / total_prod) if total_prod > 0 else 0.0

def get_green_share_methanol(network, region):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Identify methanol production links
    methanol_links = network.links[network.links.index.str.contains('methanolisation', case=False, na=False)].copy()
    methanol_links["country"] = methanol_links.index.str[:2]

    # Total methanol production (Mt)
    methanol_prod = -network.links_t.p1.loc[:, methanol_links.index].sum() * timestep / lhv_methanol / 1e6
    methanol_prod.index = methanol_prod.index.str[:2]
    methanol_prod = methanol_prod.groupby(methanol_prod.index).sum()

    # Restrict to countries in the selected region
    countries_in_region = [c for c in methanol_prod.index if c in group_countries[region]]
    methanol_prod = methanol_prod.loc[countries_in_region]

    if methanol_prod.sum() == 0:
        return 0.0

    # Get green H2 share per country
    share_green = share_green_h2(network).fillna(0)
    share_green = share_green.loc[countries_in_region]

    # Compute weighted average green share
    weighted_sum = sum(share_green[c] * methanol_prod[c] for c in countries_in_region)
    total_prod = methanol_prod.sum()

    return float(weighted_sum / total_prod) if total_prod > 0 else 0.0

def get_green_share_hvc(network, region):
    timestep = network.snapshot_weightings.iloc[0, 0]

    # Get all HVC production links
    hvc_links = network.links[network.links['bus1'].str.contains('HVC', case=False, na=False)].copy()
    hvc_links["country"] = hvc_links.index.str[:2]

    # Total HVC production
    hvc_prod = -network.links_t.p1.loc[:, hvc_links.index].sum() * timestep
    hvc_prod.index = hvc_prod.index.str[:2]
    hvc_prod = hvc_prod.groupby(hvc_prod.index).sum()

    # Restrict to countries in region
    countries_in_region = [c for c in hvc_prod.index if c in group_countries[region]]
    hvc_prod = hvc_prod.loc[countries_in_region]

    if hvc_prod.sum() == 0:
        return 0.0

    # Extract sub-tech flows
    #methanol_links = hvc_links[hvc_links.index.str.contains('methanol')].index
    naphtha_links = hvc_links[hvc_links.index.str.contains('naphtha')].index

    #hvc_methanol = -network.links_t.p1.loc[:, methanol_links].sum() * timestep
    hvc_naphtha = -network.links_t.p1.loc[:, naphtha_links].sum() * timestep

    #hvc_methanol.index = hvc_methanol.index.str[:2]
    hvc_naphtha.index = hvc_naphtha.index.str[:2]

    #hvc_methanol = hvc_methanol.groupby(hvc_methanol.index).sum().loc[countries_in_region]
    hvc_naphtha = hvc_naphtha.groupby(hvc_naphtha.index).sum().loc[countries_in_region]

    # Get shares of green naphtha (bio + FT) and grey (fossil)
    share_bio, share_ft = share_bio_naphtha(network)
    share_bio = share_bio.loc[countries_in_region].fillna(0)
    share_ft = share_ft.loc[countries_in_region].fillna(0)

    # Methanol can be green depending on H2
    #share_green_methanol = share_green_h2(network).loc[countries_in_region].fillna(0)

    # Calculate green portions
    green_naphtha = hvc_naphtha * (share_bio + share_ft)
    #green_methanol = hvc_methanol * share_green_methanol

    # Total HVC from naphtha + methanol
    total_hvc = hvc_naphtha #+ hvc_methanol

    green_total = green_naphtha #+ green_methanol

    return float((green_total.sum() / total_hvc.sum()) if total_hvc.sum() > 0 else 0.0)


def get_green_share_by_tech(tech, network, region):
    if tech.lower() == 'steel':
        return get_green_share_steel(network, region)
    elif tech.lower() == 'cement':
        return get_green_share_cement(network, region)
    elif tech.lower() == 'nh3':
        return get_green_share_ammonia(network, region)
    elif tech.lower() == 'methanol':
        return get_green_share_methanol(network, region)
    elif tech.lower() == 'hvc':
        return get_green_share_hvc(network, region)

def plot_green_share_by_region(
    scenarios,
    nice_scenario_names,
    scenario_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    save_path="full_graphs/green_share.png"
):
    region_names = list(group_countries.keys())

    # Load all networks
    networks = {}
    for year in years:
        for scenario in scenarios:
            key = (scenario, year)
            path = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
            networks[key] = pypsa.Network(path)

    n_rows = len(region_names)
    n_cols = len(technologies)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, region in enumerate(region_names):
        for j, tech in enumerate(technologies):
            ax = axes[i, j]

            for scenario in scenarios:
                green_shares = []
                annotations = []

                for year in years:
                    net = networks[(scenario, year)]

                    # Check if production is significant
                    region_percents, _ = get_production_percentages(net, tech)
                    prod_percent = region_percents.get(region, 0.0)

                    if prod_percent < 1.0:
                        green_shares.append(np.nan)
                        annotations.append("")
                        continue

                    green_share = get_green_share_by_tech(tech, net, region)
                    green_share_pct = round(green_share * 100, 1)

                    green_shares.append(green_share_pct)
                    annotations.append(f"{green_share_pct:.1f}%")

                # Plot line
                ax.plot(
                    years,
                    green_shares,
                    marker='o',
                    label=nice_scenario_names.get(scenario, scenario),
                    color=scenario_colors.get(scenario, f"C{scenarios.index(scenario)}")
                )

                # Annotate points with green share %
                for x, y, label in zip(years, green_shares, annotations):
                    if not np.isnan(y):
                        ax.annotate(
                            label,
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 6),
                            ha='center',
                            fontsize=8,
                            color='gray'
                        )

            ax.set_ylim(0, 100)
            if i == n_rows - 1:
                ax.set_xlabel("Year")
            if j == 0:
                ax.set_ylabel(f"{region}\nGreen share (%)", fontsize=10)
                
            # Title for top row
            custom_titles = {
                "hvc": "Plastics",
                "nh3": "Ammonia"
            }
            
            if i == 0:
                title = custom_titles.get(tech.lower(), tech.title())
                ax.set_title(f"{title}", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.3)

    fig.legend(
        handles=[mlines.Line2D([], [], color=scenario_colors[s], label=nice_scenario_names[s]) for s in scenarios],
        loc='center right',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        frameon=True
    )

    plt.suptitle("Green Share of Industrial Production by Region, Technology, and Scenario", fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# %% 
# CLIMATE POLICY CASE
scenarios = ["base_eu_regain", "policy_eu_regain"]
nice_scenario_names = {
    'base_eu_regain': 'No climate policy',
    'policy_eu_regain': 'Climate policy'
}
scenario_colors = {
    'base_eu_regain': '#D95F02',
    'policy_eu_regain': '#1B9E77'
}

plot_green_share_by_region(
    scenarios,
    nice_scenario_names,
    scenario_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    save_path="full_graphs/green_share_climate.png"
)
