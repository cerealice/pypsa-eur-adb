# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 08:38:08 2025

@author: Dibella
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:58:40 2025

@author: Dibella
"""


# %% IMPORTS AND CONFIG

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd


def normed(s):
    return s / s.sum()

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
    steel_prod_green_h2_eaf = (steel_eaf * share_h2 * share_green).sum() * 1e3  # tons
    steel_prod_grey_h2_eaf = (steel_eaf * share_h2 * (1 - share_green)).sum() * 1e3
    steel_prod_ch4_eaf = (steel_eaf * (1 - share_h2)).sum() * 1e3
    steel_prod_bof_uncapt = steel_bof.sum() * share_uncaptured  * 1e3
    steel_prod_bof_capt = steel_bof.sum() * share_captured  * 1e3

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

    # Convert total production from kt to tons
    cement_total_mt = total_cement_mwh * 1e3

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
    total_mt = total_mwh / lhv_ammonia

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
    total_mt = total_mwh / lhv_methanol

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

    # Convert from kt toMt (using 1000 multiplier)
    #hvc_methanol_mt = hvc_methanol / 1e3
    hvc_naphtha_mt = hvc_naphtha * 1e3

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
        # Match links by technology
        tech_links = n.links[
            n.links.index.str.contains(tech, case=False, na=False)
        ].index

        # Exclude SMR CC from SMR to avoid double-counting
        if tech == 'SMR':
            tech_links = tech_links[~tech_links.str.contains("CC")]

        if tech_links.empty:
            tech_productions[tech] = 0.0
            continue

        prod = -n.links_t.p1[tech_links].sum().sum() * timestep  # in MWh
        tech_productions[tech] = prod / lhv_hydrogen

    return pd.Series(tech_productions)

def build_production_df(scenarios, years, parent_dir):
    """
    Builds a MultiIndex DataFrame with production per commodity per scenario and year.
    """
    index = pd.MultiIndex.from_product([scenarios, years], names=["scenario", "year"])
    production = pd.DataFrame(0,index=index, columns=commodities, dtype=float)

    for scenario in scenarios:
        for year in years:
            file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
            n = pypsa.Network(file_path)
            production.loc[(scenario, year), "steel"] = get_steel_prod_eu(n).sum()
            production.loc[(scenario, year), "cement"] = get_cement_prod_eu(n).sum()
            production.loc[(scenario, year), "ammonia"] = get_ammonia_prod_eu(n).sum()
            production.loc[(scenario, year), "methanol"] = get_methanol_prod_eu(n).sum()
            production.loc[(scenario, year), "HVC"] = get_hvc_prod_eu(n).sum()
            production.loc[(scenario, year), "H2"] = get_h2_prod_eu(n).sum()

    return production




def weighted_average_marginal_price(n, keyword, exclude_labels=None):
    mprice_cols = n.buses_t.marginal_price.columns[
        n.buses_t.marginal_price.columns.str.contains(keyword)
    ]
    if exclude_labels:
        for label in exclude_labels:
            mprice_cols = mprice_cols[~mprice_cols.str.contains(label)]
    mprice = n.buses_t.marginal_price.loc[:, mprice_cols].where(lambda df: df >= 0, 0)
    relevant_loads = mprice.columns.intersection(n.loads.index)
    mprice_loads = mprice[relevant_loads]
    loads_w_mprice = n.loads_t.p[relevant_loads]
    
    if keyword == "H2":
        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads_w_mprice = loads.T.groupby(level=0).sum().T
        mprice.columns = mprice.columns.str[:2]
        mprice_loads = mprice.T.groupby(level=0).sum().T     
        
    total_costs = mprice_loads * loads_w_mprice
    total_costs.columns = total_costs.columns.str[:2]
    total_costs = total_costs.T.groupby(level=0).sum().T.sum()
    loads_w_mprice.columns = loads_w_mprice.columns.str[:2]
    loads_w_mprice = loads_w_mprice.T.groupby(level=0).sum().T.sum()
    weighted_avg = total_costs / loads_w_mprice
    return weighted_avg.dropna().mean()


# %%

# === CONFIGURATION ===
scenarios = [
    "base_reg_deindustrial",
    "policy_reg_deindustrial",
    "base_reg_regain",
    "policy_reg_regain"
]
years = [2030, 2040, 2050]

scenario_colors = {
    "base_reg_deindustrial": "#464E47",
    "policy_reg_deindustrial": "#00B050",
    "base_reg_regain": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}

scenario_labels = {
    "base_reg_deindustrial": "No climate policy\nCurrent deindustr trend",
    "policy_reg_deindustrial": "Climate policy\nCurrent deindustr trend",
    "base_reg_regain": "No climate policy\nReindustrialize",
    "policy_reg_regain": "Climate policy\nReindustrialize"
}

scenario_colors = {
    "base_reg_deindustrial": "#464E47",
    "policy_reg_deindustrial": "#00B050",
    "base_reg_regain": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}

commodities = ["steel", "cement", "ammonia", "methanol", "HVC", "H2"]

lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t
naphtha_to_hvc = (2.31 * 12.47) * 1000
decay_emis_hvc = 0.2571 * naphtha_to_hvc / 1e3
lhv_hydrogen = 33.33 #MWh/t


# Store results
cost_data = {
    "Deindustrialization": pd.DataFrame(index=years, columns=commodities),
    "Reindustrialize": pd.DataFrame(index=years, columns=commodities)
}


# === INIT STORAGE ===
price_data = {commodity: pd.DataFrame(index=scenarios, columns=years) for commodity in commodities}

# === LOAD AND COMPUTE PRICES ===
max_value = 0
cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))

# --- Your price calculation loop ---
for scenario in scenarios:
    for year in years:
        file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0, 0]

        price_data["steel"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="steel") / 1e3
        price_data["cement"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="cement", exclude_labels=["process emissions"]) / 1e3
        price_data["ammonia"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="NH3") * lhv_ammonia

        price_data["methanol"].loc[scenario, year] = weighted_average_marginal_price(n, keyword='industry methanol') * lhv_methanol
        meth_price = n.buses_t.marginal_price.loc[
            :, n.buses_t.marginal_price.columns.str.contains('industry methanol')
        ].mean().iloc[0] * lhv_methanol
        co2_price = -n.global_constraints.loc["CO2Limit", "mu"]
        extra_methanol_cost = 0.248 * lhv_methanol * co2_price
        #price_data["methanol"].loc[scenario, year] = meth_price  # Keep or comment as needed

        hvc_price = weighted_average_marginal_price(n, keyword="HVC") / 1e3
        extra_hvc_cost = 0.2571 * 12.47 * co2_price
        price_data["HVC"].loc[scenario, year] = hvc_price
        price_data["H2"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="H2", exclude_labels=["pipeline"]) * lhv_hydrogen

        max_value = max(max_value, price_data["HVC"].loc[scenario, year])


# --- Build production data ---
production = build_production_df(scenarios, years, parent_dir)

# --- Compute cost deltas ---
for year in years:
    for commodity in commodities:
        price_delta_deindustrial = (
            price_data[commodity].loc["policy_reg_deindustrial", year] -
            price_data[commodity].loc["base_reg_deindustrial", year]
        )
        price_delta_regain = (
            price_data[commodity].loc["policy_reg_regain", year] -
            price_data[commodity].loc["base_reg_regain", year]
        )

        # Use production from base_reg_deindustrial and base_reg_regain respectively
        prod_deindustrial = production.loc[("base_reg_deindustrial", year), commodity]
        prod_regain = production.loc[("base_reg_regain", year), commodity]

        cost_data["Deindustrialization"].loc[year, commodity] = price_delta_deindustrial * prod_deindustrial / 1e9 # b€
        cost_data["Reindustrialize"].loc[year, commodity] = price_delta_regain * prod_regain / 1e9


# %%

n_commodities = len(commodities)
fig, axes = plt.subplots(1, n_commodities, figsize=(5 * n_commodities, 6), sharey=False)

for idx, (ax, commodity) in enumerate(zip(axes, commodities)):
    deindustrial = cost_data["Deindustrialization"][commodity].astype(float)
    reindustrial = cost_data["Reindustrialize"][commodity].astype(float)

    bars_deindustrial = ax.bar(
        years,
        deindustrial,
        width=5,
        label="Current deindustr trend",
        color=scenario_colors["policy_reg_deindustrial"]
    )
    bars_reindustrial = ax.bar(
        years,
        reindustrial - deindustrial,
        bottom=deindustrial,
        width=5,
        label="Reindustrialize (Difference)",
        color=scenario_colors["policy_reg_regain"]
    )

    # Add value labels on bars
    for bar in bars_deindustrial:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f"{height:.1f}",
                ha='center', va='center',
                fontsize=8, color='white', weight='bold'
            )

    for bar in bars_reindustrial:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{height:.1f}",
                ha='center', va='center',
                fontsize=8, color='white', weight='bold'
            )

    custom_titles = {
        "hvc": "Plastics",
        "h2": "Hydrogen"
    }
    
    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(f"{title}", fontsize=14)
    ax.set_xticks(years)
    ax.set_ylabel("Required subsidies [b€/yr]" if idx == 0 else "")
    ax.grid(True, linestyle='--')

    # Legend only on last subplot
    if idx == n_commodities - 1:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("./graphs/additional_costs_per_commodity_difference.png", dpi=300)
plt.show()


# %%


years = np.array([2030, 2040, 2050])
n_commodities = len(commodities)
fig, axes = plt.subplots(1, n_commodities, figsize=(6 * n_commodities, 5), sharey=False)

if n_commodities == 1:
    axes = [axes]  # Ensure iterable if only one commodity

for idx, (ax, commodity) in enumerate(zip(axes, commodities)):
    deindustrial = cost_data["Deindustrialization"][commodity].astype(float)
    reindustrial = cost_data["Reindustrialize"][commodity].astype(float)
    
    # Select only years 2030, 2040, 2050
    deindustrial_values = np.array([deindustrial[year] for year in years])
    reindustrial_values = np.array([reindustrial[year] for year in years])

    
    # Linear interpolation points
    x_interp = np.linspace(2030, 2050, 300)
    deindustrial_interp = np.interp(x_interp, years, deindustrial_values)
    reindustrial_interp = np.interp(x_interp, years, reindustrial_values)
    
    # Plot deindustrialization line (base trend)
    ax.plot(x_interp, deindustrial_interp, label="Deindustrialization trend", 
            color=scenario_colors["policy_reg_deindustrial"], linewidth=1)
    
    # Fill area below deindustrialization line (visual anchor)
    ax.fill_between(x_interp, 0, deindustrial_interp, 
                    color=scenario_colors["policy_reg_deindustrial"], alpha=0.5)
    
    # Fill area between deindustrialization and reindustrialization
    ax.fill_between(x_interp, deindustrial_interp, reindustrial_interp,
                    where=reindustrial_interp >= deindustrial_interp,
                    color=scenario_colors["policy_reg_regain"], alpha=0.5,
                    label="Reindustrialize (additional subsidies)")
    
    # Customize plot
    custom_titles = {"hvc": "Plastics", "h2": "Hydrogen"}
    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(f"{title}", fontsize=14)
    if idx == 0:
        ax.set_ylabel("Subsidies [b€/yr]")
    
    ax.set_xlim(2030, 2050)
    ax.grid(True, linestyle='--')
    
    # Legend only on last subplot
    if idx == n_commodities - 1:
        ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("./graphs/area_plot_subsidies_per_commodity.png", dpi=300)
plt.show()

# %%


selected_years = [2030, 2040, 2050]
n_commodities = len(commodities)

fig, axes = plt.subplots(1, n_commodities, figsize=(6 * n_commodities, 6), sharey=False)

if n_commodities == 1:
    axes = [axes]  # Ensure iterable for a single commodity

# First: Compute max y-limit from non-Plastics commodities
custom_titles = {"hvc": "Plastics", "h2": "Hydrogen"}

max_y = 0
for commodity in commodities:
    title = custom_titles.get(commodity.lower(), commodity.title())
    if title == "Plastics":
        continue
    deindustrial = cost_data["Deindustrialization"][commodity].astype(float)
    reindustrial = cost_data["Reindustrialize"][commodity].astype(float)
    deindustrial_total = sum([deindustrial[year] for year in selected_years]) * 10
    reindustrial_total = sum([reindustrial[year] for year in selected_years]) * 10
    max_y = max(max_y, deindustrial_total, reindustrial_total)

# Second: Plot with forced shared y-axis limit for all
for idx, (ax, commodity) in enumerate(zip(axes, commodities)):
    deindustrial = cost_data["Deindustrialization"][commodity].astype(float)
    reindustrial = cost_data["Reindustrialize"][commodity].astype(float)
    
    deindustrial_total = sum([deindustrial[year] for year in selected_years]) * 10
    reindustrial_total = sum([reindustrial[year] for year in selected_years]) * 10
    
    bar_positions = [0, 1]
    bar_heights = [deindustrial_total, reindustrial_total]
    
    bar_colors = [
        scenario_colors["policy_reg_deindustrial"],
        scenario_colors["policy_reg_regain"]
    ]
    
    bars = ax.bar(bar_positions, bar_heights, color=bar_colors, width=0.6, zorder=2)
    
    # Average per year inside bars (only if within limits)
    for bar, total in zip(bars, bar_heights):
        avg_per_year = total / 30
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(bar.get_height() / 2, max_y * 0.9),  # Keep inside visible area
            f"{avg_per_year:.1f} b€/yr",
            ha='center', va='center',
            fontsize=10, color='white', weight='bold'
        )
    
    # Show total value labels
    for bar, total in zip(bars, bar_heights):
        if total <= max_y * 1.1:
            # Normal position: just above bar
            label_y = total + max_y * 0.02
        else:
            # Overflow: show above y-limit
            label_y = max_y * 1.03
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            f"{total:.1f}",
            ha='center', va='bottom',
            fontsize=10, weight='bold'
        )

    
    # Set consistent y-limit for all
    ax.set_ylim(0, max_y * 1.1)  # Small padding
    
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(["Deindustrialize", "Reindustrialize"], rotation=15)
    
    title = custom_titles.get(commodity.lower(), commodity.title())
    ax.set_title(title, fontsize=14)
    
    if idx == 0:
        ax.set_ylabel("Total Subsidies Over Pathway [b€]")
    
    ax.grid(axis='y', linestyle='--', zorder=1)

plt.tight_layout()
plt.savefig("./graphs/total_subsidies_comparison_same_yaxis_plastics_above.png", dpi=300)
plt.show()

