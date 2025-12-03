# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 10:46:18 2025

@author: Dibella
"""


# %% IMPORTS AND CONFIG

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd

def normed(s):
    return s / s.sum()

# === LOADER ===
def load_networks(scenarios, years, root_dir, res_dir):
    """
    Load all networks once into a dict keyed by (scenario, year).
    """
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks


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
            file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
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
    elif keyword == 'methanol':
        loads_links = n.loads[n.loads.index.str.endswith('methanol')]
        loads = n.loads_t.p.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads_w_mprice = loads.T.groupby(level=0).sum().T
        mprice.columns = mprice.columns.str[:2]
        mprice_loads = mprice.T.groupby(level=0).sum().T
        
    total_costs = mprice_loads * loads_w_mprice
    total = total_costs.sum().sum() / loads_w_mprice.sum().sum()
    return total


# %%

# === CONFIGURATION ===
scenarios = [
    "base_reg_deindustrial",
    "policy_reg_deindustrial",
    "base_reg_regain",
    "policy_reg_regain",
    "import_policy_reg_deindustrial",
    "import_policy_reg_regain"
]
years = [2030, 2040, 2050]

scenario_colors = {
    "base_reg_deindustrial": "#813746",
    "policy_reg_deindustrial": "#FC814A",
    "base_reg_regain": "#6D8088",
    "policy_reg_regain": "#28C76F",
    "import_policy_reg_deindustrial": "#A6A14E",
    "import_policy_reg_regain": "#6BCDC9"
}

scenario_labels = {
    "base_reg_deindustrial": "No climate policy\nCurrent deindustr trend",
    "policy_reg_deindustrial": "Climate policy\nCurrent deindustr trend",
    "base_reg_regain": "No climate policy\nReindustrialize",
    "policy_reg_regain": "Climate policy\nReindustrialize",
    "import_policy_reg_deindustrial": "Import policy\nCurrent deindustr trend",
    "import_policy_reg_regain": "Import policy\nReindustrialize"
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
    "Reindustrialize": pd.DataFrame(index=years, columns=commodities),
    "Import_Deindustrialization": pd.DataFrame(index=years, columns=commodities),
    "Import_Reindustrialize": pd.DataFrame(index=years, columns=commodities)
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
        file_path = os.path.join(parent_dir, "results_september_new", scenario, "networks", f"base_s_39___{year}.nc")
        n = pypsa.Network(file_path)
        timestep = n.snapshot_weightings.iloc[0, 0]

        price_data["steel"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="steel") / 1e3
        price_data["cement"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="cement", exclude_labels=["process emissions"]) / 1e3
        price_data["ammonia"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="NH3") * lhv_ammonia

        price_data["methanol"].loc[scenario, year] = weighted_average_marginal_price(n, keyword='methanol') * lhv_methanol
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
        # Original scenarios
        price_delta_deindustrial = (
            price_data[commodity].loc["policy_reg_deindustrial", year] -
            price_data[commodity].loc["base_reg_deindustrial", year]
        )
        price_delta_regain = (
            price_data[commodity].loc["policy_reg_regain", year] -
            price_data[commodity].loc["base_reg_regain", year]
        )
        
        # New import scenarios
        price_delta_import_deindustrial = (
            price_data[commodity].loc["import_policy_reg_deindustrial", year] -
            price_data[commodity].loc["base_reg_deindustrial", year]
        )
        price_delta_import_regain = (
            price_data[commodity].loc["import_policy_reg_regain", year] -
            price_data[commodity].loc["base_reg_regain", year]
        )

        # Use production from base scenarios
        prod_deindustrial = production.loc[("policy_reg_deindustrial", year), commodity]
        prod_regain = production.loc[("policy_reg_regain", year), commodity]
        prod_deindustrial_import = production.loc[("import_policy_reg_deindustrial", year), commodity]
        prod_regain_import = production.loc[("import_policy_reg_regain", year), commodity]

        cost_data["Deindustrialization"].loc[year, commodity] = price_delta_deindustrial * prod_deindustrial / 1e9 # b€
        cost_data["Reindustrialize"].loc[year, commodity] = price_delta_regain * prod_regain / 1e9
        cost_data["Import_Deindustrialization"].loc[year, commodity] = price_delta_import_deindustrial * prod_deindustrial_import / 1e9
        cost_data["Import_Reindustrialize"].loc[year, commodity] = price_delta_import_regain * prod_regain_import / 1e9


# %%
selected_years = [2030, 2040, 2050]
# Exclude Hydrogen
plot_commodities = [c for c in commodities if c.lower() != "h2"]
n_commodities = len(plot_commodities)

fig, axes = plt.subplots(1, n_commodities, figsize=(4 * n_commodities, 6), sharey=False)

custom_titles = {"hvc": "Plastics"}

# Compute max y-limit across all commodities
def compute_max_y(commodity_list):
    max_y = 0
    for commodity in commodity_list:
        # Always consider all four scenarios now
        scenarios = ["Deindustrialization", "Import_Deindustrialization",
                     "Reindustrialize", "Import_Reindustrialize"]
        for scenario_type in scenarios:
            data = cost_data[scenario_type][commodity].fillna(0).astype(float)
            total = sum([data[year] for year in selected_years]) * 10/30
            max_y = max(max_y, total)
    return max_y

max_y = 60#compute_max_y(plot_commodities)

# Plotting function
def plot_commodity(ax, commodity, max_y_limit):
    # Read all four scenarios
    deindustrial = cost_data["Deindustrialization"][commodity].fillna(0).astype(float)
    import_deindustrial = cost_data["Import_Deindustrialization"][commodity].fillna(0).astype(float)
    reindustrial = cost_data["Reindustrialize"][commodity].fillna(0).astype(float)
    import_reindustrial = cost_data["Import_Reindustrialize"][commodity].fillna(0).astype(float)

    # Total over pathway
    totals = [
        sum([deindustrial[year] for year in selected_years]) * 10/30,
        sum([import_deindustrial[year] for year in selected_years]) * 10/30,
        sum([reindustrial[year] for year in selected_years]) * 10/30,
        sum([import_reindustrial[year] for year in selected_years]) * 10/30
    ]

    bar_colors = [
        scenario_colors["policy_reg_deindustrial"],
        scenario_colors["import_policy_reg_deindustrial"],
        scenario_colors["policy_reg_regain"],
        scenario_colors["import_policy_reg_regain"]
    ]

    x_labels = ["Deindustr", "Import\nDeindustr", "Reindustr", "Import\nReindustr"]

    bars = ax.bar(range(4), totals, color=bar_colors, width=0.6, zorder=2)

    # Add total values above bars
    for bar, total in zip(bars, totals):
        if total > 0:
            label_y = total + max_y_limit * 0.02 if total <= max_y_limit * 1.1 else max_y_limit * 1.03
            ax.text(bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{total:.1f}",
                    ha='center', va='bottom',
                    fontsize=10, weight='bold')

    ax.set_ylim(0, max_y_limit * 1.1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(x_labels, rotation=0, fontsize=11)
    ax.set_title(custom_titles.get(commodity.lower(), commodity.title()), fontsize=14)
    ax.grid(axis='y', linestyle='--', zorder=1)

# Plot all commodities in a single row
for col_idx, commodity in enumerate(plot_commodities):
    ax = axes[col_idx] if n_commodities > 1 else axes
    plot_commodity(ax, commodity, max_y)

# Add y-label to the first subplot
if n_commodities > 0:
    axes[0].set_ylabel("Average subsidies per year [bnEUR/a]", fontsize=13)

plt.tight_layout()
plt.savefig("./graphs/total_subsidies_single_row.png", dpi=300)
plt.show()


# %%
import numpy as np

scenarios = [
    "policy_reg_deindustrial",
    "policy_reg_regain",
    "import_policy_reg_deindustrial",
    "import_policy_reg_regain"
]

root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_september_new/"
networks = load_networks(scenarios, years, root_dir, res_dir)
# %%
# --- Prepare dataframes to store prices and emissions ---
carbon_prices = pd.DataFrame(index=years, columns=scenarios, dtype=float)
carbon_caps = pd.DataFrame(index=years, columns=scenarios, dtype=float)

for scenario in scenarios:
    for year in years:
        n = networks[(scenario, year)]
        carbon_prices.loc[year, scenario] = -n.global_constraints.loc['CO2Limit', 'mu']        # EUR/tCO2
        carbon_caps.loc[year, scenario] = n.global_constraints.loc['CO2Limit', 'constant']     # tCO2/yr

# --- Define years to calculate revenue for ---
interp_years = np.arange(2025, 2056)  # 2025 to 2055 inclusive

# --- Interpolate prices and caps linearly ---
carbon_prices_interp = carbon_prices.apply(lambda col: np.interp(interp_years, years, col))
carbon_caps_interp = carbon_caps.apply(lambda col: np.interp(interp_years, years, col))

# --- Calculate revenue per year (bnEUR/yr) ---
carbon_revenues_interp = (carbon_prices_interp * carbon_caps_interp / 1e9)

average_per_year_revenues = carbon_revenues_interp.sum() / len(interp_years)

# === Calculate average subsidies per year across all commodities ===
average_subsidies = {}

for scenario in ["Deindustrialization", "Import_Deindustrialization",
                 "Reindustrialize", "Import_Reindustrialize"]:
    total_subsidy = 0
    for commodity in plot_commodities:  # exclude H2 already
        data = cost_data[scenario][commodity].fillna(0).astype(float)
        # Total over selected years, then averaged to "per year"
        total_subsidy += sum([data[year] for year in selected_years]) * 10/30
    
    average_subsidies[scenario] = total_subsidy

# Convert to DataFrame
df_average_subsidies = pd.DataFrame.from_dict(
    average_subsidies, orient="index", columns=["Average subsidies per year [bnEUR/a]"]
)

# --- Convert average_per_year_revenues to a DataFrame ---
df_average_revenues = pd.DataFrame(
    average_per_year_revenues,
    index=average_per_year_revenues.index,
    columns=["Average carbon revenues per year [bnEUR/a]"]
)

# --- Remove 'base' scenarios if present ---
df_average_revenues = df_average_revenues.loc[~df_average_revenues.index.str.startswith("base")]

# --- Combine subsidies and revenues into a single DataFrame ---
df_combined = df_average_subsidies.join(df_average_revenues, how="inner")

# Display
df_combined




# %%


"""
# %%

import seaborn as sns

# --- Prepare data for heatmap ---
scenario_order = ["Deindustrialization", "Reindustrialize", "Import_Deindustrialization", "Import_Reindustrialize"]
commodity_order = [c.title() if c not in ["HVC", "H2"] else {"HVC": "Plastics", "H2": "Hydrogen"}[c] for c in commodities]

# Build DataFrame: rows = commodities, columns = scenarios
heatmap_data = pd.DataFrame(index=commodity_order, columns=scenario_order, dtype=float)
annot_data = pd.DataFrame(index=commodity_order, columns=scenario_order, dtype=object)

# Fill data
for commodity in commodities:
    display_name = {"HVC": "Plastics", "H2": "Hydrogen"}.get(commodity, commodity.title())

    for scenario_key, scenario_name in zip(
        ["Deindustrialization", "Reindustrialize", "Import_Deindustrialization", "Import_Reindustrialize"],
        scenario_order
    ):
        if display_name in ["Cement", "Plastics"] and "Import" in scenario_name:
            # Skip import scenarios for Cement and Plastics
            heatmap_data.loc[display_name, scenario_name] = None
            annot_data.loc[display_name, scenario_name] = ""
            continue

        total = sum([cost_data[scenario_key][commodity].astype(float)[year] for year in selected_years]) * 10
        avg_per_year = total / 30

        heatmap_data.loc[display_name, scenario_name] = total
        annot_data.loc[display_name, scenario_name] = f"{avg_per_year:.1f}"

# Plot heatmap
plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    heatmap_data,
    annot=annot_data,
    fmt="",
    cmap="YlOrRd",
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={"label": "Total subsidies 2030–2050 [bnEUR]"},
    square=True,
    annot_kws={"fontsize": 10, "weight": "bold", "color": "black"}
)

plt.title("Total Subsidies by Scenario and Commodity\n(White text = Avg per year [bnEUR/a])", fontsize=14, weight='bold')
plt.xticks(rotation=20, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig("./graphs/total_subsidies_comparison_heatmap.png", dpi=300)
plt.show()


# %%

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# --- Prepare data for heatmap ---
scenario_order = ["Deindustrialization",  "Import_Deindustrialization", "Reindustrialize","Import_Reindustrialize"]
commodity_order = [c.title() if c not in ["HVC", "H2"] else {"HVC": "Plastics", "H2": "Hydrogen"}[c] for c in commodities]

heatmap_data = pd.DataFrame(index=commodity_order, columns=scenario_order, dtype=float)
annot_data = pd.DataFrame(index=commodity_order, columns=scenario_order, dtype=object)

for commodity in commodities:
    display_name = {"HVC": "Plastics", "H2": "Hydrogen"}.get(commodity, commodity.title())

    for scenario_key, scenario_name in zip(
        ["Deindustrialization",  "Import_Deindustrialization", "Reindustrialize", "Import_Reindustrialize"],
        scenario_order
    ):
        if display_name in ["Cement", "Plastics"] and "Import" in scenario_name:
            heatmap_data.loc[display_name, scenario_name] = None
            annot_data.loc[display_name, scenario_name] = ""
            continue

        total = sum([cost_data[scenario_key][commodity].astype(float)[year] for year in selected_years]) * 10
        avg_per_year = total / 30

        heatmap_data.loc[display_name, scenario_name] = total
        annot_data.loc[display_name, scenario_name] = f"{avg_per_year:.1f}"

# === Create custom colormap ===

# Flatten and drop NaNs to compute boundaries
all_vals = heatmap_data.values.flatten()
all_vals = all_vals[~np.isnan(all_vals)]

vmin = 0
v2 = np.percentile(all_vals, 25)
v3 = np.percentile(all_vals, 50)
v4 = np.percentile(all_vals, 75)
vmax = all_vals.max()

v2 = 250
v3 = 500
v4 = 1500

boundaries = [vmin,v2, v3, v4, vmax]
colors = [ "lightblue", "white", "pink", "red", "darkred"]

# Build colormap + norm
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(
    np.linspace(0, 1, len(boundaries)),
    colors
)))
norm = BoundaryNorm(boundaries, ncolors=custom_cmap.N, clip=True)

# === Plot heatmap ===
plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    heatmap_data,
    annot=annot_data,
    fmt="",
    cmap=custom_cmap,
    norm=norm,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={"label": "Total subsidies 2030–2050 [bnEUR]"},
    square=True,
    annot_kws={"fontsize": 10, "weight": "bold", "color": "black"}
)

# Replace x-axis tick labels with shorter, prettier names
ax.set_xticklabels(["Deindust",  "Deindust\nImport  ","Reindust", "Reindust\nImport  "], rotation=0, ha='center', fontsize=11)


plt.title("Black number = Avg per year [bnEUR/a]", fontsize=14, weight='bold')
plt.xticks(rotation=20, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

fig = plt.gcf()
fig.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.12)
plt.savefig("./graphs/total_subsidies_comparison_heatmap_custom_colormap.png", dpi=300)
plt.show()

# %%

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import pandas as pd

# --- Prepare data for heatmap ---
scenario_order = ["Deindustrialization",  "Import_Deindustrialization", "Reindustrialize","Import_Reindustrialize"]
commodity_order = [
    c.title() if c not in ["HVC", "H2"] else {"HVC": "Plastics", "H2": "Hydrogen"}[c] 
    for c in commodities
]

heatmap_data = pd.DataFrame(index=commodity_order, columns=scenario_order, dtype=float)
annot_data = pd.DataFrame(index=commodity_order, columns=scenario_order, dtype=object)

for commodity in commodities:
    display_name = {"HVC": "Plastics", "H2": "Hydrogen"}.get(commodity, commodity.title())

    for scenario_key, scenario_name in zip(
        ["Deindustrialization",  "Import_Deindustrialization", "Reindustrialize", "Import_Reindustrialize"],
        scenario_order
    ):
        if display_name in ["Cement", "Plastics"] and "Import" in scenario_name:
            heatmap_data.loc[display_name, scenario_name] = np.nan
            annot_data.loc[display_name, scenario_name] = ""
            continue

        total = sum([cost_data[scenario_key][commodity].astype(float)[year] for year in selected_years]) * 10
        avg_per_year = total / 30

        # Store avg per year in heatmap
        heatmap_data.loc[display_name, scenario_name] = avg_per_year
        annot_data.loc[display_name, scenario_name] = f"{avg_per_year:.1f}"

# === Create custom colormap for avg annual cost ===
all_vals = heatmap_data.values.flatten()
all_vals = all_vals[~np.isnan(all_vals)]

vmin = 0
v2 = 20
v3 = 50
v4 = 150
vmax = 300 #all_vals.max()

boundaries = [vmin, v2, v3, v4, vmax]
colors = ["lightblue", "white", "pink", "red", "#AA2622"]

custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", list(zip(
    np.linspace(0, 1, len(boundaries)),
    colors
)))
norm = BoundaryNorm(boundaries, ncolors=custom_cmap.N, clip=True)

# === Plot heatmap ===
plt.figure(figsize=(12, 6))
ax = sns.heatmap(
    heatmap_data,
    annot=annot_data,       # optional, you can set annot=None to remove numbers entirely
    fmt="",
    cmap=custom_cmap,
    norm=norm,
    linewidths=0.5,
    linecolor='gray',
    cbar_kws={"label": "Avg annual subsidies [bnEUR/a]"},
    square=True,
    annot_kws={"fontsize": 12, "weight": "bold", "color": "black"}
)

# Adjust x-axis labels
ax.set_xticklabels(
    ["Deindust",  "Deindust\nImport", "Reindust", "Reindust\nImport"], 
    rotation=20, ha='center', fontsize=11
)

#plt.title("Black number = Avg per year [bnEUR/a]", fontsize=14, weight='bold')
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

fig = plt.gcf()
fig.subplots_adjust(left=0.2, right=0.7, top=0.9, bottom=0.12)
plt.savefig("./graphs/avg_annual_subsidies_heatmap.png", dpi=300)
plt.show()
"""