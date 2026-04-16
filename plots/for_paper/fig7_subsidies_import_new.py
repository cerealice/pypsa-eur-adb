# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 21:25:13 2025

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

def build_production_df(scenarios, years, networks):
    """
    Builds a MultiIndex DataFrame with production per commodity per scenario and year,
    using already-loaded networks.
    """
    index = pd.MultiIndex.from_product([scenarios, years], names=["scenario", "year"])
    production = pd.DataFrame(0, index=index, columns=commodities, dtype=float)

    for scenario in scenarios:
        for year in years:
            n = networks[(scenario, year)]
            production.loc[(scenario, year), "steel"] = get_steel_prod_eu(n).sum()
            production.loc[(scenario, year), "ammonia"] = get_ammonia_prod_eu(n).sum()
            production.loc[(scenario, year), "methanol"] = get_methanol_prod_eu(n).sum()
            production.loc[(scenario, year), "HVC"] = get_hvc_prod_eu(n).sum()

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
    "policy_reg_deindustrial",
    "policy_reg_regain",
    "import_policy_reg_deindustrial",
    "import_policy_reg_regain"
]
years = [2030, 2040, 2050]

scenario_colors = {
    "policy_reg_deindustrial": "#FC814A",
    "policy_reg_regain": "#28C76F",
    "import_policy_reg_deindustrial": "#A6A14E",
    "import_policy_reg_regain": "#6BCDC9"
}

scenario_labels = {
    "policy_reg_deindustrial": "Climate policy\nCurrent deindustr trend",
    "policy_reg_regain": "Climate policy\nReindustrialize",
    "import_policy_reg_deindustrial": "Import policy\nCurrent deindustr trend",
    "import_policy_reg_regain": "Import policy\nReindustrialize"
}

commodities = ["steel", "ammonia", "methanol", "HVC",]

lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t
naphtha_to_hvc = (2.31 * 12.47) * 1000
decay_emis_hvc = 0.2571 * naphtha_to_hvc / 1e3
lhv_hydrogen = 33.33 #MWh/t

ft_to_hvc = 2.31 * 12.47

min_import_costs = {
    "steel": 417,
    "ammonia": 87.7 * lhv_ammonia,
    "methanol": 106.8 * lhv_methanol,
    "HVC": 109.8 * ft_to_hvc,
    #"H2": 57.5 * lhv_hydrogen
}


# Store results
cost_data = {
    "Continued Decline": pd.DataFrame(index=years, columns=commodities),
    "Reindustrialize": pd.DataFrame(index=years, columns=commodities),
    "Import_Continued Decline": pd.DataFrame(index=years, columns=commodities),
    "Import_Reindustrialize": pd.DataFrame(index=years, columns=commodities)
}


# === INIT STORAGE ===
price_data = {commodity: pd.DataFrame(index=scenarios, columns=years) for commodity in commodities}

# === LOAD AND COMPUTE PRICES ===
max_value = 0
cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_october/"

networks = load_networks(scenarios, years, root_dir, res_dir)


# --- Your price calculation loop (refactored to use preloaded networks) ---
for scenario in scenarios:
    for year in years:
        n = networks[(scenario, year)]  # pull from preloaded dict
        timestep = n.snapshot_weightings.iloc[0, 0]

        # Steel and cement (€/t → convert to k€/t if needed)
        price_data["steel"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="steel") / 1e3


        # Ammonia (€/MWh → €/t via LHV)
        price_data["ammonia"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="NH3") * lhv_ammonia

        # Methanol
        price_data["methanol"].loc[scenario, year] = weighted_average_marginal_price(n, keyword="methanol") * lhv_methanol
        meth_price = n.buses_t.marginal_price.loc[
            :, n.buses_t.marginal_price.columns.str.contains("industry methanol")
        ].mean().iloc[0] * lhv_methanol

        co2_price = -n.global_constraints.loc["CO2Limit", "mu"]
        extra_methanol_cost = 0.248 * lhv_methanol * co2_price
        # Optionally replace with meth_price or add extra_methanol_cost

        # HVC
        hvc_price = weighted_average_marginal_price(n, keyword="HVC") / 1e3
        extra_hvc_cost = 0.2571 * 12.47 * co2_price
        price_data["HVC"].loc[scenario, year] = hvc_price

        # Track maximum HVC price for plotting
        max_value = max(max_value, price_data["HVC"].loc[scenario, year])



# --- Build production data ---
production = build_production_df(scenarios, years, networks)


# --- Compute cost deltas vs. min import price ---
for year in years:
    for commodity in commodities:
        # Get min import reference price
        if commodity not in min_import_costs:
            continue  # skip if no reference defined
        ref_price = min_import_costs[commodity]

        # Compute deltas for each scenario relative to ref_price
        price_delta_deindustrial = max(
            price_data[commodity].loc["policy_reg_deindustrial", year] - ref_price, 0
        )
        price_delta_regain = max(
            price_data[commodity].loc["policy_reg_regain", year] - ref_price, 0
        )
        price_delta_import_deindustrial = max(
            price_data[commodity].loc["import_policy_reg_deindustrial", year] - ref_price, 0
        )
        price_delta_import_regain = max(
            price_data[commodity].loc["import_policy_reg_regain", year] - ref_price, 0
        )

        # Production volumes
        prod_deindustrial = production.loc[("policy_reg_deindustrial", year), commodity]
        prod_regain = production.loc[("policy_reg_regain", year), commodity]
        prod_deindustrial_import = production.loc[("import_policy_reg_deindustrial", year), commodity]
        prod_regain_import = production.loc[("import_policy_reg_regain", year), commodity]

        # Apply deltas × production (convert to bn EUR)
        cost_data["Continued Decline"].loc[year, commodity] = price_delta_deindustrial * prod_deindustrial / 1e9
        cost_data["Reindustrialize"].loc[year, commodity] = price_delta_regain * prod_regain / 1e9
        cost_data["Import_Continued Decline"].loc[year, commodity] = price_delta_import_deindustrial * prod_deindustrial_import / 1e9
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

max_y = 45 #compute_max_y(plot_commodities)

# Plotting function
def plot_commodity(ax, commodity, max_y_limit):
    # Read all four scenarios
    deindustrial = cost_data["Continued Decline"][commodity].fillna(0).astype(float)
    import_deindustrial = cost_data["Import_Continued Decline"][commodity].fillna(0).astype(float)
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

    x_labels = ["Cont. Decline", "Import\nCont. Decline", "Reindust", "Import\nReindust"]

    bars = ax.bar(range(4), totals, color=bar_colors, width=0.6, zorder=2, edgecolor="black", linewidth=0.7,)

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
selected_years = [2040, 2050]
# Exclude Hydrogen
plot_commodities = [c for c in commodities if c.lower() != "h2"]
n_commodities = len(plot_commodities)

fig, axes = plt.subplots(1, n_commodities, figsize=(4 * n_commodities, 6), sharey=False)

custom_titles = {"hvc": "Plastics"}


max_y = 60 #compute_max_y(plot_commodities)

# Plotting function
def plot_commodity(ax, commodity, max_y_limit):
    # Read all four scenarios
    deindustrial = cost_data["Continued Decline"][commodity].fillna(0).astype(float)
    import_deindustrial = cost_data["Import_Continued Decline"][commodity].fillna(0).astype(float)
    reindustrial = cost_data["Reindustrialize"][commodity].fillna(0).astype(float)
    import_reindustrial = cost_data["Import_Reindustrialize"][commodity].fillna(0).astype(float)

    # Total over pathway
    totals = [
        sum([deindustrial[year] for year in selected_years]) * 10/20,
        sum([import_deindustrial[year] for year in selected_years]) * 10/20,
        sum([reindustrial[year] for year in selected_years]) * 10/20,
        sum([import_reindustrial[year] for year in selected_years]) * 10/20
    ]

    bar_colors = [
        scenario_colors["policy_reg_deindustrial"],
        scenario_colors["import_policy_reg_deindustrial"],
        scenario_colors["policy_reg_regain"],
        scenario_colors["import_policy_reg_regain"]
    ]

    x_labels = ["Cont. Decline", "Import\nCont. Decline", "Reindust", "Import\nReindust"]

    bars = ax.bar(range(4), totals, color=bar_colors, width=0.6, zorder=2, edgecolor="black", linewidth=0.7,)

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
plt.savefig("./graphs/total_subsidies_single_row_2040.png", dpi=300)
plt.show()

