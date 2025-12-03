# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 09:04:35 2025

@author: Dibella
"""


import pypsa
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


scenario_labels = {
    "policy_reg_deindustrial": "Deindustrialization\nNo relocation",
    "policy_eu_deindustrial": "Deindustrialization\nRelocation",
    "policy_reg_regain": "Reindustrialization\nNo relocation",
    "policy_eu_regain": "Reindustrialization\nRelocation"
}

scenario_colors = {
    "policy_reg_deindustrial": "#FC814A",
    "policy_reg_regain": "#28C76F",
    "policy_eu_deindustrial": "#AA6DA3",
    "policy_eu_regain": "#3C89CD"
}

scenarios = ["policy_reg_deindustrial",
             "policy_eu_deindustrial",
             "policy_reg_regain",
             "policy_eu_regain",]


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


def extract_avg_prices_per_country(networks, scenarios, years):

    avg_prices = {}
    for scenario in scenarios:
        avg_prices[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]
            timestep = n.snapshot_weightings.iloc[0,0]

            # Prices at buses (€/MWh), restrict to main nodes
            prices = n.buses_t.marginal_price
            prices = prices.loc[:, prices.columns.str.endswith(" 0")]
            prices.columns = prices.columns.str[:3]

            elec_gen = n.generators[n.generators["bus"].str.endswith(" 0")]
            gen_dispatch = n.generators_t.p.loc[:,elec_gen.index] * timestep
            gen_dispatch.columns = gen_dispatch.columns.str[:3]
            #gen_dispatch = gen_dispatch.T.groupby(gen_dispatch.columns).sum().T    
            
            elec_sto = n.storage_units_t.p.loc[:,n.storage_units_t.p.columns.str.contains('hydro')].sum()
            sto_dispatch = n.storage_units_t.p.loc[:,elec_sto.index] * timestep
            sto_dispatch.columns = sto_dispatch.columns.str[:3]
            #sto_dispatch = sto_dispatch.T.groupby(sto_dispatch.columns).sum().T  
            
            elec_plants = ["CCGT", "OCGT", "CHP", "lignite","nuclear","coal"]
            elec_links = n.links[n.links.index.str.contains("|".join(elec_plants), case= False)]
            lin_dispatch = -n.links_t.p1.loc[:,elec_links.index] * timestep
            lin_dispatch.columns = lin_dispatch.columns.str[:3]
            #lin_dispatch = lin_dispatch.T.groupby(lin_dispatch.columns).sum().T  
            
            tot_dispatch = pd.concat([gen_dispatch, sto_dispatch, lin_dispatch], axis = 1)
            tot_dispatch = tot_dispatch.T.groupby(tot_dispatch.columns).sum().T
            
            tot_cost_country = (tot_dispatch * prices).sum()
            tot_cost_country.index = tot_cost_country.index.str[:2]
            tot_cost_country = tot_cost_country.groupby(level=0).sum()
            tot_dispatch_country = tot_dispatch.sum()
            tot_dispatch_country.index = tot_dispatch_country.index.str[:2]
            tot_dispatch_country = tot_dispatch_country.groupby(level=0).sum()
            
            avg_price_country = tot_cost_country / tot_dispatch_country
            
            avg_prices[scenario][year] = avg_price_country
    
    return avg_prices

def industrial_elec_per_country(networks, scenarios, years):
    
    ind_plants0 = ["EAF","Haber-Bosch","Electrolysis"]
    ind_plants2 = ["methanolisation"]
    ind_plants3 = ["BF-BOF","DRI","BOF CC"]
    ind_plants4 = ["cement TGR","naphtha steam"]

    ind_elec_cons = {}
    for scenario in scenarios:
        ind_elec_cons[scenario] = {}
        for year in years:
            n = networks[(scenario, year)]
            timestep = n.snapshot_weightings.iloc[0,0]

            elec_ind0 = n.links_t.p0.loc[:,n.links_t.p0.columns.str.contains("|".join(ind_plants0))]
            elec_ind2 = n.links_t.p2.loc[:,n.links_t.p2.columns.str.contains("|".join(ind_plants2))]
            elec_ind3 = n.links_t.p3.loc[:,n.links_t.p3.columns.str.contains("|".join(ind_plants3))]
            elec_ind4 = n.links_t.p4.loc[:,n.links_t.p4.columns.str.contains("|".join(ind_plants4))]

            elec_ind = pd.concat([elec_ind0,elec_ind2,elec_ind3,elec_ind4], axis = 1)
            elec_ind.columns = elec_ind.columns.str[:2]
            elec_ind = elec_ind.T.groupby(elec_ind.columns).sum().T
            elec_ind = elec_ind.sum() * timestep

            ind_elec_cons[scenario][year] = elec_ind
    
    return ind_elec_cons

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_september_new/"
scenario = "base_eu_regain"
regions_fn = root_dir + "resources/" + scenario + "/regions_onshore_base_s_39.geojson"

networks = load_networks(scenarios, years, root_dir, res_dir)

# %%
# Extract average electricity prices per country
avg_prices = extract_avg_prices_per_country(networks, scenarios, years)

ind_elec_cons = industrial_elec_per_country(networks, scenarios, years)

# %%


# Compute total spending per scenario and year (already in billions €)
total_spending = {}
for scenario in scenarios:
    total_spending[scenario] = {}
    for year in years:
        prices = avg_prices[scenario][year]
        cons = ind_elec_cons[scenario][year]
        common_countries = prices.index.intersection(cons.index)
        prices = prices.loc[common_countries]
        cons = cons.loc[common_countries]
        spending_country = prices * cons
        total_spending[scenario][year] = spending_country.sum() / 1e9  # B€

# Convert to DataFrame
total_spending_df = pd.DataFrame(total_spending, index=years)
total_spending_df.index.name = "Year"

# --- Plot grouped bar chart ---
fig, ax = plt.subplots(figsize=(8, 8))
width = 0.2
x = np.arange(len(years))

# Map scenario to label for plotting
scenario_label_map = {s: scenario_labels.get(s, s) for s in scenarios}

# Plot bars
for i, scenario in enumerate(scenarios):
    ax.bar(x + i*width, total_spending_df[scenario], width=width, edgecolor='black',linewidth=0.7,
           color=scenario_colors[scenario], label=scenario_label_map[scenario])

# --- Function to add difference boxes ---
def add_difference_box(ax, xpos, val_no, val_reloc):
    diff_abs = val_reloc - val_no
    diff_pct = 100 * diff_abs / val_no
    ax.annotate(
        f"{diff_abs:+.2f} bnEUR\n({diff_pct:+.1f}%)",
        xy=(xpos, val_reloc),
        xytext=(xpos, ((val_no + val_reloc)/2)+10),
        textcoords='data',
        ha='center', va='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
        #arrowprops=dict(arrowstyle="->", color='black', lw=1)
    )

# Add difference boxes between No-relocation and Relocation bars
for j, year in enumerate(years):
    # Deindustrialization pair
    x_pos_deind = x[j] - width/2 + 1.5*width  # center between two bars
    val_no_deind = total_spending_df.loc[year, "policy_reg_deindustrial"]
    val_rel_deind = total_spending_df.loc[year, "policy_eu_deindustrial"]
    add_difference_box(ax, x_pos_deind, val_no_deind, val_rel_deind)
    
    # Reindustrialization pair
    x_pos_reind = x[j] + width/2 + 2.5*width  # center between two bars
    val_no_reind = total_spending_df.loc[year, "policy_reg_regain"]
    val_rel_reind = total_spending_df.loc[year, "policy_eu_regain"]
    add_difference_box(ax, x_pos_reind, val_no_reind, val_rel_reind)

# Final formatting
ax.set_xticks(x + width*1.5)
ax.set_xticklabels(years)
ax.set_ylabel("bnEUR/a")
ax.set_title("Total industrial electricity spending", fontsize = 14)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("./graphs/ind_elec_costs_reloc.png", dpi=300)
plt.show()


