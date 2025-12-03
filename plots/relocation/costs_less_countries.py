# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:02:16 2025

@author: Dibella
"""

import pypsa
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

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

scenarios = [
    "policy_eu_regain",
    "policy_reg_regain",
    "policy_reg_regain_noH2"
]
years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
networks = load_networks(scenarios, years, root_dir, res_dir)

# %%
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
save_path = "./full_graphs/cost_share_by_country.png"

group_countries = {
    'North-Western Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'Southern Europe': ['ES', 'IT', 'PT', 'GR'],
    'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}

custom_colors = {
    'Southern Europe': '#D8973C',
    'North-Western Europe': '#1B264F',
    'Eastern Europe': '#9B7EDE',
}

# --- Cost allocation ---
grouped_costs = defaultdict(lambda: defaultdict(float))  # [scenario][(year, region)] = total_cost
country_costs = defaultdict(lambda: defaultdict(float))  # [scenario][(year, country)] = cost
link_costs_dict = {}  # Initialize dictionary before the loop

for (scenario, year), n in networks.items():
    # Generator costs
    
    industrial_keywords = [
        "DRI-EAF", "BF-BOF", "TGR", "Cement Plant", "methanolisation",
        "Haber", "naphtha steam cracker", "Electrolysis", "Fischer", "H2 pipeline"
    ]

    # Link costs
    links = n.links.copy()
    links = links[(links['p_nom_opt'] > 0) & (links['p_nom_extendable'] == True)]
    # Filter links by checking if their bus0 contains any industrial keyword (case-insensitive)
    mask = links.index.to_series().apply(lambda x: any(kw in x for kw in industrial_keywords))
    links = links[mask]
    # Remove "H2 pipeline " prefix from index if present
    links.index = links.index.to_series().apply(
        lambda x: x.replace("H2 pipeline ", "") if x.startswith("H2 pipeline ") else x
    )

    links['country'] = links.bus0.str[:2]
    links['cost'] = links.p_nom_opt * links.capital_cost
    link_costs = links.groupby('country')['cost'].sum()
    
    link_costs_dict[(scenario, year)] = link_costs

    # Total cost per country
    total_costs = link_costs
    total_costs = total_costs[total_costs.index.str.match(r'^[A-Z]{2}$')]
    total_costs = total_costs[total_costs.index != 'EU']

    # Safely iterate over a static copy of items
    total_cost_items = list(total_costs.items())
    for c, val in total_cost_items:
        country_costs[scenario][(year, c)] += val


    # Group per region
    for region, countries in group_countries.items():
        regional_cost = total_costs[total_costs.index.isin(countries)].sum()
        grouped_costs[scenario][(year, region)] = regional_cost

fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

bar_width = 0.35
offset = {"policy_eu_regain": -bar_width/2, "policy_reg_regain": bar_width/2}
x = pd.Series([i for i in range(len(years))], index=years)
region_order = ['North-Western Europe', 'Southern Europe', 'Eastern Europe']

# --- Absolute values plot (top) ---
ax_abs = axes[0]
for scenario in scenarios:
    bar_bottom = pd.Series(0, index=years)
    for region in region_order:
        abs_values = []
        for year in years:
            region_cost = grouped_costs[scenario].get((year, region), 0)
            abs_values.append(region_cost/1e9)

        ax_abs.bar(
            x[years] + offset[scenario],
            abs_values,
            bottom=bar_bottom,
            width=bar_width,
            label=region if scenario == scenarios[0] else None,
            color=custom_colors[region],
            edgecolor='white'
        )

        # Label each bar segment
        for i, (xx, y, base) in enumerate(zip(x[years] + offset[scenario], abs_values, bar_bottom)):
            if y > 0:
                ax_abs.text(xx, base + y/2, f"{int(y):,}", ha='center', va='center', fontsize=8, color='white')

        bar_bottom += abs_values

ax_abs.set_ylabel("Total Investment Cost [b€]")
ax_abs.set_title("Absolute Investment by Region")
ax_abs.grid(True, linestyle="--", axis='y')
ax_abs.legend(title="Region", loc="upper left", bbox_to_anchor=(1, 1))

# --- Percentage plot (bottom) ---
ax_pct = axes[1]
for scenario in scenarios:
    bar_bottom = pd.Series(0, index=years)
    for region in region_order:
        percent = []
        for year in years:
            region_cost = grouped_costs[scenario].get((year, region), 0)
            total = sum(
                grouped_costs[scenario].get((year, r), 0)
                for r in group_countries
            )
            pct = (region_cost / total * 100) if total > 0 else 0
            percent.append(pct)

        ax_pct.bar(
            x[years] + offset[scenario],
            percent,
            bottom=bar_bottom,
            width=bar_width,
            label=region if scenario == scenarios[0] else None,
            color=custom_colors[region],
            edgecolor='white'
        )

        for i, (xx, y, base) in enumerate(zip(x[years] + offset[scenario], percent, bar_bottom)):
            if y > 1:
                ax_pct.text(xx, base + y/2, f"{int(y)}%", ha='center', va='center', fontsize=8, color='white')

        bar_bottom += percent

ax_pct.set_xticks(x)
ax_pct.set_xticklabels(years)
ax_pct.set_ylabel("Share of Total Investment [%]")
ax_pct.set_title("Investment Distribution by Region")
ax_pct.grid(True, linestyle="--", axis='y')

plt.tight_layout()
plt.savefig("./graphs/investment_by_region_absolute_and_percentage.png", dpi=300)
plt.show()

# %%
fig, axes = plt.subplots(1, len(years), figsize=(16, 5), sharey=True)

# Countries sorted alphabetically
all_countries = sorted({c for countries in group_countries.values() for c in countries})

# Colors for countries
import matplotlib.cm as cm
cmap = cm.get_cmap('tab20', len(all_countries))
country_colors = {country: cmap(i) for i, country in enumerate(all_countries)}

bar_width = 0.7
x = pd.Series([i for i in range(len(all_countries))], index=all_countries)

for ax, year in zip(axes, years):
    diffs = []
    for country in all_countries:
        cost_reg = country_costs["policy_reg_regain"].get((year, country), 0)
        cost_eu = country_costs["policy_eu_regain"].get((year, country), 0)
        diffs.append((cost_reg - cost_eu) / 1e9)  # difference in billion €

    bars = ax.bar(x[all_countries], diffs, color=[country_colors[c] for c in all_countries])
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.05:  # only label if difference > 50 million €
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height if height > 0 else 0,
                f"{height:.2f}",
                ha='center',
                va='bottom' if height > 0 else 'top',
                fontsize=8,
                rotation=90,
                color='black'
            )

    ax.set_title(f"Difference in Investment (Reg - EU) - {year}")
    ax.set_xticks(x[all_countries])
    ax.set_xticklabels(all_countries, rotation=90)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_ylabel("Difference [b€]" if ax == axes[0] else "")
    ax.grid(True, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig("./graphs/investment_difference_by_country.png", dpi=300)
plt.show()

# %%

all_countries = sorted({c for countries in group_countries.values() for c in countries})

scenarios = ["policy_eu_regain", "policy_reg_regain"]
bar_width = 0.35
years_idx = np.arange(len(years))

# Prepare colors for scenarios
scenario_colors = {
    "policy_eu_regain": "#00B050",  # blue
    "policy_reg_regain": "#3AAED8"  # orange
}
# Map country codes to full country names (can expand if you want more)
# Map country codes to full country names (can expand if you want more)
country_names = {
    "AT": "Austria", "BE": "Belgium", "CH": "Switzerland", "DE": "Germany",
    "FR": "France", "LU": "Luxembourg", "NL": "Netherlands", "DK": "Denmark",
    "EE": "Estonia", "FI": "Finland", "LV": "Latvia", "LT": "Lithuania",
    "NO": "Norway", "SE": "Sweden", "GB": "United Kingdom", "IE": "Ireland",
    "ES": "Spain", "IT": "Italy", "PT": "Portugal", "GR": "Greece",
    "BG": "Bulgaria", "CZ": "Czech Republic", "HU": "Hungary", "PL": "Poland",
    "RO": "Romania", "SK": "Slovakia", "SI": "Slovenia", "AL": "Albania",
    "BA": "Bosnia & Herzegovina", "HR": "Croatia", "ME": "Montenegro",
    "MK": "North Macedonia", "RS": "Serbia", "XK": "Kosovo"
}

scenario_labels = {
    "base_eu_regain": "No climate policy\nCompetitive industry\nRelocation",
    "policy_eu_regain": "Climate policy\nCompetitive industry\nRelocation",
    "policy_eu_deindustrial": "Climate policy\nDeindustrialization\nRelocation",
    "policy_reg_regain": "Climate policy\nCompetitive industry\nHistorical hubs"
}

# Adjust your plotting code to use these labels and names:
ncols = 9
nrows = 4
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))  # no sharey, variable scales
axes = axes.flatten()

all_countries = sorted(
    {c for countries in group_countries.values() for c in countries},
    key=lambda c: country_names.get(c, c)
)
# Then loop over countries_sorted instead of countries
for ax_idx, country in enumerate(all_countries):
    # your plotting code here
    ax = axes[ax_idx]

    costs_eu = [country_costs["policy_eu_regain"].get((year, country), 0) / 1e6 for year in years]
    costs_reg = [country_costs["policy_reg_regain"].get((year, country), 0) / 1e6 for year in years]

    ax.bar(years_idx - bar_width/2, costs_eu, width=bar_width, label=scenario_labels["policy_eu_regain"], color=scenario_colors["policy_eu_regain"], edgecolor='black')
    ax.bar(years_idx + bar_width/2, costs_reg, width=bar_width, label=scenario_labels["policy_reg_regain"], color=scenario_colors["policy_reg_regain"], edgecolor='black')

    ax.set_title(country_names.get(country, country), fontsize = 14)
    ax.set_xticks(years_idx)
    ax.set_xticklabels(years)
    ax.grid(True, axis='y', linestyle='--')

    if ax_idx % ncols == 0:
        ax.set_ylabel("Industrial investments [M€]")

for ax in axes[len(all_countries):]:
    fig.delaxes(ax)

# Place legend once, with nice labels
fig.legend(
    [scenario_labels["policy_eu_regain"], scenario_labels["policy_reg_regain"]],
    loc='lower right', fontsize=12, bbox_to_anchor=(0.95,0.05)
)

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.suptitle("Investment by Country and Scenario", y=1.02, fontsize=16)
plt.savefig("./graphs/investment_by_country_grouped_bars_variable_scales_named.png", dpi=300)
plt.show()

# %%

# Countries to include in plot
selected_countries = ["DE", "IT", "FR", "ES", "PL", "NL"]

# Sort them by country name for nice order
all_countries = sorted(
    selected_countries,
    key=lambda c: country_names.get(c, c)
)

# Adjust layout based on number of selected countries
ncols = 3
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=False)
axes = axes.flatten()

for ax_idx, country in enumerate(all_countries):
    ax = axes[ax_idx]

    costs_eu = [country_costs["policy_eu_regain"].get((year, country), 0) / 1e6 for year in years]
    costs_reg = [country_costs["policy_reg_regain"].get((year, country), 0) / 1e6 for year in years]

    ax.bar(years_idx - bar_width/2, costs_eu, width=bar_width,
           label=scenario_labels["policy_eu_regain"],
           color=scenario_colors["policy_eu_regain"], edgecolor='black')
    ax.bar(years_idx + bar_width/2, costs_reg, width=bar_width,
           label=scenario_labels["policy_reg_regain"],
           color=scenario_colors["policy_reg_regain"], edgecolor='black')

    ax.set_title(country_names.get(country, country), fontsize=14)
    ax.set_xticks(years_idx)
    ax.set_xticklabels(years)
    ax.grid(True, axis='y', linestyle='--')

    if ax_idx % ncols == 0:
        ax.set_ylabel("Industrial investments [M€]")

# Remove any unused axes
for ax in axes[len(all_countries):]:
    fig.delaxes(ax)

# Add a shared legend
fig.legend(
    [scenario_labels["policy_eu_regain"], scenario_labels["policy_reg_regain"]],
    loc='lower right', fontsize=12, bbox_to_anchor=(0.95, 0.05)
)

plt.tight_layout(rect=[0, 0, 0.95, 1])
plt.suptitle("Investment by Country and Scenario", y=1.02, fontsize=16)
plt.savefig("./graphs/investment_selected_countries.png", dpi=300)
plt.show()

# %%

# New structure to store per-tech costs
# New structure to store per-tech costs
country_tech_costs = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
# structure: [scenario][(year, country)][tech] = cost

for (scenario, year), n in networks.items():
    industrial_keywords = [
        "DRI-EAF", "BF-BOF", "TGR", "Cement Plant", "methanolisation",
        "Haber", "naphtha steam cracker", "Electrolysis", "Fischer", "H2 pipeline"
    ]

    links = n.links.copy()
    links = links[(links['p_nom_opt'] > 0) & (links['p_nom_extendable'] == True)]

    # Filter for industrial links
    mask = links.index.to_series().apply(lambda x: any(kw in x for kw in industrial_keywords))
    links = links[mask]

    # Clean H2 pipeline names
    links.index = links.index.to_series().apply(
        lambda x: x.replace("H2 pipeline ", "") if x.startswith("H2 pipeline ") else x
    )

    links['country'] = links.index.str[:2]
    links['technology'] = links['carrier']
    
    links['cost'] = links.p_nom_opt * links.capital_cost

    # Accumulate costs per scenario, year, country, technology
    for _, row in links.iterrows():
        country = row['country']
        tech = row['technology']
        cost = row['cost']
        if len(country) == 2 and country != 'EU':
            country_tech_costs[scenario][(year, country)][tech] += cost
            

# %%

# Create a color dictionary for the techs (pick any palette you like)
tech_colors = {
    "DRI-EAF": "#00BFFF",             # Fresh green — for clean steel (DRI-EAF)
    "BF-BOF": "#333333",              # Dark gray/black — for legacy steel
    "cement TGR": "#E74C3C",          # Strong red — emissions heavy
    "Cement Plant": "#A9A9A9",        # Neutral dark grey
    "methanolisation": "#9B59B6",     # Soft purple — distinct & visible
    "Haber-Bosch": "#8c564b",         # Original brown — still fits
    "naphtha steam cracker": "#555555", # Mid-grey — fossil heavy
    "H2 Electrolysis": "#4CAF50",     # Bright blue — modern, clean tech
    "Fischer-Tropsch": "#C71585",     # Deep pink/magenta — bold, clear
    "H2 pipeline": "yellow",         # Steel blue — matches H2 infra
    "default": "#cccccc"              # Neutral light gray fallback
}



# Updated plotting code

selected_countries = ["DE", "IT", "FR", "ES", "PL", "NL"]
scenarios = ["policy_eu_regain", "policy_reg_regain", "policy_reg_regain_noH2"]
scenario_labels = {
    "policy_eu_regain": "Climate policy\nCompetive industry\nRelocation",
    "policy_reg_regain": "Climate policy\nCompetive industry\nHistorical hubs",
    "policy_reg_regain_noH2": "Climate policy\nCompetive industry\nHistorical hubs\nNo H2 network"
}

nrows = len(selected_countries)
ncols = len(scenarios)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey='row')
if nrows == 1:
    axes = axes[np.newaxis, :]  # Ensure 2D array for consistency

years_idx = np.arange(len(years))  # Assuming years and years_idx are defined outside

for row_idx, country in enumerate(selected_countries):
    for col_idx, scenario in enumerate(scenarios):
        ax = axes[row_idx, col_idx]
        bottom = np.zeros(len(years))

        # Collect all techs for this scenario, year, and country
        techs = set()
        for year in years:
            techs.update(country_tech_costs[scenario][(year, country)].keys())
        techs = sorted(techs)

        for tech in techs:
            values = [
                country_tech_costs[scenario][(year, country)].get(tech, 0) / 1e6
                for year in years
            ]
            color = tech_colors.get(tech, tech_colors["default"])
            ax.bar(
                years_idx,
                values,
                width=0.8,
                bottom=bottom,
                label=tech,
                color=color,
                edgecolor='black',
                alpha=0.7
            )
            bottom += values

        # Titles and labels
        if row_idx == 0:
            ax.set_title(scenario_labels[scenario], fontsize=14)
        if col_idx == 0:
            ax.set_ylabel(f"{country_names[country]}\nInvestment [M€]", fontsize=12)

        # Show xticks only on last row
        if row_idx == nrows - 1:
            ax.set_xticks(years_idx)
            ax.set_xticklabels(years, rotation=45)
        else:
            ax.set_xticks([])  # Hide x-axis ticks and labels for other rows

        ax.grid(True, axis='y', linestyle='--')

# Remove any unused axes if present (unlikely in this setup)
total_axes = nrows * ncols
for i in range(len(selected_countries)*ncols, total_axes):
    fig.delaxes(axes.flat[i])

handles, labels = axes[0, 0].get_legend_handles_labels()
filtered = {}
for label, handle in zip(labels, handles):
    lower_label = label.lower()
    if lower_label == "bf-bof" or "cement" in lower_label or "naphtha" in lower_label:
        continue
    # Capitalize only methanolisation
    if lower_label == "methanolisation":
        filtered["Methanolisation"] = handle
    else:
        filtered[label] = handle

fig.legend(
    filtered.values(),
    filtered.keys(),
    loc='center right',
    bbox_to_anchor=(0.19, 0.84), 
    title="",
    fontsize=10,
    title_fontsize=11
)


plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("./graphs/stacked_investments_selected_countries_by_tech_scenarios.png", dpi=300)
plt.show()

# %%
# Define country colors (pick a palette of 6 distinct, readable colors)
country_colors = {
    "DE": "#1f77b4",  # Blue
    "IT": "#ff7f0e",  # Orange
    "FR": "#2ca02c",  # Green
    "ES": "#d62728",  # Red
    "PL": "#9467bd",  # Purple
    "NL": "#8c564b",  # Brown
}

fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 5), sharey=True)

if len(scenarios) == 1:
    axes = [axes]  # Ensure iterable

years_idx = np.arange(len(years))

for col_idx, scenario in enumerate(scenarios):
    ax = axes[col_idx]
    
    for country in selected_countries:
        # Sum across all technologies for each year
        total_investments = [
            sum(country_tech_costs[scenario][(year, country)].values()) / 1e6
            for year in years
        ]
        ax.plot(
            years_idx,
            total_investments,
            label=country_names[country],
            color=country_colors[country],
            linewidth=2
        )

    ax.set_title(scenario_labels[scenario], fontsize=14)
    ax.set_xticks(years_idx)
    ax.set_xticklabels(years, rotation=45)
    ax.grid(True, axis='y', linestyle='--')
    if col_idx == 0:
        ax.set_ylabel("Total Investment [M€]", fontsize=12)

# Add legend once
fig.legend(
    handles=[plt.Line2D([0], [0], color=country_colors[c], lw=2) for c in selected_countries],
    labels=[country_names[c] for c in selected_countries],
    loc='center right',
    bbox_to_anchor=(1.0, 0.5),
    title="Countries",
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.suptitle("Total Industrial Investment by Country and Scenario", y=1.02, fontsize=16)
plt.savefig("./graphs/line_total_investments_by_country_per_scenario.png", dpi=300)
plt.show()
