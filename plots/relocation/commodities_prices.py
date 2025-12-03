# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 11:58:40 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
scenario = "policy_reg_regain"
years = [ 2030, 2040, 2050]
commodities = ["steel", "cement", "ammonia", "methanol", "HVC"]
commodities = ["steel", "cement", "ammonia", "methanol", "HVC", "hydrogen", "electricity"]

lhv_ammonia = 5.166  # MWh / t
lhv_methanol = 5.528  # MWh / t

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


cwd = os.getcwd()
parent_dir = os.path.dirname(os.path.dirname(cwd))


# === STORAGE ===
country_price_data = {commodity: {} for commodity in commodities}

for year in years: 
    file_path = os.path.join(parent_dir, "results_3h_juno", scenario, "networks", f"base_s_39___{year}.nc")
    n = pypsa.Network(file_path)
    co2_price = -n.global_constraints.loc["CO2Limit", "mu"]

    for commodity in commodities:
        country_price_data[commodity][year] = {}

        if commodity == "methanol":
            buses = n.buses_t.marginal_price.filter(like="industry methanol")
            for col in buses.columns:
                iso2 = col[:2]
                value = buses[col].mean() * lhv_methanol + 0.248 * lhv_methanol * co2_price
                country_price_data[commodity][year].setdefault(iso2, []).append(value)

        elif commodity == "HVC":
            buses = n.buses_t.marginal_price.filter(like="HVC")
            buses = buses.loc[:,~buses.columns.str.contains("sequestered")]
            for col in buses.columns:
                iso2 = col[:2]
                value = buses[col].mean() / 1e3 #- 0.2571 * 12.47 * co2_price
                country_price_data[commodity][year].setdefault(iso2, []).append(value)

        elif commodity == "ammonia":
            buses = n.buses_t.marginal_price.filter(like="NH3")
            for col in buses.columns:
                iso2 = col[:2]
                value = buses[col].mean() * lhv_ammonia
                country_price_data[commodity][year].setdefault(iso2, []).append(value)
                
        elif commodity == "hydrogen":
            buses = n.buses_t.marginal_price.filter(like="H2")
            buses = buses.loc[:, buses.columns.str.endswith("H2")]  # filter "H2 0" buses only
            for col in buses.columns:
                iso2 = col[:2]
                val = buses[col].mean()
                if val > 0:
                    country_price_data[commodity][year].setdefault(iso2, []).append(val)

        elif commodity == "electricity":
            buses = n.buses_t.marginal_price
            buses = buses.loc[:, buses.columns.str.endswith(" 0")]
            for col in buses.columns:
                iso2 = col[:2]
                val = buses[col].mean()
                if val > 0:
                    country_price_data[commodity][year].setdefault(iso2, []).append(val)


        else:  # steel, cement
            buses = n.buses_t.marginal_price.filter(like=commodity)
            if commodity == "cement":
                buses = buses.loc[:, ~buses.columns.str.contains("process emissions")]
            for col in buses.columns:
                iso2 = col[:2]
                value = buses[col].where(lambda x: x >= 0, 0).mean() / 1e3
                country_price_data[commodity][year].setdefault(iso2, []).append(value)
                


# Average per country
for commodity in country_price_data:
    for year in country_price_data[commodity]:
        for iso2 in country_price_data[commodity][year]:
            values = country_price_data[commodity][year][iso2]
            avg_val = sum(values) / len(values)
            country_price_data[commodity][year][iso2] = avg_val

# === GROUP BY REGION ===
region_price_data = {commodity: {year: {} for year in years} for commodity in commodities}

for commodity in commodities:
    for year in years:
        if year == 2020:
            for region in group_countries:
                region_price_data[commodity][year][region] = None
            continue

        for region, iso_list in group_countries.items():
            region_vals = [
                country_price_data[commodity][year].get(iso2)
                for iso2 in iso_list
                if country_price_data[commodity][year].get(iso2, 0) > 0
            ]
            region_avg = sum(region_vals) / len(region_vals) if region_vals else None
            region_price_data[commodity][year][region] = region_avg
            

region_range_data = {commodity: {} for commodity in commodities}

for commodity in commodities:
    for year in years:
        values = [
            val for val in region_price_data[commodity][year].values()
            if val is not None
        ]
        if values:
            region_range_data[commodity][year] = {
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values)
            }
        else:
            region_range_data[commodity][year] = {
                "min": None, "max": None, "mean": None
            }



# === PLOT ===
fig, axes = plt.subplots(1, len(commodities), figsize=(21, 5), sharex=True)
custom_titles = {
    "HVC": "Plastics",
    "hydrogen": "Hydrogen",
    "electricity": "Electricity"
}


for idx, (commodity, ax) in enumerate(zip(commodities, axes)):
    for region in group_countries:
        series = [region_price_data[commodity][year].get(region) for year in years]
        if any(val is not None for val in series):
            ax.plot(
                years,
                series,
                marker='o',
                label=region,
                color=custom_colors.get(region, 'black')
            )


    ax.set_title(custom_titles.get(commodity, commodity.title()))
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)
    if idx == 0:
        ax.set_ylabel("Price [€/t]")
    ax.grid(True, linestyle='--')
    
    range_min = [region_range_data[commodity][year]["min"] for year in years]
    range_max = [region_range_data[commodity][year]["max"] for year in years]
    range_mean = [region_range_data[commodity][year]["mean"] for year in years]
    
    if any(v is not None for v in range_mean):
        ax.plot(years, range_mean, linestyle='--', color='black',
                marker='x', label='Regional mean')
        ax.fill_between(years, range_min, range_max, color='gray', alpha=0.3,
                        label='Regional min–max range')


# Add shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
plt.tight_layout(rect=[0, 0.15, 1, 1])
plt.savefig("./graphs/commodity_prices_by_region_policy_reg_regain.png", dpi=300)
plt.show()

