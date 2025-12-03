# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 12:00:30 2025

@author: Dibella
"""

import pandas as pd
import pypsa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Extract capacity factor for each RES technology and country
def get_capacity_factors(network, res_techs, countries):
    cf_data = {tech: {} for tech in res_techs}

    for tech in res_techs:
        # Filter generators by tech
        gens = network.generators_t.p_max_pu
        
        is_res = gens.columns.str.contains(tech, case=False, na=False)
        selected_gens = gens.loc[:,is_res & ~gens.columns.str.contains("thermal", case=False, na=False)]

        if selected_gens.empty:
            continue

        # Sum p_max_pu over time (8760 hours assumed), group by country
        summed_p_max_pu = selected_gens.sum()

        for gen_id, total_pu in summed_p_max_pu.items():
            country = gen_id[:2]
            if country not in countries:
                continue
            if country not in cf_data[tech]:
                cf_data[tech][country] = 0.0
            cf_data[tech][country] += total_pu / 8760  # avg CF

    return cf_data

# Step 2: Convert to feature matrix
def make_feature_matrix(cf_data, all_countries, res_techs):
    data = []
    for country in all_countries:
        row = []
        for tech in res_techs:
            row.append(cf_data.get(tech, {}).get(country, 0.0))
        data.append(row)
    return pd.DataFrame(data, index=all_countries, columns=res_techs)

# Step 3: Run clustering
def cluster_and_plot(df, n_clusters=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X_scaled)

    df['Cluster'] = labels

    sns.clustermap(df.drop(columns='Cluster'), cmap='viridis', standard_scale=1)
    plt.title("Country Clustering by RES Capacity Factors")
    plt.show()

    return df

# --- Example usage ---
res_techs = ['solar', 'onwind', 'offwind']
all_countries = ['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR', 'HR', 'HU',
                 'IE', 'IT', 'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK', 'XK']

path = 'C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/results_8h_juno/policy_eu_regain/networks/base_s_39___2050.nc'
network = pypsa.Network(path)

cf_data = get_capacity_factors(network, res_techs, all_countries)
df = make_feature_matrix(cf_data, all_countries, res_techs)
clustered_df = cluster_and_plot(df, n_clusters=4)


def classify_and_sort_countries(cf_data, solar_threshold=0.1, wind_threshold=0.2):
    high_solar = []
    high_wind = []
    low_res = []

    all_countries = set()
    for tech_cf in cf_data.values():
        all_countries.update(tech_cf.keys())

    for country in sorted(all_countries):
        solar_cf = cf_data.get('solar', {}).get(country, 0.0)
        onwind_cf = cf_data.get('onwind', {}).get(country, 0.0)
        offwind_cf = cf_data.get('offwind', {}).get(country, 0.0)
        wind_cf = max(onwind_cf, offwind_cf)

        if solar_cf > solar_threshold:
            high_solar.append((country, solar_cf))
        elif wind_cf > wind_threshold:
            high_wind.append((country, wind_cf))
        else:
            low_cf = max(solar_cf, wind_cf)
            low_res.append((country, low_cf))

    # --- Force DK and GB into high wind ---
    forced_high_wind = ['DK', 'GB']
    for c in forced_high_wind:
        wind_cf = max(
            cf_data.get('onwind', {}).get(c, 0.0),
            cf_data.get('offwind', {}).get(c, 0.0)
        )
        # Remove from other groups if present
        high_solar = [(cc, cf) for cc, cf in high_solar if cc != c]
        low_res = [(cc, cf) for cc, cf in low_res if cc != c]
        high_wind = [(cc, cf) for cc, cf in high_wind if cc != c]
        high_wind.append((c, wind_cf))

    # Sort groups
    high_solar.sort(key=lambda x: -x[1])
    high_wind.sort(key=lambda x: -x[1])
    low_res.sort(key=lambda x: -x[1])

    # Format into columns
    group_dict = {
        f'High Solar CF > {solar_threshold}': [f"{c} ({cf:.2f})" for c, cf in high_solar],
        f'High Wind CF > {wind_threshold}': [f"{c} ({cf:.2f})" for c, cf in high_wind],
        f'Low RES CF ≤ {max(solar_threshold, wind_threshold)}': [f"{c} ({cf:.2f})" for c, cf in low_res],
    }

    max_len = max(len(v) for v in group_dict.values())
    padded_dict = {k: v + [None] * (max_len - len(v)) for k, v in group_dict.items()}

    df_summary = pd.DataFrame(padded_dict)
    df_summary.to_csv("res_capacity_factor_clusters.csv", index=False)

    return df_summary



# Example usage
df_res_groups = classify_and_sort_countries(cf_data)
print(df_res_groups)
