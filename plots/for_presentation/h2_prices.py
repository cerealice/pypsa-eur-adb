# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:32:42 2025

@author: Dibella
"""

import matplotlib.pyplot as plt
import pypsa
import pandas as pd
import numpy as np
import seaborn as sns

# --- Country and Scenario Setup ---

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

scenario_colors = {
    "base_eu_regain": "#464E47",
    "policy_eu_regain": "#00B050",
    "policy_eu_deindustrial": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}
scenario_labels = {
    "base_eu_regain": "NO CLIMATE POLICY\nCompetive industry\nRelocation",
    "policy_eu_regain": "CLIMATE POLICY\nCompetive industry\nRelocation",
    "policy_eu_deindustrial": "Climate policy\nDEINDUSTRIALIZATION\nRelocation",
    "policy_reg_regain": "Climate policy\nCompetive industry\nHISTORICAL HUBS"
}

scenarios = ["policy_eu_regain", "base_eu_regain", "policy_eu_deindustrial", "policy_reg_regain", ]

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"

# --- Load networks ---

def load_networks(scenarios, years, root_dir, res_dir):
    networks = {}
    for scenario in scenarios:
        for year in years:
            path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
            networks[(scenario, year)] = pypsa.Network(path)
    return networks

# --- Compute weighted average hydrogen prices ---
def compute_weighted_h2_prices(networks, threshold_ratio=0.05):
    result = {}
    for (scenario, year), n in networks.items():
        h2_loads = [load for load in n.loads.index if 'H2' in load]
        if not h2_loads:
            continue

        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice = mprice.loc[:, mprice.columns.str.endswith("H2")]
        mprice.columns = mprice.columns.str[:2]
        mprice = mprice.T.groupby(level=0).sum().T     

        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads = loads.T.groupby(level=0).sum().T

        common_cols = mprice.columns.intersection(loads.columns)
        if common_cols.empty:
            continue

        mprice = mprice[common_cols]
        loads = loads[common_cols]

        total_costs = mprice * loads
        grouped_costs = total_costs.sum(axis=0)
        grouped_loads = loads.sum(axis=0)

        max_load = grouped_loads.max()
        threshold = max_load * threshold_ratio

        weighted_avg_price = grouped_costs / grouped_loads
        weighted_avg_price[grouped_loads < threshold] = float('nan')

        if scenario not in result:
            result[scenario] = pd.DataFrame()

        weighted_avg_price_df = pd.DataFrame(weighted_avg_price).T
        weighted_avg_price_df.index = [year]

        result[scenario] = pd.concat([result[scenario], weighted_avg_price_df])

    return result



# --- Compute weighted min hydrogen prices ---
def compute_weighted_h2_min_prices(networks, threshold_ratio=0.05):
    result = {}
    for (scenario, year), n in networks.items():
        h2_loads = [load for load in n.loads.index if 'H2' in load]
        if not h2_loads:
            continue

        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice = mprice.loc[:, mprice.columns.str.endswith("H2")]
        mprice.columns = mprice.columns.str[:2]
        mprice = mprice.T.groupby(level=0).sum().T

        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads = loads.T.groupby(level=0).sum().T

        common_cols = mprice.columns.intersection(loads.columns)
        if common_cols.empty:
            continue

        mprice = mprice[common_cols]
        loads = loads[common_cols]

        total_loads = loads.sum()
        max_load = total_loads.max()
        threshold = max_load * threshold_ratio

        weighted_min_price_by_country = []
        for country in common_cols:
            ld = loads[country]
            mp = mprice[country]
            if ld.sum() < threshold:
                weighted_min_price_by_country.append(float('nan'))
            else:
                weighted_min_price_by_country.append((mp * ld).sum() / ld.sum())

        weighted_min_price_by_country = pd.Series(weighted_min_price_by_country, index=common_cols)

        if scenario not in result:
            result[scenario] = pd.DataFrame()

        weighted_min_price_df = pd.DataFrame(weighted_min_price_by_country).T
        weighted_min_price_df.index = [year]

        result[scenario] = pd.concat([result[scenario], weighted_min_price_df])
    return result



# --- Compute weighted max hydrogen prices ---
def compute_weighted_h2_max_prices(networks, threshold_ratio=0.05):
    result = {}
    for (scenario, year), n in networks.items():
        h2_loads = [load for load in n.loads.index if 'H2' in load]
        if not h2_loads:
            continue

        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice = mprice.loc[:, mprice.columns.str.endswith("H2")]
        mprice.columns = mprice.columns.str[:2]
        mprice = mprice.T.groupby(level=0).sum().T

        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads = loads.T.groupby(level=0).sum().T

        common_cols = mprice.columns.intersection(loads.columns)
        if common_cols.empty:
            continue

        mprice = mprice[common_cols]
        loads = loads[common_cols]

        total_loads = loads.sum()
        max_load = total_loads.max()
        threshold = max_load * threshold_ratio

        weighted_max_price_by_country = []
        for country in common_cols:
            ld = loads[country]
            mp = mprice[country]
            if ld.sum() < threshold:
                weighted_max_price_by_country.append(float('nan'))
            else:
                weighted_max_price_by_country.append((mp * ld).sum() / ld.sum())

        weighted_max_price_by_country = pd.Series(weighted_max_price_by_country, index=common_cols)

        if scenario not in result:
            result[scenario] = pd.DataFrame()

        weighted_max_price_df = pd.DataFrame(weighted_max_price_by_country).T
        weighted_max_price_df.index = [year]

        result[scenario] = pd.concat([result[scenario], weighted_max_price_df])
    return result

def compute_total_h2_production(networks):
    result = {}  # DataFrames keyed by scenario

    for (scenario, year), n in networks.items():
        # Filter H2 links (same as before)
        loads_links = n.links[n.links.bus1.str.endswith(' H2') & ~n.links.index.str.contains('pipeline')]
        if loads_links.empty:
            continue

        # Extract production (negative flow from links_t.p1)
        loads = -n.links_t.p1.loc[:, loads_links.index]
        loads.columns = loads.columns.str[:2]
        loads = loads.T.groupby(level=0).sum().T

        if scenario not in result:
            result[scenario] = pd.DataFrame()

        # Sum production per country over time for each year
        total_production = loads.sum(axis=0)
        total_production_df = pd.DataFrame(total_production).T
        total_production_df.index = [year]

        result[scenario] = pd.concat([result[scenario], total_production_df])

    return result


# --- Main execution ---
# %%
networks = load_networks(scenarios, years, root_dir, res_dir)
# %%
weigh_aver_h2 = compute_weighted_h2_prices(networks)
weigh_min_h2 = compute_weighted_h2_min_prices(networks)
weigh_max_h2 = compute_weighted_h2_max_prices(networks)

# %%


ncols = 4

# Colors for scenarios
scenario_colors = {
    "base_eu_regain": "#464E47",
    "policy_eu_regain": "#00B050",
    "policy_eu_deindustrial": "#FF92D4",
    "policy_reg_regain": "#3AAED8"
}

# Collect all countries present in the data
all_countries = sorted({c for df in weigh_aver_h2.values() for c in df.columns})
nrows = int(np.ceil(len(all_countries) / ncols))

for scenario in weigh_aver_h2.keys():
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=False)
    axes = axes.flatten()

    for idx, country in enumerate(all_countries):
        ax = axes[idx]
        if country not in weigh_aver_h2[scenario].columns:
            ax.axis("off")
            continue

        avg = weigh_aver_h2[scenario][country]
        min_ = weigh_min_h2[scenario][country]
        max_ = weigh_max_h2[scenario][country]

        color = scenario_colors.get(scenario, "gray")
        ax.plot(avg.index, avg.values, label="Avg", color=color, linewidth=2)
        ax.fill_between(avg.index, min_.values, max_.values, color=color, alpha=0.7, label="Min-Max")

        ax.set_title(country)
        ax.grid(True, linestyle="--")
        if idx % ncols == 0:
            ax.set_ylabel("H₂ Price [€/MWh]")
        if idx >= len(all_countries) - ncols:
            ax.set_xlabel("Year")
        ax.set_ylim(bottom=0)

    # Remove unused subplots
    for i in range(len(all_countries), len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"Hydrogen Prices — {scenario.replace('_', ' ').title()}", fontsize=14)
    fig.legend(loc='upper center', ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"./graphs/h2_price_bands_{scenario}.png", dpi=300)
    plt.show()


# %%


fig, axes = plt.subplots(1, len(scenario_colors), figsize=(18, 5), sharey=False)

for idx, (scenario, ax) in enumerate(zip(scenario_colors.keys(), axes)):

    avg_df = weigh_aver_h2.get(scenario, pd.DataFrame())
    min_df = weigh_min_h2.get(scenario, pd.DataFrame())
    max_df = weigh_max_h2.get(scenario, pd.DataFrame())

    if avg_df.empty or min_df.empty or max_df.empty:
        continue

    for region, countries in group_countries.items():
        valid_countries = [c for c in countries if c in avg_df.columns]
        if not valid_countries:
            continue

        avg = avg_df[valid_countries].mean(axis=1)
        min_ = min_df[valid_countries].min(axis=1)
        max_ = max_df[valid_countries].max(axis=1)

        ax.plot(avg.index, avg.values, label=region, color=custom_colors[region], linewidth=2)

        ax.fill_between(avg.index, min_.values, max_.values, color=custom_colors[region], alpha=0.3)

    ax.set_title(scenario_labels[scenario], fontsize=10)
    #ax.set_ylim(0, 80)
    ax.set_xticks(years)
    if idx == 0:
        ax.set_ylabel("Hydrogen Price [€/MWh]")
    ax.grid(True, linestyle="--")
    ax.set_ylim(bottom=0)

# Legend outside
handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper center', ncol=3, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("./graphs/h2_price_bands_by_scenario.png", dpi=300)
plt.show()


# %%

fig, axes = plt.subplots(1, len(scenario_colors), figsize=(18, 5), sharey=True)

for idx, (scenario, ax) in enumerate(zip(scenario_colors.keys(), axes)):

    avg_df = weigh_aver_h2.get(scenario, pd.DataFrame())
    if avg_df.empty:
        continue

    xtick_labels = []
    box_data = []
    outlier_labels = []

    for i, year in enumerate(years):
        if year not in avg_df.index:
            continue

        valid_countries = [
            c for region_countries in group_countries.values() for c in region_countries if c in avg_df.columns
        ]
        data_series = avg_df.loc[year, valid_countries].dropna()

        values = data_series.values
        countries = data_series.index

        box_data.append(values)
        xtick_labels.append(str(year))
        outlier_labels.append(countries)

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        labels=xtick_labels,
        medianprops=dict(color="black"),
        boxprops=dict(facecolor=scenario_colors[scenario], alpha=0.5),
        whiskerprops=dict(color="gray"),
        capprops=dict(color="gray"),
        flierprops=dict(markerfacecolor="red", marker="o", markersize=5, alpha=0.6)
    )

    # Annotate outliers with region names (once per region per year)
    for year_idx, (flier, countries) in enumerate(zip(bp["fliers"], outlier_labels)):
        y_outliers = flier.get_ydata()
        x_outliers = flier.get_xdata()
        year_values = box_data[year_idx]
        year_countries = countries

        annotated_regions = set()  # Track which regions already labeled for this year

        for x, y in zip(x_outliers, y_outliers):
            try:
                idx_matches = np.where(np.isclose(year_values, y, rtol=0.01))[0]
                for idx_val in idx_matches:
                    country = year_countries[idx_val]

                    # Determine region for the outlier country
                    region_name = next(
                        (r for r, cs in group_countries.items() if country in cs),
                        None
                    )
                    if region_name and region_name not in annotated_regions:
                        ax.annotate(
                            region_name,
                            (x, y),
                            textcoords="offset points",
                            xytext=(5, 5),
                            ha='left',
                            fontsize=8,
                            rotation=30
                        )
                        annotated_regions.add(region_name)  # Prevent repeat annotations
                        break  # Annotate only once per outlier
            except Exception:
                continue

    ax.set_title(scenario_labels[scenario], fontsize=10)
    ax.set_xlabel("Year")
    if idx == 0:
        ax.set_ylabel("Hydrogen Price [€/MWh]")
    ax.grid(True, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./graphs/h2_boxplot_regions_outliers_labeled.png", dpi=300)
plt.show()

# %%


# Prepare figure
fig, axes = plt.subplots(1, len(scenario_colors), figsize=(18, 5), sharey=True)

# Loop through scenarios
for idx, (scenario, ax) in enumerate(zip(scenario_colors.keys(), axes)):
    avg_df = weigh_aver_h2.get(scenario, pd.DataFrame())
    if avg_df.empty:
        continue

    # Prepare long-format DataFrame for seaborn violinplot
    records = []
    for year in years:
        if year not in avg_df.index:
            continue
        for country in avg_df.columns:
            value = avg_df.loc[year, country]
            if pd.notna(value):
                region = next((r for r, cs in group_countries.items() if country in cs), None)
                records.append({
                    "Year": year,
                    "Country": country,
                    "Region": region,
                    "Value": value
                })

    df_long = pd.DataFrame(records)

    # Plot violin plot
    sns.violinplot(
        data=df_long,
        x="Year",
        y="Value",
        ax=ax,
        palette=[scenario_colors[scenario]] * len(years),
        inner=None,
        linewidth=1,
        cut=0
    )

    # Overlay box plot to show medians
    sns.boxplot(
        data=df_long,
        x="Year",
        y="Value",
        ax=ax,
        color="white",
        fliersize=0,
        linewidth=1
    )

    # Annotate outliers (detected per year)
    for year in years:
        year_data = df_long[df_long["Year"] == year]
        values = year_data["Value"]
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = year_data[(values < lower_bound) | (values > upper_bound)]

        # Track annotated regions to avoid repeating
        annotated_regions = set()

        for _, row in outliers.iterrows():
            region = row["Region"]
            if region and region not in annotated_regions:
                ax.annotate(
                    region,
                    (year, row["Value"]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=8,
                    rotation=30
                )
                annotated_regions.add(region)

    # Axis formatting
    ax.set_title(scenario_labels[scenario], fontsize=10)
    if idx == 0:
        ax.set_ylabel("Hydrogen Price [€/MWh]")
    else:
        ax.set_ylabel("")
    ax.grid(True, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("./graphs/h2_violinplot_regions_outliers_labeled.png", dpi=300)
plt.show()


# %%

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# --- Cluster countries by hydrogen price ---
def cluster_countries_by_price(weigh_aver_h2, n_clusters=2):
    cluster_results = {}  # {(scenario, year): DataFrame with country & cluster}

    for scenario, df in weigh_aver_h2.items():
        for year in df.index:
            data = df.loc[year].dropna().values.reshape(-1, 1)
            countries = df.columns[df.loc[year].notna()]

            if len(countries) < n_clusters:
                continue  # Not enough countries for clustering

            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)

            # Assign cluster labels to countries
            cluster_df = pd.DataFrame({
                "Country": countries,
                "Cluster": labels,
                "Price": df.loc[year, countries].values
            })

            # Optional: sort so that Cluster 0 is always the cheaper one
            cluster_means = cluster_df.groupby("Cluster")["Price"].mean()
            cheap_cluster = cluster_means.idxmin()
            cluster_df["Cluster"] = cluster_df["Cluster"].apply(lambda x: 0 if x == cheap_cluster else 1)

            cluster_results[(scenario, year)] = cluster_df

    return cluster_results

def plot_clusters_subplots(cluster_results, scenarios, years):
    fig, axes = plt.subplots(len(scenarios), len(years), figsize=(5 * len(years), 4 * len(scenarios)), sharey=True)

    for i, scenario in enumerate(scenarios):
        for j, year in enumerate(years):
            ax = axes[i, j] if len(scenarios) > 1 else axes[j]  # handle 1D cases
            key = (scenario, year)

            if key in cluster_results:
                df = cluster_results[key].sort_values("Price")

                sns.barplot(data=df, x="Country", y="Price", hue="Cluster", palette="Set2", ax=ax)
                ax.set_title(f"{scenario_labels[scenario]}\n{year}", fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_xlabel("")
                ax.set_ylabel("H₂ Price [€/MWh]" if j == 0 else "")
                if j != 0:
                    ax.get_legend().remove()
            else:
                ax.set_visible(False)

    # Only show the legend once
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Cluster (0=Low, 1=High)", loc="upper center", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Hydrogen Price Clustering by Country", fontsize=14)
    plt.show()


# --- Run clustering and plot ---
cluster_results = cluster_countries_by_price(weigh_aver_h2, n_clusters=2)
plot_clusters_subplots(cluster_results, scenarios, years)


# %%

# Rename for clarity
def cluster_countries_by_production(total_production, n_clusters=2):
    cluster_results = {}

    for scenario, df in total_production.items():
        for year in df.index:
            data = df.loc[year].dropna().values.reshape(-1, 1)
            countries = df.columns[df.loc[year].notna()]

            if len(countries) < n_clusters:
                continue

            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data)

            cluster_df = pd.DataFrame({
                "Country": countries,
                "Cluster": labels,
                "Production": df.loc[year, countries].values
            })

            # Optional: cluster 0 = highest production
            cluster_means = cluster_df.groupby("Cluster")["Production"].mean()
            high_cluster = cluster_means.idxmax()
            cluster_df["Cluster"] = cluster_df["Cluster"].apply(lambda x: 0 if x == high_cluster else 1)

            cluster_results[(scenario, year)] = cluster_df

    return cluster_results


def plot_production_clusters_subplots(cluster_results, scenarios, years):
    fig, axes = plt.subplots(len(scenarios), len(years), figsize=(5 * len(years), 4 * len(scenarios)), sharey=False)

    for i, scenario in enumerate(scenarios):
        for j, year in enumerate(years):
            ax = axes[i, j] if len(scenarios) > 1 else axes[j]
            key = (scenario, year)

            if key in cluster_results:
                df = cluster_results[key].sort_values("Production")

                sns.barplot(data=df, x="Country", y="Production", hue="Cluster", palette="Set2", ax=ax)
                ax.set_title(f"{scenario_labels[scenario]}\n{year}", fontsize=10)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_xlabel("")
                ax.set_ylabel("H₂ Production [MWh]" if j == 0 else "")
                if j != 0:
                    ax.get_legend().remove()
            else:
                ax.set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Cluster (0=High, 1=Low)", loc="upper center", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Hydrogen Production Clustering by Country", fontsize=14)
    plt.show()


total_production = compute_total_h2_production(networks)
cluster_results_production = cluster_countries_by_production(total_production, n_clusters=2)
plot_production_clusters_subplots(cluster_results_production, scenarios, years)
