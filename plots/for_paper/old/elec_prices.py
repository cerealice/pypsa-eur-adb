# -*- coding: utf-8 -*-
"""
Created on Tue Jul 3, 2025
Script for Europe-wide electricity price boxplots per scenario
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pypsa
import pandas as pd
import numpy as np
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms

# --- Setup ---

scenarios = [
    "base_reg_deindustrial",
    "policy_reg_deindustrial",
    "base_reg_regain",
    "policy_reg_regain"
]

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

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"



class QuadraticScale(mscale.ScaleBase):
    name = 'quadratic'

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        
    def get_transform(self):
        return self.QuadraticTransform()

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(plt.MaxNLocator(6))
        axis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(0, vmin), vmax

    class QuadraticTransform(mtransforms.Transform):
        input_dims = output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.power(a, 2)

        def inverted(self):
            return QuadraticScale.InvertedQuadraticTransform()

    class InvertedQuadraticTransform(mtransforms.Transform):
        input_dims = output_dims = 1
        is_separable = True

        def transform_non_affine(self, a):
            return np.sqrt(a)

        def inverted(self):
            return QuadraticScale.QuadraticTransform()

# Register the scale so matplotlib can find it
mscale.register_scale(QuadraticScale)

def compute_weighted_green_share_europe(n) -> float:
    """
    Computes the Europe-wide weighted green electricity share [%], weighted by each country's total electricity production.

    Parameters:
        n: PyPSA network
        timestep: float (e.g. 1.0 for hourly resolution)
        countries: list of country codes to include in weighting.

    Returns:
        Scalar float [%] representing Europe-wide green electricity share weighted by production.
    """
    # Compute generation from different assets
    timestep = n.snapshot_weightings.iloc[0,0]
    generation_raw = n.generators_t.p.sum() * timestep * 1e-6  # TWh
    generation_raw = generation_raw[~generation_raw.index.str.contains('|'.join(['EU','thermal']), case=False, na=False)]

    hydro_generation = n.storage_units_t.p.sum() * timestep * 1e-6  # TWh
    hydro_generation = hydro_generation[~hydro_generation.index.str.contains('PHS')]

    tot_prod_links = -n.links_t.p1.sum() * timestep * 1e-6
    elec_links_name = ['OCGT', 'coal-', 'CCGT', 'CHP']
    elec_links = tot_prod_links[tot_prod_links.index.str.contains('|'.join(elec_links_name), case=False, na=False)]

    generation = pd.concat([generation_raw, hydro_generation, elec_links])

    # Parse generation index
    generation = (pd.concat([
        generation.index.str.extract(r'([A-Z][A-Z]\d?)', expand=True),          # Node
        generation.index.str.extract(r'([A-Z][A-Z])', expand=True),            # Country
        generation.index.str.extract(r'[A-Z][A-Z]\d? ?\d? (.+)', expand=True),  # Source type
        generation.reset_index()
    ], ignore_index=True, axis=1)
    .drop(3, axis=1)
    .rename(columns={0: 'Node', 1: 'Country', 2: 'Source type', 4: 'Generation [TWh/yr]'}))

    generation = generation[generation['Generation [TWh/yr]'] > 1e-6]
    generation['Source type'] = generation['Source type'].str.split('-').str[0].str.strip().str.title()
    generation['Source type'] = generation['Source type'].replace({'Ccgt': 'CCGT', 'Ocgt': 'OCGT'})

    # Compute green generation
    green_sources = ['Hydro', 'Ror', 'Solar', 'Wind', 'Biomass','Nuclear']
    is_green = generation['Source type'].str.contains('|'.join(green_sources), case=False)

    gen = generation.groupby('Country')['Generation [TWh/yr]'].sum().sum()
    green_gen = generation[is_green].groupby('Country')['Generation [TWh/yr]'].sum().sum()

    return (green_gen/gen) * 100


# %%


# --- Load Networks ---

networks = {}
for scenario in scenarios:
    for year in years:
        path = f"{root_dir}{res_dir}{scenario}/networks/base_s_39___{year}.nc"
        networks[(scenario, year)] = pypsa.Network(path)

# %%
# --- Compute Europe-wide Weighted Electricity Price Time Series ---

all_data = []

for scenario in scenarios:
    for year in years:
        n = networks[(scenario, year)]
        timestep_hours = pd.to_timedelta(n.snapshot_weightings.generators, unit='h').values[0]
        
        # Marginal prices
        mprice = n.buses_t.marginal_price.clip(lower=0)
        # Filter to loads
        mprice_loads = mprice[mprice.columns.intersection(n.loads.index)]
        loads = n.loads_t.p[n.loads_t.p.columns.intersection(mprice_loads.columns)]
        
        # Select electricity buses only
        elec_mask = mprice_loads.columns.str.endswith(" 0")
        mprice_elec = mprice_loads.loc[:, elec_mask]
        loads_elec = loads.loc[:, elec_mask]
        
        # Rename by country code (first two letters)
        mprice_elec.columns = mprice_elec.columns.str[:2]
        loads_elec.columns = loads_elec.columns.str[:2]
        
        # Group by country
        mprice_grouped = mprice_elec.groupby(axis=1, level=0).mean()
        loads_grouped = loads_elec.groupby(axis=1, level=0).sum()

        # Compute Europe-wide load-weighted price at each timestep
        total_load = loads_grouped.sum(axis=1)
        weighted_price = (mprice_grouped * loads_grouped).sum(axis=1) / total_load

        # Store results
        df_tmp = pd.DataFrame({
            "Electricity Price [€/MWh]": weighted_price.values,
            "Year": year,
            "Scenario": scenario_labels[scenario],
        })
        all_data.append(df_tmp)

# Combine all into single DataFrame
df_all = pd.concat(all_data, ignore_index=True)

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

for ax, scenario_label in zip(axes, scenario_labels.values()):
    df_scenario = df_all[df_all["Scenario"] == scenario_label].copy()
    
    # Apply sqrt transform to y-values for quadratic-like scale
    df_scenario["Transformed Price"] = np.sqrt(df_scenario["Electricity Price [€/MWh]"])
    
    sns.boxplot(
        x="Year",
        y="Transformed Price",
        data=df_scenario,
        ax=ax,
        palette="pastel",
        width=0.6,
        showcaps=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 4},
        boxprops={"edgecolor": "black", "facecolor": "lightgray"},
        medianprops={"color": "red", "linewidth": 2},
    )
    ax.set_title(scenario_label, fontsize=12)

    # Customize y-ticks: show squared values instead of sqrt
    yticks = ax.get_yticks()
    ax.set_yticklabels([f"{int(tick**2)}" if tick >= 0 else "" for tick in yticks])
    
    if ax == axes[0]:
        ax.set_ylabel("Europe-wide Electricity Price [€/MWh]", fontsize=11)
    else:
        ax.set_ylabel("")

plt.suptitle("Distribution of Europe-wide Electricity Prices per Scenario\n(Quadratic Y-axis scale)", fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig("graphs/europe_price_boxplot_by_scenario_quadratic.png", dpi=300)
plt.show()

# %%


# Filter only the first two scenarios
selected_scenarios = scenarios[:2]
selected_labels = [scenario_labels[s] for s in selected_scenarios]
selected_colors = [scenario_colors[s] for s in selected_scenarios]

# Filter dataset
df_selected = df_all[df_all["Scenario"].isin(selected_labels)].copy()

# Apply sqrt transform to prices (for quadratic-like scale)
df_selected["Transformed Price"] = np.sqrt(df_selected["Electricity Price [€/MWh]"])

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Plot boxplots with facecolor matching the scenario color
sns.boxplot(
    x="Year",
    y="Transformed Price",
    hue="Scenario",
    data=df_selected,
    palette={label: color for label, color in zip(selected_labels, selected_colors)},
    width=0.6,
    showcaps=True,
    showfliers=True,
    flierprops={"marker": "o", "markersize": 4},
    boxprops={"edgecolor": "black"},
    medianprops={"color": "red", "linewidth": 2},
)

#plt.title("Distribution of Europe-wide Electricity Prices (Quadratic Y-axis Scale)", fontsize=15)

# Customize y-ticks to show squared (real) values
yticks = plt.gca().get_yticks()
plt.gca().set_yticklabels([f"{int(tick**2)}" if tick >= 0 else "" for tick in yticks])

plt.ylabel("European space-averaged electricity price [€/MWh]", fontsize=12)
plt.xlabel("", fontsize=12)
plt.legend(title="Scenario", fontsize=10)
plt.tight_layout()
plt.savefig("graphs/europe_price_boxplot_by_scenario_quadratic_single.png", dpi=300)
plt.show()

# %%
# --- Compute Europe-wide Weighted Electricity Price Time Series + Green Electricity Share ---

all_data = []
green_share_data = {}

for scenario in scenarios:
    for year in years:
        n = networks[(scenario, year)]
        timestep_hours = pd.to_timedelta(n.snapshot_weightings.generators, unit='h').values[0]
        
        # Marginal prices
        mprice = n.buses_t.marginal_price.clip(lower=0)
        mprice_loads = mprice[mprice.columns.intersection(n.loads.index)]
        loads = n.loads_t.p[n.loads_t.p.columns.intersection(mprice_loads.columns)]
        
        elec_mask = mprice_loads.columns.str.endswith(" 0")
        mprice_elec = mprice_loads.loc[:, elec_mask]
        loads_elec = loads.loc[:, elec_mask]
        
        mprice_elec.columns = mprice_elec.columns.str[:2]
        loads_elec.columns = loads_elec.columns.str[:2]
        
        mprice_grouped = mprice_elec.groupby(axis=1, level=0).mean()
        loads_grouped = loads_elec.groupby(axis=1, level=0).sum()

        total_load = loads_grouped.sum(axis=1)
        weighted_price = (mprice_grouped * loads_grouped).sum(axis=1) / total_load

        # Store results
        df_tmp = pd.DataFrame({
            "Electricity Price [€/MWh]": weighted_price.values,
            "Year": year,
            "Scenario": scenario_labels[scenario],
        })
        all_data.append(df_tmp)
        
        # Compute green electricity share (YOUR FUNCTION)
        green_share = compute_weighted_green_share_europe(n)  # Assuming it returns scalar percentage (0-100)
        green_share_data[(scenario_labels[scenario], year)] = green_share

# Convert green_share_data dict to DataFrame
green_share_df = pd.DataFrame([
    {
        "Scenario": scenario,
        "Year": year,
        "Green Share [%]": green_share
    }
    for (scenario, year), green_share in green_share_data.items()
])

# Sort by Year and Scenario for clean plotting
green_share_df = green_share_df.sort_values(by=["Year", "Scenario"]).reset_index(drop=True)

# Assume green_share_df has columns: ["Year", "Scenario", "Green Share [%]"]
green_share_df_selected = green_share_df[green_share_df["Scenario"].isin(selected_labels)]

# First: Ensure Year is numeric in both DataFrames
df_selected["Year"] = df_selected["Year"].astype(int)
green_share_df_selected["Year"] = green_share_df_selected["Year"].astype(int)

# Prepare numeric positions for years
years_sorted = sorted(df_selected["Year"].unique())
year_pos = {year: pos for pos, year in enumerate(years_sorted)}
df_selected["Year_Pos"] = df_selected["Year"].map(year_pos)
green_share_df_selected["Year_Pos"] = green_share_df_selected["Year"].map(year_pos)
# ... (keep the setup code before plotting as is) ...

sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Primary axis: Boxplot (using numeric year positions)
ax1 = sns.boxplot(
    x="Year_Pos",
    y="Transformed Price",
    hue="Scenario",
    data=df_selected,
    palette={label: color for label, color in zip(selected_labels, selected_colors)},
    width=0.6,
    showcaps=True,
    showfliers=True,
    flierprops={"marker": "o", "markersize": 4},
    boxprops={"edgecolor": "black"},
    medianprops={"color": "red", "linewidth": 2},
)

# Set x-ticks and labels for years
ax1.set_xticks(list(year_pos.values()))
ax1.set_xticklabels([str(year) for year in years_sorted])

# Customize y-ticks to show squared (real) prices
yticks = ax1.get_yticks()
ax1.set_yticklabels([f"{int(tick**2)}" if tick >= 0 else "" for tick in yticks])

ax1.set_ylabel("European space-averaged electricity price [€/MWh]", fontsize=12)
ax1.set_xlabel("Year", fontsize=12)

# Secondary axis: Green electricity share (dots only)
ax2 = ax1.twinx()

for scenario, color in zip(selected_labels, selected_colors):
    df_plot = green_share_df_selected[green_share_df_selected["Scenario"] == scenario]
    x = df_plot["Year_Pos"].values
    y = df_plot["Green Share [%]"].values
    
    # Plot dots only
    ax2.scatter(x, y, color=color, label=f"Green Share - {scenario}", s=50, zorder=10)
    
    # Annotate each dot with the percentage value
    for xi, yi in zip(x, y):
        ax2.text(xi, yi + 0.25, f"{yi:.1f}%", color=color, fontsize=9, ha='center')

ax2.set_ylabel("Green electricity share [%]", fontsize=12)

# Combine legends from both axes
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(
    h1 + h2,
    l1 + l2,
    loc="upper left",
    fontsize=10,
    title="Scenario & Metric"
)

plt.tight_layout()
plt.savefig("graphs/europe_price_and_green_share_dots_only.png", dpi=300)
plt.show()


# %%

"""
# %%

import matplotlib.dates as mdates

# Filter for year 2050 only
df_2050 = df_all[df_all["Year"] == 2050].copy()

fig, ax = plt.subplots(figsize=(16,6))

for scenario in scenarios:
    label = scenario_labels[scenario]
    # Get network for (scenario, 2050) to retrieve timestamps
    n = networks[(scenario, 2050)]

    # Marginal prices and loads as before, but recompute weighted price per snapshot for plotting
    mprice = n.buses_t.marginal_price.clip(lower=0)
    mprice_loads = mprice[mprice.columns.intersection(n.loads.index)]
    loads = n.loads_t.p[n.loads_t.p.columns.intersection(mprice_loads.columns)]

    elec_mask = mprice_loads.columns.str.endswith(" 0")
    mprice_elec = mprice_loads.loc[:, elec_mask]
    loads_elec = loads.loc[:, elec_mask]

    mprice_elec.columns = mprice_elec.columns.str[:2]
    loads_elec.columns = loads_elec.columns.str[:2]

    mprice_grouped = mprice_elec.groupby(axis=1, level=0).mean()
    loads_grouped = loads_elec.groupby(axis=1, level=0).sum()

    total_load = loads_grouped.sum(axis=1)
    weighted_price = (mprice_grouped * loads_grouped).sum(axis=1) / total_load

    # Plot timeseries
    ax.plot(n.snapshots, weighted_price, label=label, color=scenario_colors[scenario])

ax.set_title("Europe-wide Weighted Electricity Prices in 2050", fontsize=16)
ax.set_xlabel("Time")
ax.set_ylabel("Electricity Price [€/MWh]")
ax.legend()
ax.grid(True)

# Optional: format x-axis nicely if timestamps are datetime
if isinstance(n.snapshots[0], pd.Timestamp):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout()
plt.show()

# %%

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, len(years), figsize=(18, 6), sharey=True)

for ax, year in zip(axes, years):
    df_year = df_all[df_all["Year"] == year]

    for scenario in scenarios:
        label = scenario_labels[scenario]
        color = scenario_colors[scenario]

        data = df_year[df_year["Scenario"] == label]["Electricity Price [€/MWh]"]

        sns.kdeplot(
            data,
            ax=ax,
            label=label,
            color=color,
            linewidth=2,
            fill=True,
            alpha=0.3,
        )

    ax.set_title(f"Year {year}", fontsize=14)
    ax.set_xlabel("Electricity Price [€/MWh]")
    if ax == axes[0]:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("")
    ax.grid(True)
    ax.legend(fontsize=9)

plt.suptitle("Distribution Shapes of Europe-wide Electricity Prices by Scenario", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%

# Map scenario_labels (values) to colors in scenario_colors by keys
palette = {scenario_labels[k]: v for k, v in scenario_colors.items()}

sns.set(style="whitegrid")

g = sns.FacetGrid(df_all, row="Year", hue="Scenario", aspect=4, height=2, palette=palette, sharex=True)

g.map(sns.kdeplot, "Electricity Price [€/MWh]", fill=True, alpha=0.7, linewidth=1.5)

g.add_legend(title="Scenario")

g.set_titles(row_template = "Year: {row_name}")
g.set_xlabels("Europe-wide Electricity Price [€/MWh]")
g.set(yticks=[])

plt.subplots_adjust(hspace=0.4)
plt.suptitle("Ridge Plot of Europe-wide Electricity Prices by Year and Scenario", y=1.02, fontsize=16)

plt.show()

# %%

sns.set_style("whitegrid")
fig, axes = plt.subplots(1, len(years), figsize=(18, 6), sharey=True)

for ax, year in zip(axes, years):
    df_year = df_all[df_all["Year"] == year]

    for scenario in scenarios:
        label = scenario_labels[scenario]
        color = scenario_colors[scenario]

        data = df_year[df_year["Scenario"] == label]["Electricity Price [€/MWh]"]

        sns.kdeplot(
            data,
            ax=ax,
            label=label,
            color=color,
            linewidth=2,
            fill=True,
            alpha=0.3,
        )

    ax.set_title(f"Year {year}", fontsize=14)
    ax.set_xlabel("Electricity Price [€/MWh]")
    if ax == axes[0]:
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("")
    ax.grid(True)
    ax.legend(fontsize=9)

    # Apply quadratic scale to x-axis
    ax.set_xscale('quadratic')

plt.suptitle("Distribution Shapes of Europe-wide Electricity Prices by Scenario", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%

import matplotlib.pyplot as plt

# Filter for the year 2050 only
df_2050 = df_all[df_all["Year"] == 2050]

# Scenarios to plot (first two)
scenarios_to_plot = scenarios[:2]

plt.figure(figsize=(8, 6))

for scenario in scenarios_to_plot:
    label = scenario_labels[scenario]
    color = scenario_colors[scenario]

    # Extract prices for the scenario, round to int
    prices = df_2050[df_2050["Scenario"] == label]["Electricity Price [€/MWh]"].round().astype(int)

    # Count occurrences of each price value
    counts = prices.value_counts().sort_index()

    # Scatter plot: counts on x, prices on y
    plt.scatter(counts.values, counts.index, label=label, color=color, alpha=0.7, edgecolor='k')

plt.xlabel("Number of occurrences")
plt.ylabel("Electricity Price [€/MWh] (integer)")
plt.title("Frequency of Electricity Prices in 2050 by Scenario")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""

