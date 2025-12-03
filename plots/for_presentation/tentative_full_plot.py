# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:59:19 2025

@author: Dibella
"""
import pypsa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# CONFIG
years = [2030, 2040, 2050]
technologies = ['steel', 'cement', 'NH3', 'methanol', 'HVC', "H2"]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"

group_countries = {
    'North-Western Europe': ['AT', 'BE', 'CH', 'DE', 'FR', 'LU', 'NL','DK', 'EE', 'FI', 'LV', 'LT', 'NO', 'SE','GB', 'IE'],
    'Southern Europe': ['ES', 'IT', 'PT', 'GR'],
    'Eastern Europe': ['BG', 'CZ', 'HU', 'PL', 'RO', 'SK', 'SI','AL', 'BA', 'HR', 'ME', 'MK', 'RS', 'XK'],
}

country_to_group = {
    country: group for group, countries in group_countries.items() for country in countries
}

def get_production(network, tech, target_region):
    

    timestep = network.snapshot_weightings.iloc[0, 0]
    links = network.links
    p1 = network.links_t.p1

    # Filter links by technology in bus1
    is_tech = links['bus1'].str.contains(tech, case=False, na=False)
    selected_links = links[is_tech].copy()
    if tech == 'methanol':
        selected_links = selected_links.loc[selected_links.index.str.contains('methanolisation'),:]
    if tech == 'H2':
        selected_links = selected_links.loc[~selected_links.index.str.contains('pipeline'),:]
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
    lhv_hydrogen = 33.33 # MWh / t

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
    elif tech.lower() == 'h2':
        return total_mwh / lhv_hydrogen / 1e6
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

def plot_industry_production_percentages(
    scenarios,
    nice_scenario_names,
    custom_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    get_production_percentages,
    save_path="full_graphs/ind_prod_perc_climate.png"
):
    region_names = list(group_countries.keys())
    color_map = plt.get_cmap("tab10")
    colors = {
        region: custom_colors.get(region, color_map(i % 10))
        for i, region in enumerate(region_names)
    }

    # Load networks
    networks = {}
    for year in years:
        for scenario in scenarios:
            key = (scenario, year)
            path = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
            networks[key] = pypsa.Network(path)

    # Set up subplot grid
    n_rows = len(scenarios)
    n_cols = len(technologies)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*n_cols, 3.5*n_rows), sharex=True, sharey=True)

    for i, scenario in enumerate(scenarios):
        for j, tech in enumerate(technologies):
            ax = axes[i, j] if n_rows > 1 else axes[j]

            # Title for top row
            custom_titles = {
                "hvc": "Plastics",
                "nh3": "Ammonia",
                "h2": "Hydrogen"
            }
            
            if i == 0:
                title = custom_titles.get(tech.lower(), tech.title())
                ax.set_title(f"{title}", fontsize=13)
            if j == 0:
                ax.set_ylabel(nice_scenario_names.get(scenario, scenario), fontsize=13,
                              rotation=90, labelpad=40, va='center')

            ax.grid(True, linestyle='--', alpha=0.3)
            x_pos = np.arange(len(years))
            bottom = np.zeros(len(years))

            for region in region_names:
                pct_values = []
                for year in years:
                    net = networks[(scenario, year)]
                    region_pct, _ = get_production_percentages(net, tech)
                    pct_values.append(region_pct.get(region, 0))
                pct_values = np.array(pct_values)

                bars = ax.bar(x_pos, pct_values, bottom=bottom, color=colors[region], label=region, width=1.0)

                # Annotate bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 3:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + height / 2,
                            f'{height:.1f}%',
                            ha='center', va='center', fontsize=8, color='white'
                        )
                bottom += pct_values

            ax.set_ylim(0, 100)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(years)

    # Legend outside
    #fig.legend(
    #    handles=[plt.Line2D([0], [0], color=colors[r], label=r) for r in region_names],
    #    loc='center right',
    #    fontsize=10,
    #    bbox_to_anchor=(0.1, 0.95),
    #    borderaxespad=0,
    #    frameon=True
    #)

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    #plt.suptitle("Industry Production Percentage Over Time by Tech, Scenario, and Region", fontsize=16)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()



def save_region_legend_png(custom_colors, save_path="full_graphs/legend_industry_regions.png"):
    legend_elements = [
        Patch(facecolor=color, label=region)
        for region, color in custom_colors.items()
    ]

    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis('off')

    legend = ax.legend(
        handles=legend_elements,
        loc='center',
        frameon=True,
        ncol=1,  # Adjust columns if needed
        fontsize=10
    )

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)



def plot_regional_production_absolute(
    networks,
    scenarios,
    nice_scenario_names,
    technologies,
    years,
    group_countries,
    custom_colors,
    save_path="full_graphs/regional_production_absolute.png",
    tech_ymax=None
):
    """
    Plot absolute production values by commodity (columns) with regions as stacked colors.
    Shows labels on bars except when value < 2% of the max total production per commodity.
    Y-axis limit unified per commodity by tech_ymax.
    """
    regions = list(group_countries.keys())
    n_rows = len(scenarios)
    n_cols = len(technologies)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharex=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    bar_width = 8  # Width of bars

    # Precompute max total production per commodity over all scenarios and years
    if tech_ymax is None:
        tech_ymax = {}
        for tech in technologies:
            max_val = 0
            for scenario in scenarios:
                for year in years:
                    net = networks.get((scenario, year))
                    if net is None:
                        continue
                    total_prod = 0
                    for region in regions:
                        total_prod += get_production(net, tech, region)
                    if total_prod > max_val:
                        max_val = total_prod
            tech_ymax[tech] = max_val * 1.1 if max_val > 0 else 1  # Add 10% headroom or set 1 to avoid zero

    for i, scenario in enumerate(scenarios):
        for j, tech in enumerate(technologies):
            ax = axes[i, j]
            
            # Store production data for each region for this technology
            tech_data = {region: [] for region in regions}

            for year in years:
                net = networks.get((scenario, year))
                if net is None:
                    for region in regions:
                        tech_data[region].append(0.0)
                    continue

                # Get production for each region for this technology
                for region in regions:
                    production = get_production(net, tech, region)
                    tech_data[region].append(production)

            # Create stacked bars
            bottom = np.zeros(len(years))
            max_total = tech_ymax.get(tech, None)
            label_threshold = 0.02 * max_total if max_total else 0  # 2% threshold
            
            for region in regions:
                values = np.array(tech_data[region])
                bars = ax.bar(
                    years,
                    values,
                    bottom=bottom,
                    width=bar_width,
                    label=region,
                    color=custom_colors.get(region, 'gray'),
                    edgecolor='black',
                    linewidth=0.3,
                    align='center'
                )
                
                # Add value labels on bars if value >= 2% of max total production
                for bar, value in zip(bars, values):
                    if value >= label_threshold:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_y() + bar.get_height() / 2,
                            f'{value:.1f}',
                            ha='center', va='center', 
                            fontsize=8, color='white', weight='bold'
                        )
                
                bottom += values

            # Formatting
            if j == 0:
                ax.set_ylabel(f"{nice_scenario_names[scenario]}\n\nMtons/yr", fontsize=12)

            # Custom titles for technologies (columns)
            custom_titles = {
                "hvc": "Plastics",
                "nh3": "Ammonia",
                "h2": "Hydrogen"
            }
            
            if i == 0:
                title = custom_titles.get(tech.lower(), tech.title())
                ax.set_title(f"{title}", fontsize=14)

            ax.set_xticks(years)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.grid(True, linestyle="--", alpha=0.3)
            
            # Use unified max y-limit per commodity
            if max_total is not None:
                ax.set_ylim(0, max_total)

    # Create legend for regions
    region_handles = [
        mpatches.Patch(color=custom_colors.get(region, 'gray'), label=region) 
        for region in regions
    ]
    
    """
    fig.legend(
        handles=region_handles,
        loc='center right',
        bbox_to_anchor=(0.99, 0.5),
        fontsize=11,
        title="Regions",
        frameon=True,
        handlelength=1.5
    )
    """
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    
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



    
# %% 
# CLIMATE POLICY CASE
"""
scenarios = ["base_eu_regain", "policy_eu_regain"]
nice_scenario_names =  {
    "base_eu_regain": "NO CLIMATE POLICY\nCompetitive industry\nRelocation",
    "policy_eu_regain": "CLIMATE POLICY\nCompetitive industry\nRelocation",
}

custom_colors = {
    'Southern Europe': '#D8973C',
    'North-Western Europe': '#1B264F',
    'Eastern Europe': '#9B7EDE',
}

plot_industry_production_percentages(
    scenarios,
    nice_scenario_names,
    custom_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    get_production_percentages,
    save_path="full_graphs/ind_prod_perc_climate.png"
)

nice_scenario_names =  {
    "policy_eu_regain": "Climate policy\nCOMPETITIVE INDUSTRY\nRelocation",
    "policy_eu_deindustrial": "Climate policy\nDEINDUSTRIALIZATION\nRelocation",
}

# INDUSTRIAL POLICY CASE
scenarios = [ "policy_eu_regain", "policy_eu_deindustrial"]

plot_industry_production_percentages(
    scenarios,
    nice_scenario_names,
    custom_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    get_production_percentages,
    save_path="full_graphs/ind_prod_perc_industrial.png"
)

# RELOCATION POLICY CASE
scenarios = [ "policy_eu_regain", "policy_reg_regain"]
nice_scenario_names =  {
    "policy_eu_regain": "Climate policy\nCompetitive industry\nRELOCATION",
    "policy_reg_regain": "Climate policy\nCompetitive industry\nHISTORICAL HUBS",
}

plot_industry_production_percentages(
    scenarios,
    nice_scenario_names,
    custom_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    get_production_percentages,
    save_path="full_graphs/ind_prod_perc_relocation.png"
)
"""
# %%

# FOUR SCENARIOS
scenarios = ["policy_eu_regain", "base_eu_regain", "policy_eu_deindustrial", "policy_reg_regain", ]
nice_scenario_names = {
    'policy_eu_regain': 'Climate policy\nCompetitive industry\nRelocation',
    'base_eu_regain': 'No climate policy\nCompetitive industry\nRelocation',
    'policy_eu_deindustrial':  'Climate policy\nDeindustrialization\nRelocation',
    'policy_reg_regain':  'Climate policy\nCompetitive industry\nHistorical hubs',
}


custom_colors = {

    'North-Western Europe': '#1B264F',
    'Eastern Europe': '#9B7EDE',
    'Southern Europe': '#D8973C',
}

plot_industry_production_percentages(
    scenarios,
    nice_scenario_names,
    custom_colors,
    group_countries,
    technologies,
    years,
    root_dir,
    res_dir,
    get_production_percentages,
    save_path="full_graphs/ind_prod_perc_4scenarios.png"
)

save_region_legend_png(custom_colors)

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
networks = load_networks(scenarios, years, root_dir, res_dir)



plot_regional_production_absolute(
    networks,
    scenarios,
    nice_scenario_names,
    technologies,
    years,
    group_countries,
    custom_colors,
    save_path="full_graphs/regional_production_absolute.png"
)

# %%


# Define scenarios and their nice names

scenarios = ["base_eu_regain","base_eu_deindustrial"]

networks = {}
for year in years:
    for scenario in scenarios:
        key = (scenario, year)
        path = root_dir + res_dir + f"{scenario}/networks/base_s_39___{year}.nc"
        networks[key] = pypsa.Network(path)

scenarios_to_plot = {
    "base_eu_regain": ("Regain competitiveness", "solid"),
    "base_eu_deindustrial": ("Deindustrialization", "dashed")
}

color_map = plt.get_cmap("tab10")
tech_colors = {tech: color_map(i) for i, tech in enumerate(technologies)}

production_data = {
    'steel': (2020, 126.219),
    'cement': (2020, 191),
    'NH3': (2020, 8.959),
    'methanol': (2020, 2.1),
    'HVC': (2020, 47.2)
}

fig, ax = plt.subplots(figsize=(10, 6))

# Use 2020 plus model years for x-axis
extended_years = [2020] + years

for scenario, (nice_name, line_style) in scenarios_to_plot.items():
    total_prod_per_tech_year = {tech: [] for tech in technologies}
    
    for tech in technologies:
        for year in years:
            total = 0.0
            net = networks.get((scenario, year))
            if net:
                for region in group_countries.keys():
                    total += get_production(net, tech, region)
            total_prod_per_tech_year[tech].append(total)
    
    for tech in technologies:
        # Get 2020 value from production_data if exists
        prod_2020 = None
        if tech in production_data:
            prod_2020 = production_data[tech][1]
        
        y_vals = total_prod_per_tech_year[tech]
        if prod_2020 is not None:
            y_full = [prod_2020] + y_vals
        else:
            y_full = y_vals
        
        ax.plot(extended_years, y_full, linestyle=line_style, marker='o', color=tech_colors[tech])
        
        # Label all points
        for x, y in zip(extended_years, y_full):
            ax.text(x, y, f'{y:.1f}', fontsize=8, ha='center', va='bottom')
            
ax.set_ylabel("European Production [Mtons/yr]")
ax.set_xticks(extended_years)
ax.grid(True, linestyle='--', alpha=0.3)

# Legend: tech colors with markers
tech_handles = [
    mlines.Line2D([], [], color=tech_colors[tech], marker='o', linestyle='-', label=tech.title())
    for tech in technologies
]
legend1 = ax.legend(handles=tech_handles, title="Commodity", bbox_to_anchor=(1, 1))

# Legend: scenario line styles
scenario_handles = [
    mlines.Line2D([], [], color='black', linestyle='solid', label='Regain competitiveness'),
    mlines.Line2D([], [], color='black', linestyle='dashed', label='Deindustrialization'),
]
legend2 = ax.legend(handles=scenario_handles, title="Scenario", bbox_to_anchor=(1.0, 0.7))

ax.add_artist(legend1)

plt.title("Total Industry Production per Technology Over Time")
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("full_graphs/total_industry_production.png", dpi=300, bbox_inches='tight')
plt.show()
