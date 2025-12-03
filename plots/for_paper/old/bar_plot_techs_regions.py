# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 15:21:01 2025

@author: Dibella
"""

import pypsa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# CONFIG
years = [2030, 2040, 2050]
technologies = ['steel', 'cement', 'NH3', 'methanol', 'HVC', "H2"]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_july/"

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


def save_region_legend_png(custom_colors, save_path="graphs/legend_industry_regions.png"):
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
    save_path="graphs/regional_production_absolute.png",
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
    
    if n_cols >= 3:
        ax2 = axes[0, 1]  # Column 2
        ax3 = axes[0, 2]  # Column 3
        x2 = ax2.get_position().x1
        x3 = ax3.get_position().x0
        x_line = (x2 + x3) / 2
    
        fig.lines.append(
            plt.Line2D([x_line, x_line], [0, 1], transform=fig.transFigure, color='black', linewidth=1)
        )
    
    # Add group titles above columns
    if n_cols >= 4:
        x_first_group = (axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2
        x_second_group = (axes[0, 2].get_position().x0 + axes[0, 3].get_position().x1) / 2
    
        fig.text(x_first_group, 0.95, "CURRENT DEINDUSTRIALIZATION", ha='center', fontsize=14)
        fig.text(x_second_group, 0.95, "REINDUSTRIALIZE", ha='center', fontsize=14)

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
    
    
def plot_regional_production_absolute_new(
    networks,
    scenarios,
    nice_scenario_names,
    technologies,
    years,
    group_countries,
    custom_colors,
    save_path="graphs/regional_production_absolute.png",
    tech_ymax=None
):
    """
    Plot absolute production values with technologies as rows and scenarios as columns.
    Regions are stacked colors. Additional group titles and separation lines kept.
    """
    regions = list(group_countries.keys())
    n_rows = len(technologies)
    n_cols = len(scenarios)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharex=True)

    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    bar_width = 8

    # Precompute max production per technology
    if tech_ymax is None:
        tech_ymax = {}
        for tech in technologies:
            max_val = 0
            for scenario in scenarios:
                for year in years:
                    net = networks.get((scenario, year))
                    if net is None:
                        continue
                    total_prod = sum(get_production(net, tech, region) for region in regions)
                    max_val = max(max_val, total_prod)
            tech_ymax[tech] = max_val * 1.1 if max_val > 0 else 1

    # Custom titles for technologies
    custom_titles = {
        "hvc": "Plastics",
        "nh3": "Ammonia",
        "h2": "Hydrogen"
    }

    for i, tech in enumerate(technologies):
        for j, scenario in enumerate(scenarios):
            ax = axes[i, j]
            
            tech_data = {region: [] for region in regions}
            for year in years:
                net = networks.get((scenario, year))
                if net is None:
                    for region in regions:
                        tech_data[region].append(0.0)
                    continue

                for region in regions:
                    production = get_production(net, tech, region)
                    tech_data[region].append(production)

            bottom = np.zeros(len(years))
            max_total = tech_ymax.get(tech, None)
            label_threshold = 0.02 * max_total if max_total else 0
            
            if i == 0:
                ax.set_title(nice_scenario_names[scenario], fontsize=14)
            
            # Set tech name on y-axis for first column
            if j == 0:
                tech_title = custom_titles.get(tech.lower(), tech.title())
                ax.set_ylabel(f"{tech_title}\n\nMtons/yr", fontsize=12)

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

            # Axis labeling
            if i == n_rows - 1:
                ax.set_xlabel("Year", fontsize=12)
            if j == 0:
                tech_title = custom_titles.get(tech.lower(), tech.title())
                ax.set_ylabel(f"{tech_title}\n\nMtons/yr", fontsize=12)
                
                # Set y-axis ticks only for first column
            if j == 0:
                ax.yaxis.set_visible(True)
            else:
                ax.yaxis.set_visible(False)

            ax.set_xticks(years)
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
            ax.grid(True, linestyle="--", alpha=0.3)
            if max_total is not None:
                ax.set_ylim(0, max_total)

    # === Keep Separation Lines and Group Labels ===
    if n_cols >= 3:
        ax2 = axes[0, 1]  # Column 2
        ax3 = axes[0, 2]  # Column 3
        x2 = ax2.get_position().x1
        x3 = ax3.get_position().x0
        
        x_line = (x2 + x3) / 2 - (x2 + x3) / 28
        print(x_line)
        fig.lines.append(
            plt.Line2D([x_line, x_line], [0, 1], transform=fig.transFigure, color='black', linewidth=1)
        )

    if n_cols >= 4:
        x_first_group = (axes[0, 0].get_position().x0 + axes[0, 1].get_position().x1) / 2
        x_second_group = (axes[0, 2].get_position().x0 + axes[0, 3].get_position().x1) / 2

        fig.text(x_first_group, 0.95, "CURRENT DEINDUSTRIALIZATION", ha='center', fontsize=14)
        fig.text(x_second_group, 0.95, "REINDUSTRIALIZE", ha='center', fontsize=14)
        
    """
    # === Legend ===
    region_handles = [
        mpatches.Patch(color=custom_colors.get(region, 'gray'), label=region)
        for region in regions
    ]
    
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

    plt.tight_layout(rect=[0, 0, 0.9, 0.93])
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

# FOUR SCENARIOS
scenarios = ["policy_reg_deindustrial",  "policy_eu_deindustrial", "policy_reg_regain", "policy_eu_regain", ]

nice_scenario_names = {
    'policy_reg_deindustrial': 'No relocation',
    'policy_eu_deindustrial':  'Relocation within EU',
    'policy_reg_regain':  'No relocation',
    'policy_eu_regain': 'Relocation within EU',

}


custom_colors = {

    'North-Western Europe': '#1B264F',
    'Eastern Europe': '#9B7EDE',
    'Southern Europe': '#D8973C',
}


save_region_legend_png(custom_colors)

years = [2030, 2040, 2050]
root_dir = "C:/Users/Dibella/Desktop/CMCC/pypsa-adb-industry/"
res_dir = "results_3h_juno/"
networks = load_networks(scenarios, years, root_dir, res_dir)

# %%

plot_regional_production_absolute_new(
    networks,
    scenarios,
    nice_scenario_names,
    technologies,
    years,
    group_countries,
    custom_colors,
    save_path="graphs/regional_production_absolute.png"
)