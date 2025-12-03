# -*- coding: utf-8 -*-
"""
Created on Thu May  8 09:31:02 2025

@author: Dibella
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
csv_path = 'capacities_s_39.csv'  # Replace with your CSV file path
columns_to_use = ['EAF', 'DRI + EAF', 'Integrated steelworks'] 

# Replace ISO2 country codes with full names
country_names = {
    "AL": "Albania", "AT": "Austria", "BA": "Bosnia", "BE": "Belgium", "BG": "Bulgaria",
    "CH": "Switzerland", "CZ": "Czechia", "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "UK", "GR": "Greece", "HR": "Croatia", "HU": "Hungary",
    "IE": "Ireland", "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "ME": "Montenegro",
    "MK": "North Macedonia", "NL": "Netherlands", "NO": "Norway", "PL": "Poland", "PT": "Portugal", "RO": "Romania",
    "RS": "Serbia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "XK": "Kosovo"
}


# --- LOAD DATA ---
df = pd.read_csv(csv_path, index_col=0)

# --- SELECT COLUMNS ---
df = df[columns_to_use]

# --- COMBINE EAF AND DRI + EAF ---
df['CH4 EAF'] = df['EAF'] + df['DRI + EAF']
df = df.drop(columns='DRI + EAF')
df = df.drop(columns='EAF')

# --- TRIM INDEX TO FIRST 2 LETTERS AND GROUP ---
df.index = df.index.str[:2]
df_grouped = df.groupby(df.index).sum()
df_grouped.index = df_grouped.index.map(country_names.get)


# --- SUM TOTALS AND SELECT TOP 10 ---
df_grouped['Total'] = df_grouped.sum(axis=1)
df_top10 = df_grouped.sort_values(by='Total', ascending=False).drop(columns='Total').head(10)
df_top10 = df_top10.rename(columns={'Integrated steelworks': 'BOF'})
df_top10 = df_top10[['BOF', 'CH4 EAF']]  # Reorder to show BOF first (left), then EAF
df_top10 = df_top10 / 1e3  # Convert from ktsteel to Mt steel


# --- PLOT HORIZONTAL STACKED BAR CHART WITH CUSTOM COLORS ---
colors = ['black', '#552C2D']  # EAF, BOF

ax = df_top10.iloc[::-1].plot(
    kind='barh', 
    stacked=True, 
    figsize=(3, 4), 
    color=colors
)
ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
# Style y-axis ticks
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=45, ha='right')


plt.xlabel('Mt/yr')
plt.ylabel('')
plt.title('Top 10 steel producers (2024)')
plt.legend(title='', bbox_to_anchor=(1, 0.01), loc='lower right', prop={'size': 7})
plt.tight_layout()
plt.savefig("graphs/exis_steel_cap.png", bbox_inches='tight')

plt.show()

def plot_single_tech_sector(column, sector_name, output_filename, unit='Mt/yr', color='#336699', top_n=10):
    # Load and prepare data
    df_sector = pd.read_csv(csv_path, index_col=0)[[column]]
    df_sector.index = df_sector.index.str[:2]
    df_sector = df_sector.groupby(df_sector.index).sum()
    df_sector['Total'] = df_sector.sum(axis=1)
    df_sector = df_sector.sort_values(by='Total', ascending=False).drop(columns='Total').head(top_n)
    df_sector = df_sector / 1e3  # Convert to Mt
    df_sector.index = df_sector.index.map(country_names.get)

    # Plot
    ax = df_sector.iloc[::-1].plot(
        kind='barh',
        stacked=True,
        figsize=(3, 2.5 if top_n <= 3 else 4),
        color=color,
        legend=False
    )
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=45, ha='right')
    plt.xlabel(unit)
    plt.ylabel('')
    plt.title(f'Top {top_n} {sector_name} producers (2024)')
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.show()



# --- PLOT FOR EACH SECTOR ---
plot_single_tech_sector('Cement', 'cement', 'graphs/exis_cement_cap.png', color='grey')
plot_single_tech_sector('Ammonia', 'ammonia', 'graphs/exis_ammonia_cap.png', color='#C4418B')
plot_single_tech_sector('Ethylene', 'HVC', 'graphs/exis_ethylene_cap.png', color='#4C3E7B')
plot_single_tech_sector('Methanol', 'methanol', 'graphs/exis_methanol_cap.png', color='#558B2F', top_n=3)

