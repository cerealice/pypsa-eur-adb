# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 15:45:44 2025

@author: Dibella
"""

import pandas as pd
import os
import pypsa
import pycountry
# -*- coding: utf-8 -*-


def get_country_name(alpha_2_code):
    try:
        if alpha_2_code == "XK":
            return "Kosovo"
        elif alpha_2_code == "HU":
            return "Hungary"
        elif alpha_2_code == "LT":
            return "Lithuania" 
        elif alpha_2_code == "LV":
            return "Latvia"
        country = pycountry.countries.get(alpha_2=alpha_2_code)
        return country.name if country else "Unknown"
    except (ValueError, AttributeError, LookupError):
        return "Unknown"


def extract_industry_investments(n, year, scenario, model, countries, industry_links, industry_name):
    links = n.links.loc[industry_links & (n.links.p_nom_extendable == True)]
    capacity_new = n.links.p_nom_opt[links.index]
    capacity_costs = n.links.capital_cost[links.index]
    capacity_investments = capacity_new * capacity_costs / 1e6

    capacity_investments = capacity_investments[~capacity_investments.index.str.contains('EU')]

    df = pd.concat([
        capacity_investments.index.str.extract(r'([A-Z][A-Z]\d?)').rename(columns={0: 'Node'}),
        capacity_investments.index.str.extract(r'([A-Z][A-Z])').rename(columns={0: 'Country'}),
        capacity_investments.index.str.extract(r'[A-Z][A-Z]\d? ?\d? (.+)').rename(columns={0: 'Source type'}),
        capacity_investments.reset_index(drop=True).rename("Investments [Meuro]")
    ], axis=1)

    df['Investments [Meuro]'] = df['Investments [Meuro]'].apply(lambda x: x if x > 1e-6 else 0)
    df['Source type'] = df['Source type'].str.strip().str.rsplit('-', n=1).str[0].str.title()
    df['Country'] = df['Country'].apply(get_country_name)
    df = df[df['Country'] != 'Europe']

    df_grouped = df.groupby(['Country', 'Source type'])['Investments [Meuro]'].sum().reset_index()

    return pd.DataFrame([
        {
            'model': model,
            'scenario': scenario,
            'region': row['Country'],
            'variable': f"Investments|Energy Supply|Industry|{industry_name}|{row['Source type']}",
            'unit': 'Meuro/y',
            'year': year,
            'value': row['Investments [Meuro]']
        }
        for _, row in df_grouped.iterrows()
    ])



# ------------------------- MAIN -------------------------

my_path = os.getcwd()
parent_directory = os.path.dirname(my_path)

years = ['2030','2040','2050']
results_dir = '../results_3h_juno/'

scenarios = ['base_eu_regain','policy_eu_regain']
model = 'PyPSA-Eur'


countries = ['Albania', 'Austria', 'Bosnia and Herzegovina', 'Belgium', 'Bulgaria', 'Switzerland',
             'Czechia', 'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'United Kingdom',
             'Greece', 'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania', 'Luxembourg', 'Latvia',
             'Montenegro', 'North Macedonia', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
             'Serbia', 'Sweden', 'Slovenia', 'Slovakia', 'Kosovo']


# Full DataFrame to collect all results
all_investments_df = []

for scenario in scenarios:
    for year in years:
        file_name = f"base_s_39___{year}"
        n = pypsa.Network(os.path.join(results_dir, scenario, "networks", file_name + ".nc"))

        # --- STEEL industry links ---
        steel_links_filter = (
            n.links.index.str.contains("BF-BOF", case=False, na=False) |
            n.links.index.str.contains("EAF", case=False, na=False) |
            n.links.index.str.contains("steel BOF CC", case=False, na=False)
        )

        # Extract steel investment data and append
        df_steel = extract_industry_investments(
            n=n,
            year=year,
            scenario=scenario,
            model=model,
            countries=countries,
            industry_links=steel_links_filter,
            industry_name='Steel'
        )
        
        # --- CEMENT industry links ---
        cement_links_filter = (
            n.links.index.str.contains("Cement Plant", case=False, na=False) |
            n.links.index.str.contains("cement TGR", case=False, na=False)
        )

        # Extract cement investment data and append
        df_cement = extract_industry_investments(
            n=n,
            year=year,
            scenario=scenario,
            model=model,
            countries=countries,
            industry_links=cement_links_filter,
            industry_name='Cement'
        )
        
        # --- AMMONIA industry links ---
        ammonia_links_filter = (
            n.links.index.str.contains("Haber Bosch", case=False, na=False)
        )

        # Extract ammonia investment data and append
        df_ammonia = extract_industry_investments(
            n=n,
            year=year,
            scenario=scenario,
            model=model,
            countries=countries,
            industry_links=ammonia_links_filter,
            industry_name='Ammonia'
        )

        all_investments_df.append(df_ammonia)
        
        # --- METHANOL industry links ---
        methanol_links_filter = (
            n.links.index.str.contains("methanolisation", case=False, na=False)
        )

        # Extract ammonia investment data and append
        df_methanol = extract_industry_investments(
            n=n,
            year=year,
            scenario=scenario,
            model=model,
            countries=countries,
            industry_links=methanol_links_filter,
            industry_name='Methanol'
        )

        all_investments_df.append(df_methanol)
        
        # --- Plastics industry links ---
        plastics_links_filter = (
            n.links.index.str.contains("naphtha steam", case=False, na=False) 
        )

        # Extract ammonia investment data and append
        df_plastics = extract_industry_investments(
            n=n,
            year=year,
            scenario=scenario,
            model=model,
            countries=countries,
            industry_links=plastics_links_filter,
            industry_name='Plastics'
        )

        all_investments_df.append(df_plastics)
        
        # --- Feedstocks industry links ---
        feedstocks_links_filter = (
            n.links.index.str.contains("Fischer-Tropsch", case=False, na=False) |
            n.links.index.str.contains("Electrolysis", case=False, na=False) |
            n.links.index.str.contains("SMR", case=False, na=False)
        )

        # Extract ammonia investment data and append
        df_feedstocks = extract_industry_investments(
            n=n,
            year=year,
            scenario=scenario,
            model=model,
            countries=countries,
            industry_links=feedstocks_links_filter,
            industry_name='Feedstocks'
        )

        all_investments_df.append(df_feedstocks)

# Concatenate all and save to one CSV
final_df = pd.concat(all_investments_df, ignore_index=True)
final_df.to_csv("investments_industry.csv", index=False, encoding='utf-8')


