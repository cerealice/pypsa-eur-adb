# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build spatial distribution of industries from Hotmaps database.

Inputs
-------

- ``resources/regions_onshore_base_s_{clusters}.geojson``
- ``resources/pop_layout_base_s_{clusters}.csv``

Outputs
-------

- ``resources/industrial_distribution_key_base_s_{clusters}.csv``

Description
-------

This rule uses the `Hotmaps database <https://gitlab.com/hotmaps/industrial_sites/industrial_sites_Industrial_Database>`. After removing entries without valid locations, it assigns each industrial site to a bus region based on its location.
Then, it calculates the nodal distribution key for each sector based on the emissions of the industrial sites in each region. This leads to a distribution key of 1 if there is only one bus per country and <1 if there are multiple buses per country. The sum over buses of one country is 1.

The following subcategories of industry are considered:
- Iron and steel
- Cement
- Refineries
- Paper and printing
- Chemical industry
- Glass
- Non-ferrous metals
- Non-metallic mineral products
- Other non-classified
Furthermore, the population distribution is added
- Population
"""

import logging
import uuid
from itertools import product

import country_converter as coco
import geopandas as gpd
import pandas as pd
from _helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)
cc = coco.CountryConverter()


def locate_missing_industrial_sites(df):
    """
    Locate industrial sites without valid locations based on city and
    countries.

    Should only be used if the model's spatial resolution is coarser
    than individual cities.
    """
    try:
        from geopy.extra.rate_limiter import RateLimiter
        from geopy.geocoders import Nominatim
    except ImportError:
        raise ModuleNotFoundError(
            "Optional dependency 'geopy' not found."
            "Install via 'conda install -c conda-forge geopy'"
            "or set 'industry: hotmaps_locate_missing: false'."
        )

    locator = Nominatim(user_agent=str(uuid.uuid4()))
    geocode = RateLimiter(locator.geocode, min_delay_seconds=2)

    def locate_missing(s):
        if pd.isna(s.City) or s.City == "CONFIDENTIAL":
            return None

        loc = geocode([s.City, s.Country], geometry="wkt")
        if loc is not None:
            logger.debug(f"Found:\t{loc}\nFor:\t{s['City']}, {s['Country']}\n")
            return f"POINT({loc.longitude} {loc.latitude})"
        else:
            return None

    missing = df.index[df.geom.isna()]
    df.loc[missing, "coordinates"] = df.loc[missing].apply(locate_missing, axis=1)

    # report stats
    num_still_missing = df.coordinates.isna().sum()
    num_found = len(missing) - num_still_missing
    share_missing = len(missing) / len(df) * 100
    share_still_missing = num_still_missing / len(df) * 100
    logger.warning(
        f"Found {num_found} missing locations. \nShare of missing locations reduced from {share_missing:.2f}% to {share_still_missing:.2f}%."
    )

    return df


def prepare_hotmaps_database(regions):
    """
    Load hotmaps database of industrial sites and map onto bus regions.
    """
    df = pd.read_csv(snakemake.input.hotmaps, sep=";", index_col=0)

    df[["srid", "coordinates"]] = df.geom.str.split(";", expand=True)

    if snakemake.params.hotmaps_locate_missing:
        df = locate_missing_industrial_sites(df)

    # remove those sites without valid locations
    df.drop(df.index[df.coordinates.isna()], inplace=True)

    df["coordinates"] = gpd.GeoSeries.from_wkt(df["coordinates"])

    gdf = gpd.GeoDataFrame(df, geometry="coordinates", crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    # the .sjoin can lead to duplicates if a geom is in two overlapping regions
    if gdf.index.duplicated().any():
        # get all duplicated entries
        duplicated_i = gdf.index[gdf.index.duplicated()]
        # convert from raw data country name to iso-2-code
        code = cc.convert(gdf.loc[duplicated_i, "Country"], to="iso2")  # noqa: F841
        # screen out malformed country allocation
        gdf_filtered = gdf.loc[duplicated_i].query("country == @code")
        # concat not duplicated and filtered gdf
        gdf = pd.concat([gdf.drop(duplicated_i), gdf_filtered])

    return gdf


def prepare_gem_database(regions):
    """
    Load GEM database of steel plants and map onto bus regions.
    """

    df = pd.read_excel(
        snakemake.input.steel_gem,
        sheet_name="Steel Plants",
        na_values=["N/A", "unknown", ">0"],
    ).query("Region == 'Europe'")

    df["Retired Date"] = pd.to_numeric(
        df["Retired Date"].combine_first(df["Idled Date"]), errors="coerce"
    )
    df["Start date"] = pd.to_numeric(
        df["Start date"].str.split("-").str[0], errors="coerce"
    )

    latlon = (
        df["Coordinates"]
        .str.split(", ", expand=True)
        .rename(columns={0: "lat", 1: "lon"})
    )
    geometry = gpd.points_from_xy(latlon["lon"], latlon["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf

def prepare_cement_database(regions):
    """
    Load Spatial Finance Initiatice Global Cement Database cement plants and map onto bus regions.
    """

    df = pd.read_excel(f"{snakemake.input.cement_sfi}", sheet_name="SFI_ALD_Cement_Database", index_col=0, header=0)
    df.loc[:,'country'] = cc.convert(df.loc[:,'country'], to="ISO2")
    df = df[df['country'].isin(countries)]
    df = df[df['plant_type'] != 'Grinding']
    df = df[df['production_type'] != 'Wet'] #for now only dry route (Wet as only 20 Mt/yr of produciton in Europe)

    latlon = df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]]
    geometry = gpd.points_from_xy(latlon["lon"], latlon["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf = gdf[~gdf.index.duplicated(keep='first')]

    gdf["country"] = gdf.bus.str[:2]

    return gdf

def prepare_chemicals_database(regions):
    """
    Load data from ECM paper "Modelling the market diffusion of hydrogen-based steel and basic chemical production in Europe – A site-specific approach"
    which in Supplementary Data 2 contains info about ammonia, HVC and chlorine plants and map onto bus regions.
    https://doi.org/10.1016/j.enconman.2024.119117
    """

    df = pd.read_excel(f"{snakemake.input.chemicals_ecm}", sheet_name="Database", index_col=0, header=0)
    df = df[df['Country'].isin(countries)]
    df = df[df['Product'] != 'Steel, primary'] # Better database for steel plants from GEM
    df = df.rename(columns={'Production in tons (calibrated)': 'capacity'})
    df = df.rename(columns={'Year of last modernisation': 'year'})
    df = df.rename(columns={'Country': 'country'})

    latlon = df.rename(columns={"Latitude": "lat", "Longitude": "lon"})[["lat", "lon"]]
    geometry = gpd.points_from_xy(latlon["lon"], latlon["lat"])
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf = gdf[~gdf.index.duplicated(keep='first')]

    gdf["Country"] = gdf.bus.str[:2]

    return gdf

def prepare_ammonia_database(regions):
    """
    Load ammonia database of plants and map onto bus regions.
    """
    df = pd.read_csv(snakemake.input.ammonia, index_col=0)

    geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf

#ADB to remove
def prepare_cement_supplement(regions):
    """
    Load supplementary cement plants from non-EU-(NO-CH) and map onto bus
    regions.
    """

    df = pd.read_csv(snakemake.input.cement_supplement, index_col=0)

    geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf


def prepare_refineries_supplement(regions):
    """
    Load supplementary refineries from non-EU-(NO-CH) and map onto bus regions.
    """

    df = pd.read_csv(snakemake.input.refineries_supplement, index_col=0)

    geometry = gpd.points_from_xy(df.Longitude, df.Latitude)
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    gdf = gpd.sjoin(gdf, regions, how="inner", predicate="within")

    gdf.rename(columns={"name": "bus"}, inplace=True)
    gdf["country"] = gdf.bus.str[:2]

    return gdf


def build_nodal_distribution_key(
    hotmaps, steel_gem, cement_sfi, chemicals_ecm, ammonia, cement, refineries, regions, countries
):
    """
    Build nodal distribution keys for each sector.
    """
    sectors = hotmaps.Subsector.unique()

    keys = pd.DataFrame(index=regions.index, columns=sectors, dtype=float)

    pop = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    pop["country"] = pop.index.str[:2]
    ct_total = pop.total.groupby(pop["country"]).sum()
    keys["population"] = pop.total / pop.country.map(ct_total)

    for sector, country in product(sectors, countries):
        regions_ct = regions.index[regions.index.str.contains(country)]

        facilities = hotmaps.query("country == @country and Subsector == @sector")

        if not facilities.empty:
            emissions = facilities["Emissions_ETS_2014"].fillna(
                hotmaps["Emissions_EPRTR_2014"].dropna()
            )
            if emissions.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                # assume 20% quantile for missing values
                emissions = emissions.fillna(emissions.quantile(0.2))
                key = emissions / emissions.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        elif sector == "Cement" and country in cement.country.unique():
            facilities = cement.query("country == @country")
            production = facilities["Cement [kt/a]"]
            if production.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                key = production / production.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        elif sector == "Refineries" and country in refineries.country.unique():
            facilities = refineries.query("country == @country")
            production = facilities["Capacity [bbl/day]"]
            if production.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                key = production / production.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = keys.loc[regions_ct, "population"]

        keys.loc[regions_ct, sector] = key

    ###
    # STEEL
    # Clean Global Energy Monitor database to get steel plants capacities and build year for each node

    # Steel subsectors
    steel_processes = ["EAF", "DRI + EAF", "Integrated steelworks"]
    # Define the final dataframes
    steel_capacities = pd.DataFrame(index=regions.index, columns=steel_processes)
    steel_start_dates = pd.DataFrame(index=regions.index, columns=steel_processes)

    for process, country in product(steel_processes, countries):
        regions_ct = regions.index[regions.index.str.contains(country)]
        # Retrieve the steel plant capacities in the country
        facilities = steel_gem.query("country == @country")

        # Check the type of steelmaking process
        if process == "EAF":
            # Define a list of valid statuses for the Electric Arc Furnace (EAF) process
            status_list = [
                "construction",
                "operating",
                "operating pre-retirement",
                "retired",
            ]
            # Filter facilities based on:
            # - Their status being in the valid status list
            # - Their retirement date either being unspecified (NaN) or after 2025
            # Select the nominal EAF steel capacity column, dropping any NaN values
            capacities = facilities.loc[
                facilities["Capacity operating status"].isin(status_list)
                & (
                    facilities["Retired Date"].isna()
                    | facilities["Retired Date"].gt(2025)
                ),
                "Nominal EAF steel capacity (ttpa)",
            ].dropna()

        elif process == "DRI + EAF":
            # Define a list of valid statuses for the Direct Reduced Iron (DRI) + EAF process
            status_list = [
                "construction",
                "operating",
                "operating pre-retirement",
                "retired",
                "announced",
            ]
            # Define the columns relevant to DRI + EAF capacity calculations
            sel = [
                "Nominal BOF steel capacity (ttpa)",  # Basic Oxygen Furnace capacity
                "Nominal OHF steel capacity (ttpa)",  # Open Hearth Furnace capacity
                "Nominal iron capacity (ttpa)",       # Iron capacity
            ]
            # Filter conditions:
            # 1. The status is in the valid status list
            status_filter = facilities["Capacity operating status"].isin(status_list)
            # 2. The retirement date is either unspecified (NaN) or after 2030
            retirement_filter = facilities["Retired Date"].isna() | facilities[
                "Retired Date"
            ].gt(2030)
            # 3. The start date is either unspecified (and status is not "announced") or before/equal to 2030
            start_filter = (
                facilities["Start date"].isna()
                & ~facilities["Capacity operating status"].eq("announced")
            ) | facilities["Start date"].le(2030)
            # Filter the facilities using the above conditions, sum relevant columns row-wise, and drop NaN values
            capacities = (
                facilities.loc[status_filter & retirement_filter & start_filter, sel]
                .sum(axis=1)
                .dropna()
            )

        elif process == "Integrated steelworks":
            # Define a list of valid statuses for Integrated steelworks
            status_list = [
                "construction",
                "operating",
                "operating pre-retirement",
                "retired",
            ]
            # Define the columns relevant to Integrated steelworks capacity calculations
            sel = [
                "Nominal BOF steel capacity (ttpa)",  # Basic Oxygen Furnace capacity
                "Nominal OHF steel capacity (ttpa)",  # Open Hearth Furnace capacity
            ]
            # Filter facilities based on:
            # - Their status being in the valid status list
            # - Their retirement date either being unspecified (NaN) or after 2025
            # Select and sum the relevant columns row-wise, dropping NaN values
            capacities = (
                facilities.loc[
                    facilities["Capacity operating status"].isin(status_list)
                    & (
                        facilities["Retired Date"].isna()
                        | facilities["Retired Date"].gt(2025)
                    ),
                    sel,
                ]
                .sum(axis=1)
                .dropna()
            )

        else:
            # Raise an error if an unknown process is provided
            raise ValueError(f"Unknown process {process}")

        # Sum capacities and store in the corresponding country and process in steel_capacities dataframe
        capacities_sum = capacities.sum() if not capacities.empty else 0
        steel_capacities.loc[regions_ct, process] = capacities_sum
        
        # Calculate the weighted average of start dates using capacities as weights
        if not capacities.empty:
            start_dates = facilities.loc[capacities.index, "Start date"].dropna()
            filtering = capacities[(start_dates != 0) & (capacities != 0)].index
            filtered_capacities = capacities.loc[filtering]
            filtered_start_dates = start_dates.loc[filtering]
            filtered_capacities_sum = filtered_capacities.sum()

            if filtered_capacities_sum > 0:
                weighted_sum = (filtered_capacities * filtered_start_dates).sum()
                weighted_avg = weighted_sum / filtered_capacities_sum
                steel_start_dates.loc[regions_ct, process] = weighted_avg
            else:
                # If no valid capacities, assign 0
                steel_start_dates.loc[regions_ct, process] = 0
        else:
            # If capacities are empty, assign 0
            steel_start_dates.loc[regions_ct, process] = 0

        if not capacities.empty:
            if capacities.sum() == 0:
                key = pd.Series(1 / len(capacities), capacities.index)
            else:
                key = capacities / capacities.sum()
            buses = facilities.loc[capacities.index, "bus"]
            key = key.groupby(buses).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = keys.loc[regions_ct, "population"]

        keys.loc[regions_ct, process] = key

    # Data input might change, so this warning should highlight if all values of start dates and thus capacities are 0
    if (steel_start_dates == 0).all().all():
        logger.warning("All values in the steel capacities and build year are 0. Check your data input")

    ###
    # CEMENT
    # Clean Sustainable Finance Initiative database to get cement plants capacities and build year for each node

    # Initialize DataFrames to store cement capacities and build year for each node
    cement_capacities = pd.DataFrame(0, index=regions.index, columns=['capacity'])
    cement_start_dates = pd.DataFrame(0, index=regions.index, columns=['year'])

    # Iterate through each country to compute cement capacities and start dates
    for country in countries:
        # Identify regions corresponding to the current country
        regions_ct = regions.index[regions.index.str.contains(country)]

        # Query cement facilities for the current country
        facilities = cement_sfi.query("country == @country")

        # Extract capacity values, ensuring they are non-null, non-zero, and not duplicated
        capacities = facilities['capacity'].dropna()
        capacities = capacities[capacities != 0]
        capacities = capacities[~capacities.index.duplicated(keep='first')]
        capacities = capacities * 1e3  # Convert capacities from Mt/yr to kt/yr

        # Calculate the total capacity and store it in the corresponding regions
        capacities_sum = capacities.sum() if not capacities.empty else 0
        cement_capacities.loc[regions_ct, 'capacity'] = capacities_sum

        # Calculate the weighted average start date using capacities as weights
        if not capacities.empty:
            # Get start dates corresponding to the valid capacities
            start_dates = facilities.loc[capacities.index, "year"].dropna()
            start_dates = start_dates[~start_dates.index.duplicated(keep='first')]

            # Filter capacities to match start dates
            filtered_capacities = capacities.loc[start_dates.index]
            filtered_capacities_sum = filtered_capacities.sum()

            if filtered_capacities_sum > 0:
                # Calculate weighted average of start dates
                weighted_sum = (filtered_capacities * start_dates).sum()
                weighted_avg = weighted_sum / filtered_capacities_sum
                cement_start_dates.loc[regions_ct, 'year'] = round(weighted_avg)
            else:
                # If no valid capacities, assign 0
                cement_start_dates.loc[regions_ct, 'year'] = 0
        else:
            # If capacities are empty, assign 0
            cement_start_dates.loc[regions_ct, 'year'] = 0

        # Compute keys for cement SFI allocation
        if not capacities.empty:
            if capacities.sum() == 0:
                # If all capacities are zero, assign equal weights
                key = pd.Series(1 / len(capacities), index=capacities.index)
            else:
                # Compute weights based on the proportion of capacities
                key = capacities / capacities.sum()

            # Group keys by bus and sum them, filling missing regions with 0
            buses = facilities.loc[capacities.index, "bus"]
            print(f"Buses {buses}")
            key = key.groupby(buses).sum().reindex(regions_ct, fill_value=0.0)
        else:
            # If capacities are empty, fallback to population-based keys
            key = keys.loc[regions_ct, "population"]

    # Store the computed keys in the keys DataFrame
    keys.loc[regions_ct, 'Cement_SFI'] = key

    # Data input might change, so this warning should highlight if all values of start dates and thus capacities are 0
    if (cement_start_dates == 0).all().all():
        logger.warning("All values in the cement capacities and build year are 0. Check your data input")

    ###
    # AMMONIA

    # OLD CODE: this is overwritten byt the following code, since it contains more accurate data on ammonia plants location and build year
    # add ammonia
    for country in countries:
        regions_ct = regions.index[regions.index.str.contains(country)]

        facilities = ammonia.query("country == @country")

        if not facilities.empty:
            production = facilities["Ammonia [kt/a]"]
            if production.sum() == 0:
                key = pd.Series(1 / len(facilities), facilities.index)
            else:
                # assume 50% of the minimum production for missing values
                production = production.fillna(0.5 * facilities["Ammonia [kt/a]"].min())
                key = production / production.sum()
            key = key.groupby(facilities.bus).sum().reindex(regions_ct, fill_value=0.0)
        else:
            key = 0.0

        keys.loc[regions_ct, "Ammonia"] = key

    # add chemicals plants
    chemicals = ['Ammonia','Ethylene','Methanol']
    chemicals_capacities = pd.DataFrame(0, index = regions.index, columns = chemicals)  
    chemicals_start_dates = pd.DataFrame(0, index = regions.index, columns = chemicals)

    for country in countries:
        for chem in chemicals:

            regions_ct = regions.index[regions.index.str.contains(country)]

            facilities = chemicals_ecm.query("country == @country and Product == @chem")

            capacities = facilities['capacity'].dropna()
            capacities = capacities[capacities != 0]
            capacities = capacities[~capacities.index.duplicated(keep='first')]
            capacities = capacities / 1e3 # from t/yr to kt/yr (coherent with cement)

            # Sum capacities and store in the corresponding country and process in steel_capacities dataframe
            capacities_sum = capacities.sum() if not capacities.empty else 0
            chemicals_capacities.loc[regions_ct, chem] = capacities_sum

            # Calculate the weighted average of start dates using capacities as weights
            if not capacities.empty:
                start_dates = facilities.loc[capacities.index, "year"].dropna()
                start_dates = start_dates[~start_dates.index.duplicated(keep='first')]
                filtered_capacities = capacities.loc[start_dates.index]
                filtered_capacities_sum = filtered_capacities.sum()
                if filtered_capacities_sum > 0:
                    weighted_sum = (filtered_capacities * start_dates).sum()
                    weighted_avg = weighted_sum / filtered_capacities_sum
                    chemicals_start_dates.loc[regions_ct, chem] = round(weighted_avg)
                else:
                    # If no valid capacities, assign 0
                    chemicals_start_dates.loc[regions_ct, chem] = 0
            else:
                # If capacities are empty, assign 0
                logger.warning("chemicals_start_dates is 0. Check your data input")
                chemicals_start_dates.loc[regions_ct, chem] = 0

            chemicals_start_dates = chemicals_start_dates.fillna(0)


    return keys, steel_capacities, steel_start_dates, cement_capacities, cement_start_dates, chemicals_capacities, chemicals_start_dates


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_industrial_distribution_key",
            clusters=128,
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    countries = snakemake.params.countries

    regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")

    hotmaps = prepare_hotmaps_database(regions)

    steel_gem = prepare_gem_database(regions)

    cement_sfi = prepare_cement_database(regions)

    chemicals_ecm = prepare_chemicals_database(regions)

    ammonia = prepare_ammonia_database(regions)

    cement = prepare_cement_supplement(regions)

    refineries = prepare_refineries_supplement(regions)

    keys, steel_capacities, steel_start_dates, cement_capacities, cement_start_dates, chemicals_capacities, chemicals_start_dates = build_nodal_distribution_key(
        hotmaps, steel_gem, cement_sfi, chemicals_ecm, ammonia, cement, refineries, regions, countries
    )

    keys.to_csv(snakemake.output.industrial_distribution_key)
    steel_capacities.to_csv(snakemake.output.steel_capacities)
    steel_start_dates.to_csv(snakemake.output.steel_start_dates)
    cement_capacities.to_csv(snakemake.output.cement_capacities)
    cement_start_dates.to_csv(snakemake.output.cement_start_dates)
    chemicals_capacities.to_csv(snakemake.output.chemicals_capacities)
    chemicals_start_dates.to_csv(snakemake.output.chemicals_start_dates)