# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Change discount rate in costs_{year}.csv.
"""

import pandas as pd
import logging
from scripts._helpers import configure_logging, set_scenario_config

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("change_discount_rate", year=2030)
        rootpath = ".."
    else:
        rootpath = "."

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    logger = logging.getLogger(__name__)

    input_file = snakemake.input.costs_raw
    output_file = snakemake.output.costs
    discount_rate = snakemake.params.discount_rate["discount rate"]
    javier = snakemake.params.javier

    try:
        df = pd.read_csv(input_file)
        logger.info(f"Loaded input file: {input_file}")
        logger.info(f"Columns in input: {df.columns.tolist()}")

        if not javier:
            logger.info("Javier flag is False — no changes made.")
        else:
            if "parameter" in df.columns and "value" in df.columns:
                mask = df["parameter"].str.lower() == "discount rate"
                if mask.any():
                    df.loc[mask, "value"] = discount_rate
                    logger.info(f"Discount rate updated to {discount_rate}.")
                else:
                    logger.warning("No 'discount rate' found in the input file.")
            else:
                logger.warning("Required columns 'parameter' and 'value' not found.")

        df.to_csv(output_file, index=False)
        logger.info(f"Output saved to: {output_file}")

    except Exception as e:
        logger.exception(f"Error while changing discount rate: {e}")
        raise
