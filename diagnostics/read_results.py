# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 11:57:42 2024

@author: alice
"""


import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa


root_dir = "C:/Users/alice/Desktop/CMCC/pypsa-eur-adb/"
scenario = "ff55"
res_dir = "results/"


years = [2030, 2040, 2050]
scenarios = ['baseline', 'ff55']


fn = (root_dir + res_dir + scenario + "/postnetworks/base_s_39_lvopt___2040.nc")
n = pypsa.Network(fn)

alinks = n.links

# %%
# Retrive the links for ETS, ETS2, non ETS
bus_columns = [col for col in alinks.columns if col.startswith("bus")]

ets = pd.DataFrame(columns=alinks.columns)
ets2 = pd.DataFrame(columns=alinks.columns)
nonets = pd.DataFrame(columns=alinks.columns)

for col in bus_columns:
    ets = pd.concat([ets, alinks[alinks[col] == 'co2_ets']])
    ets2 = pd.concat([ets2, alinks[alinks[col] == 'co2_ets2']])
    nonets = pd.concat([nonets, alinks[alinks[col] == 'co2_nonets']])

ets = ets.drop_duplicates()
ets2 = ets2.drop_duplicates()
nonets = nonets.drop_duplicates()

# %%

astore = n.stores
aconstr = n.global_constraints
aloads = n.loads_t.p.sum()

acarr = n.carriers
