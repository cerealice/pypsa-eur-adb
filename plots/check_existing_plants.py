# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 11:32:38 2025

@author: Dibella
"""

import pypsa
import pandas as pd

n = pypsa.Network(r"C:\Users\Dibella\Desktop\CMCC\pypsa-adb-industry\results_3h_juno\policy_eu_regain\networks\base_s_39___2030.nc")
timestep = n.snapshot_weightings.iloc[0,0]

alinks = n.links
alinks_2020 = alinks[alinks.index.str.contains("-2020")]
alinks_t = -n.links_t.p1.loc[:,alinks_2020.index].sum()*timestep
alinks_2020_p = alinks_2020.p_nom
df = pd.concat([alinks_t,alinks_2020_p], axis=1)
