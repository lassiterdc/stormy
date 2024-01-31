#%%
from pyswmm import Simulation, Output

f_inp = "rz1_yr764_strm5.inp"

with Simulation(f_inp) as sim:
    for step in sim:
        pass
    sim._model.swmm_end()