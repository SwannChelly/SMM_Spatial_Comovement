import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gpd


low_high = True
if low_high: 
    folder = "../bins/"

emp_chi_si = np.load(os.path.join(folder,"emp_chi_si.npy"))
filter_A_downstream = np.load(os.path.join(folder,"filter_A_downstream.npy"))
filter_N_upstream = np.load(os.path.join(folder,"filter_N_upstream.npy"))
filter_regions = filter_A_downstream*filter_N_upstream
# emp_chi_si = NPZ.npzread(joinpath(folder,"emp_chi_si.npy"))
# emp_chi_si = emp_chi_si[(filter_N_upstream.*filter_out_reference_region).!=0.0]


france = gpd.read_file(os.path.join(folder,"france.shp"))

M_ij = np.load(os.path.join(folder,"M_ij.npy"))
M_ij = pd.DataFrame(M_ij, index=france[region].values, columns=france[region].values)
M_ij.reset_index(inplace=True)
M_ij.rename(columns={'index': 'ze2010_i'}, inplace=True)
M_ij = M_ij.melt(id_vars='ze2010_i', var_name='ze2010_j', value_name='M_ij')


if __name__ == "__main__":

    fig,axs = plt.subplots(1,1)
    axs.matshow(M_ij)
    # fig.savefig(os.path.join(folder,"M_ij.png"))
    fig.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    france['selected_ze'] = france.ze2010_code.isin(tmp.ze2010)
    france = france.set_crs(crs="EPSG:4326")
    france.query('selected_ze').plot(ax=ax, color="red", edgecolor="black", alpha=0.2)
    france.query('~selected_ze').plot(ax=ax, color="white", edgecolor="black", alpha=0.2)




