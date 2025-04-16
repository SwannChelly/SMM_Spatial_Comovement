import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors


low_high = True
if low_high: 
    folder = "../bins/"


# 1101  : Paris 
# 0061  : Toulouse
# 9308  : Aix en provence 
# 5203  : Nantes
# 9310  : Marseille

emp_chi_si = np.load(os.path.join(folder,"emp_chi_si.npy"))
filter_A_downstream = np.load(os.path.join(folder,"filter_A_downstream.npy"))
filter_N_upstream = np.load(os.path.join(folder,"filter_N_upstream.npy"))
filter_regions = filter_A_downstream*filter_N_upstream
# emp_chi_si = NPZ.npzread(joinpath(folder,"emp_chi_si.npy"))
# emp_chi_si = emp_chi_si[(filter_N_upstream.*filter_out_reference_region).!=0.0]


france = gpd.read_file(os.path.join(folder,"france.shp"))

if __name__ == "__main__":

    M_ij = np.load(os.path.join(folder,"M_ij.npy"))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.matshow(M_ij)
    fig.savefig(os.path.join(folder,f"M_ij.png"))

    M_ij = pd.DataFrame(M_ij, index=france["ze2010"].values, columns=france["ze2010"].values)
    M_ij.reset_index(inplace=True)
    M_ij.rename(columns={'index': 'ze2010_i'}, inplace=True)
    M_ij = M_ij.melt(id_vars='ze2010_i', var_name='ze2010_j', value_name='M_ij')


    ze_to_shock = {"1101":"Paris","0061":"Toulouse","9310":"Marseille - Aubagne","5203":"Nantes"}

    i = 0
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    for ze_shocked,name_ze_shocked in ze_to_shock.items():
        tmp = M_ij.query(f'ze2010_j == "{ze_shocked}"')
        tmp = france.merge(tmp,right_on = "ze2010_i",left_on = "ze2010")
        tmp.query('M_ij == 0').plot(color = "gray",ax=ax[i//2,i%2])

        norm = mcolors.LogNorm(vmin=tmp.query('M_ij > 0').M_ij.min(), vmax=tmp.query('M_ij > 0').M_ij.max())
        tmp.query('M_ij > 0').plot(column = "M_ij",ax=ax[i//2,i%2],norm = norm, cmap = "viridis")
        #tmp.plot(column = "M_ij",ax=ax[i//2,i%2], edgecolor="black")

        ax[i//2,i%2].set_title(name_ze_shocked)

        tmp.query(f'ze2010_i == "{ze_shocked}"').plot(facecolor="none",ax=ax[i//2,i%2], edgecolor="red",hatch ="//")

        divider = make_axes_locatable(ax[i//2, i%2])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])  # Dummy array for the scalar mappable
        fig.colorbar(sm, cax=cax)

        ax[i//2,i%2].set_xticks([])
        ax[i//2,i%2].set_yticks([])
        ax[i//2,i%2].set_title(name_ze_shocked)
        
        i+=1


    fig.savefig(os.path.join(folder,f"map.png"))

    






