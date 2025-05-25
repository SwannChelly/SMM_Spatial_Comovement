import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


################## Parameters and functions #################
plt.style.use('default')
# Set the width of your LaTeX document in points
document_width_pt = 418.25368  # Adjust this according to your LaTeX template
document_width_pt = 398.3386 # Beamer
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
cmap = plt.get_cmap('viridis')
plt.rcParams.update({"axes.grid" : False})

font_size = 20  # Adjust according to your preference
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size+4,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "legend.title_fontsize": font_size,
    "figure.titlesize": font_size,
})

def add_colorbar(ax,norm):
    divider = make_axes_locatable(ax)
    cax = inset_axes(ax, width="80%", height="5%", loc='lower center', borderpad=-1.5)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm._A = []
    fig.colorbar(sm, cax=cax, orientation='horizontal')

def load_pi_jA(file_name):
    pi_jA = np.load(os.path.join(folder,f"{file_name}.npy"))
    pi_jA = np.array([1-sum(pi_jA)] + list(pi_jA))
    df = filter_N_upstream_df[["ze2010","pi_jA"]].drop_duplicates()
    df.loc[~df.pi_jA.isna(),"sim_pi_jA"] = pi_jA
    df.pi_jA.fillna(0,inplace = True)
    df.sim_pi_jA.fillna(0,inplace = True)
    return df


def plot_pi_jA(col_name,ax):

    tmp = france.merge(df,on = "ze2010",how = "left")
    tmp[col_name].fillna(0,inplace = True)
    norm = mcolors.LogNorm(vmin=10**(-4), vmax=1)
    print(tmp.query(col_name+' == 0'))
    tmp.query(col_name+' == 0').plot(color = "gray",ax=ax)
    tmp.query(col_name+' > 0').plot(column = col_name,ax=ax,norm = norm, cmap = "viridis",edgecolor ="black")

    ax_inset = inset_axes(ax, width="65%", height="65%", loc="upper right",bbox_to_anchor=(0.745, 0.775, 0.25, 0.25), bbox_transform=ax.transAxes, borderpad=1)  
    data_idf = tmp[tmp['ze2010'].isin(idf_ze)]
    data_idf.query(col_name+' == 0').plot(color = "gray",ax=ax_inset)
    data_idf.query(col_name+' > 0').plot(column = col_name,ax=ax_inset,norm = norm, cmap = "viridis",edgecolor ="black")

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax.set_xlim(-5, 10)
    ax.set_ylim(42, 52)
    ax.set_xticks([])
    ax.set_yticks([])

    add_colorbar(ax,norm)
    return norm


################## Imports and constants #################
folder = "../baseline/"

idf_ze = ['1101', '1111', '1102', '1104', '1118', '1115', '1116', '1105',
       '1117', '1110', '1119', '1112', '1103', '1109', '1106', '1114',
       '1113', '1108', '1107']
       
filter_N_upstream_df = pd.read_csv(os.path.join(folder,"filter_N_upstream.csv"))
filter_N_upstream_df.ze2010 = filter_N_upstream_df.ze2010.astype(str).str.zfill(4)


france = gpd.read_file(os.path.join(folder,"france.shp")).sort_values(by = 'ze2010')

df = load_pi_jA("pi_jA")

########### Plot pi_jA ########


fig,axs = plt.subplots(1,2,figsize = (10,5))

plot_pi_jA("pi_jA",axs[0])
plot_pi_jA("sim_pi_jA",axs[1])

axs[0].set_title(r'Empirical $\pi_{jA}$')
axs[1].set_title(r'Simulated $\pi_{jA}$')

fig.tight_layout()
fig.savefig(os.path.join(folder,'pi_jA.png'),bbox_inches='tight')

################ Plot Toulouse vs Evry ###################
# ze_shocked = "0061"
# fig, axs = plt.subplots(1, 2, figsize=(20, 15))

# ax = axs[0]
# plot_trade_flow("M_ij",ze_shocked,ax)
# ax.set_title(f'{ze_to_shock[ze_shocked]} \n Airbus Aerospace')

# ze_shocked = "1115"
# ax = axs[1]
# plot_trade_flow("M_ij",ze_shocked,ax)
# ax.set_title(f'{ze_to_shock[ze_shocked]} \n Safran Aircraft Engines')

# for i in range(2):    
#     axs[i].set_xticks([])
#     axs[i].set_yticks([])
# fig.tight_layout()
# fig.savefig(os.path.join(folder,'simulation_shocks.png'),bbox_inches='tight')

################ Plot matrix ###################

# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# ax.matshow(M_ij)
# fig.savefig(os.path.join(folder,f"M_ij.png"))


################ Plot 4 regions hit ################


# M_ij = load_M_ij("M_ij")
# i = 0
# fig, ax = plt.subplots(2, 2, figsize=(20, 15))
# for ze_shocked,name_ze_shocked in ze_to_shock.items():
#     tmp = M_ij.query(f'ze2010_j == "{ze_shocked}"')
#     tmp = france.merge(tmp,right_on = "ze2010_i",left_on = "ze2010")
#     tmp['M_ij'] /= tmp['M_ij'].sum()
#     tmp.query('M_ij == 0').plot(color = "gray",ax=ax[i//2,i%2])

#     norm = mcolors.LogNorm(vmin=tmp.query('M_ij > 0').M_ij.min(), vmax=tmp.query('M_ij > 0').M_ij.max())
#     tmp.query('M_ij > 0').plot(column = "M_ij",ax=ax[i//2,i%2],norm = norm, cmap = "viridis")
#     #tmp.plot(column = "M_ij",ax=ax[i//2,i%2], edgecolor="black")

#     ax[i//2,i%2].set_title(name_ze_shocked)

#     tmp.query(f'ze2010_i == "{ze_shocked}"').plot(facecolor="none",ax=ax[i//2,i%2], edgecolor="red",hatch ="//")

#     divider = make_axes_locatable(ax[i//2, i%2])
#     cax = divider.append_axes("right", size="5%", pad=0.1)
#     sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
#     sm.set_array([])  # Dummy array for the scalar mappable
#     fig.colorbar(sm, cax=cax)

#     ax[i//2,i%2].set_xticks([])
#     ax[i//2,i%2].set_yticks([])
#     ax[i//2,i%2].set_title(name_ze_shocked)
    
#     i+=1
# fig.savefig(os.path.join(folder,f"map_high_search_frictions.png"))

    







