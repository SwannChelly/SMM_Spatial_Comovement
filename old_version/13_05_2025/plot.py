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

def plot_trade_flow(name,ze_shocked,ax,normalize = True,zero_one_norm = False):

    tmp = load_M_ij(name).query(f'ze2010_j == "{ze_shocked}"')
    if normalize:    tmp['M_ij'] /= tmp["M_ij"].sum()
    tmp = france.merge(tmp,right_on = "ze2010_i",left_on = "ze2010")
    if zero_one_norm :
        norm = mcolors.LogNorm(vmin=10**(-4), vmax=1)
    else: 
        norm = mcolors.LogNorm(vmin=tmp.query('M_ij > 0').M_ij.min(), vmax=tmp.query('M_ij > 0').M_ij.max())
    tmp.query('M_ij == 0').plot(color = "gray",ax=ax)
    tmp.query('M_ij > 0').plot(column = "M_ij",ax=ax,norm = norm, cmap = "viridis",edgecolor ="black")
    tmp.query(f'ze2010_i == "{ze_shocked}"').plot(facecolor="none",ax=ax, edgecolor="red",hatch ="//")

    ax_inset = inset_axes(ax, width="65%", height="65%", loc="upper right",bbox_to_anchor=(0.745, 0.775, 0.25, 0.25), bbox_transform=ax.transAxes, borderpad=1)  
    data_idf = tmp[tmp['ze2010_i'].isin(idf_ze)]
    data_idf.query('M_ij == 0').plot(color = "gray",ax=ax_inset)
    data_idf.query('M_ij > 0').plot(column = "M_ij",ax=ax_inset,norm = norm, cmap = "viridis",edgecolor ="black")
    if ze_shocked in idf_ze:
        data_idf.query(f'ze2010_i == "{ze_shocked}"').plot(facecolor="none",ax=ax_inset, edgecolor="red",hatch ="//")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    add_colorbar(ax,norm)
    return norm


################## Imports and constants #################

low_high = True
if low_high: 
    folder = "../bins/"

idf_ze = ['1101', '1111', '1102', '1104', '1118', '1115', '1116', '1105',
       '1117', '1110', '1119', '1112', '1103', '1109', '1106', '1114',
       '1113', '1108', '1107']
       
ze_to_shock = {"1101":"Paris","0061":"Toulouse","9310":"Marseille - Aubagne","5203":"Nantes","8301":"Montluçon","1115":"Évry-Courcouronnes"}

emp_chi_si = np.load(os.path.join(folder,"emp_chi_si.npy"))
N_si = np.load(os.path.join(folder,"N_si.npy"))
filter_A_downstream = np.load(os.path.join(folder,"filter_A_downstream.npy"))
filter_N_upstream = np.load(os.path.join(folder,"filter_N_upstream.npy"))
filter_regions = filter_A_downstream*filter_N_upstream

france = gpd.read_file(os.path.join(folder,"france.shp")).sort_values(by = 'ze2010')


def load_M_ij(name):
    M_ij = np.load(os.path.join(folder,f"{name}.npy"))
    M_ij = pd.DataFrame(M_ij, index=france["ze2010"].values, columns=france["ze2010"].values)
    M_ij.reset_index(inplace=True)
    M_ij.rename(columns={'index': 'ze2010_i'}, inplace=True)
    M_ij = M_ij.melt(id_vars='ze2010_i', var_name='ze2010_j', value_name='M_ij')
    return M_ij


# M_ij = 


########### Plot potential suppliers vs simulation ########

ze_shocked = "0061"
N_si = pd.DataFrame(N_si.sum(axis = 0),columns = ['SIREN'])
N_si['ze2010'] = france['ze2010']

N_firms = france.merge(N_si,on  = "ze2010")
fig, axs = plt.subplots(1, 3, figsize=(20, 15))

ax = axs[0]
norm = mcolors.LogNorm(vmin=200, vmax=N_firms.SIREN.max()+1)
N_firms.query('SIREN <= 200').plot(color = "gray",ax=ax,norm = norm)
N_firms.query('SIREN > 200').plot(column = "SIREN",ax=ax,norm = norm,cmap = "viridis",edgecolor ="black")

ax_inset = inset_axes(ax, width="65%", height="65%", loc="upper right",bbox_to_anchor=(0.745, 0.775, 0.25, 0.25), bbox_transform=ax.transAxes, borderpad=1)  
data_idf = N_firms[N_firms['ze2010'].isin(idf_ze)]
data_idf.query('SIREN <= 200').plot(color = "gray",ax=ax_inset)
data_idf.query('SIREN > 200').plot(column = "SIREN",ax=ax_inset,norm = norm, cmap = "viridis")
ax_inset.set_xticks([])
ax_inset.set_yticks([])

add_colorbar(ax,norm)
ax.set_title(r'# Potential suppliers (Data)')

ax = axs[1]
plot_trade_flow("M_ij_no_search_frictions",ze_shocked,ax,zero_one_norm = True)
ax.set_title(r'No search frictions')

ax = axs[2]
plot_trade_flow("M_ij_high_search_frictions",ze_shocked,ax,zero_one_norm = True)
ax.set_title(r'High search frictions')

for i in range(3):    
    axs[i].set_xticks([])
    axs[i].set_yticks([])

fig.tight_layout()
fig.savefig(os.path.join(folder,'simulation.png'),bbox_inches='tight')

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

    







