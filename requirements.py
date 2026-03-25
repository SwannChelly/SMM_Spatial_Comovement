import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import argparse
from pathlib import Path
import seaborn as sns
import statsmodels.api as sm


K = -1




# # Set the width of your LaTeX document in points
document_width_pt = 511.  # Adjust this according to your LaTeX template
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
toulouse_color = (132/255, 46/255, 27/255)
missing_color = (0/255, 0/255, 0/255)
label_size = 18
font_size = 15  # Adjust according to your preference
plt.rcParams.update({
    "font.size": font_size,
    "axes.labelsize": font_size,
    "axes.titlesize": font_size+2,
    "xtick.labelsize": font_size-3,
    "ytick.labelsize": font_size-3,
    "legend.fontsize": font_size-3,
    "legend.title_fontsize": font_size,
    "figure.titlesize": font_size,
})





def get_figsize(document_width_pt = document_width_pt, wf=1, hf=0.5):
    """Parameters:
      - wf [float]:  width fraction in columnwidth units
      - hf [float]:  height fraction in columnwidth units.
                     Set by default to golden ratio.
      - columnwidth [float]: width of the column in latex. Get this from LaTeX 
                             using \showthe\columnwidth
    Returns:  [fig_width,fig_height]: that should be given to matplotlib
    """
    fig_width_pt = document_width_pt*wf
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*hf      # height in inches
    return [fig_width, fig_height]



# Plots


def load_pi_r(emp_pi_r):
    pi_r = emp_pi_r
    pi_r = np.array([1-sum(pi_r)] + list(pi_r))
    df = filter_N_upstream_df[["ze2010","pi_r"]].drop_duplicates()
    df.loc[~df.pi_r.isna(),"sim_pi_r"] = pi_r
    df.pi_r.fillna(0,inplace = True)
    df.sim_pi_r.fillna(0,inplace = True)
    return df



def plot_downstream(df,col_name,ax,fig):

    tmp = france.merge(df,on = "ze2010",how = "left")
    tmp[col_name].fillna(0,inplace = True)
    
    if col_name != "productivity":
        vmin = min(df.query('pi_r >0')["pi_r"].to_list())#+df.query('sim_pi_r >0')["sim_pi_r"].to_list())
        vmax = max(df["pi_r"].to_list()+df["sim_pi_r"].to_list())
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else: 
        vmin,vmax = min(df.query('productivity >0').productivity),max(df['productivity'])
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        print(df.productivity.describe())

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

    add_colorbar(ax,fig,norm)
    return norm

def load_productivity(file_name):
    prod = np.load(os.path.join(folder,f"{file_name}.npy"))
    df = filter_N_upstream_df[["ze2010","pi_r"]].drop_duplicates()
    df.loc[~df.pi_r.isna(),"productivity"] = prod
    df.productivity.fillna(0,inplace = True)
    return df

def add_colorbar(ax,fig,norm):
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.02, pad=0.04, aspect=50, format='%1.4f')
    cbar.minorticks_off()

def unpack_simulated_moments(sim_moments, empirical_moments):
    keys = [
        "agg_labor_share",
        "agg_industry_share",
        "emp_gamma_ls",
        "reg_coef",
        "emp_pi_r"
    ]

    sizes = [m.size for m in empirical_moments]
    splits = np.cumsum(sizes)[:-1]
    blocks = np.split(sim_moments, splits, axis=0)

    return {
        k: b.reshape(m.shape + (b.shape[1],))
        for k, m, b in zip(keys, empirical_moments, blocks)
    }


def unpack_params(params, S=9, R_downstream=35, N_beta=5):
    beta = params[0:N_beta]
    labor_share_tech = params[N_beta]
    input_share_tech = params[N_beta+1 : N_beta+1+S]
    input_share_tech = input_share_tech / np.sum(input_share_tech)
    productivity_ = params[N_beta+1+S : N_beta+1+S+R_downstream]
    T_ = params[N_beta+1+S+R_downstream:]
    
    return {
        "beta": beta,
        "labor_share_tech": labor_share_tech,
        "input_share_tech": input_share_tech,
        "productivity": productivity_,
        "T": T_
    }
    
def bubble_scatter(ax, x, y, xlabel, ylabel, title, size_scale=300,regression_line = False):
    mask = x > 0
    x, y = x[mask], y[mask]

    sizes = size_scale * x / x.max()
    lims = [min(x.min(), y.min())*1.2, max(x.max(), y.max())*1.2]
    data = pd.DataFrame({"x":x,"y":y})

    

    ax.scatter(x, y, s=sizes, alpha=0.6, edgecolor="black",color = toulouse_color)
    
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(linestyle = "dashed",alpha = 0.5)
    ols = sm.OLS.from_formula('y ~0+ x', data=data).fit()
    b = ols.params[0]
    ax.text(0.98, 0.09, fr'Coefficient: ${np.round(ols.params[-1],3)}$', ha='right', va='bottom', fontsize=10, color='black', transform=plt.gca().transAxes)
    ax.text(0.98, 0.01, fr't-stat: ${np.round(ols.tvalues[-1],1)}$', ha='right', va='bottom', fontsize=10, color='black', transform=plt.gca().transAxes)
    
    if regression_line:
        X = np.linspace(0,1,100)
        Y = b*X
        ax.plot(X, Y, linestyle='--', color='green')  # '--' for dashed line, linewidth for thickness
    else :
        ax.plot(lims, lims, color="black")
    print(lims)

    sns.despine()
