# SMM_Spatial_Comovement


## Use

Run the model from main.jl. 

**Parameters**:
    - low_high (bool): if set to true, split rho_si between high and low productive firms. 
    - reduced  (bool): if set to true, moments are average and variance of rho_si and chi_si
    - first_loop (bool): if set to true run the first optimisation of halton grid. Otherwise, run local search. 


**Files**:
    - model_CP.jl  : SMM code for the CP version of the model
    - plot_results.py : Plot the distribution of downstream activities in France.  

    
pip freeze > requirements_py.txt
pip install -r requirements_py.txt