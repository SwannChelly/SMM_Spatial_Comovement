# SMM_Spatial_Comovement


## Use

Run the model from main.jl. 

Output of the model are: 
    - report.txt: A file comparing the empirical and simulated moments. 
    - M_ij_high_trade_cost/M_ij_low_trade_cost.npy: M_ij matrix simulated from the best parameter set with 50% lower beta (low) and 50% higher beta (high). 
    - pi_jA / productivity : Simulated pi_jA and best productivity parameters. 


Build reporting using plot_pi_jA.py

The code for the model is stored in model_CP.jl


pip freeze > requirements_py.txt
pip install -r requirements_py.txt