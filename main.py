import initial
import best_fit

#initialize power spectrum and covariance matrices for NGC and SGC
print("##### Module 1: Initial.py now running #####")
pk_N, pk_S, cov_N, cov_S = initial.load_data()
data = [pk_N,cov_N,pk_S,cov_S]

#Generate template for interpolation 
Ok, pk_dewiggled = initial.templates()
templates = [Ok,pk_dewiggled]
print("##### Module 1 Complete #####")

#Perform Chi2 minimization to find the best model
print("\n##### Module 2: best_fit.py now running #####")
best_fit_parameters = best_fit.best_fit_model(data,templates)
print("##### Module 2 complete #####")

print("Check the 'Plots' folder to see the output :).")
