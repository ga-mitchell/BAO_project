import numpy as np
import nbodykit.lab as nb
import best_fit as bf
import scipy.optimize as op
import matplotlib.pyplot as plt
import scipy.interpolate as py


def load_data():
    ''' Loads the covariance matrices and the power spectrum '''
    pk_N = np.loadtxt("data/pk_N.dat")
    pk_S = np.loadtxt("data/pk_S.dat")
    cov_N = np.loadtxt("data/cov_N.dat")
    cov_S = np.loadtxt("data/cov_S.dat")
    print("Loaded NGC and SGC power spectrum and covariance matrices.")
    return  pk_N, pk_S, cov_N, cov_S
    
def templates():
    ''' calculates the template power spectrum (Ok = pk_lin/pk_smooth) for the relevant cosmology. See https://nbodykit.readthedocs.io/en/latest/index.html on how to find the linear and dewiggled power spectra '''
    #Generate templates
    cosmology = nb.cosmology.Cosmology(Omega0_cdm=.26185743, Omega0_b=.04814257,h=.676) #cosmology calculator for the relevant parameters
    print("Cosmology is:", cosmology)
    print("Generating templates")
    k = np.linspace(.001,.4,100) #We choose a k range larger than the data in order to interpolate well
    pk_lin = nb.cosmology.power.linear.LinearPower(cosmology,0,transfer="CLASS") #this is the linear power spectrum using the CLASS codes
    pk_dewiggled = nb.cosmology.power.linear.LinearPower(cosmology,0,transfer="NoWiggleEisensteinHu") #this is the linear power spectrum but without the BAO signal using the Eisenstein-Hu method 1998
    print("Templates Generated")
    
    #plot Linear vs NoWiggle
    plt.figure()
    plt.plot(k,k*pk_lin(k), label='Linear Pk')
    plt.plot(k,k*pk_dewiggled(k), label = 'Dewiggled Pk')
    plt.xlabel('k')
    plt.ylabel('kPk')
    plt.title('Linear vs NoWiggleEisensteinHu Power Spectra')
    plt.legend()
    plt.savefig('Plots/linear_v_dewiggle_pk.png')
    
    #Get oscillation signature Ok
    print("Performing best-fit of dewiggled to linear")
    values = [0,0,0,0,0,0] #starting values for minimization
    covariance = np.identity(len(k)) #The covariance can be taken as unity here
    parameters = op.basinhopping(optimize,values,minimizer_kwargs={'args': (k,pk_lin(k),pk_dewiggled(k),covariance),'method': 'Nelder-Mead'}) #We now fit our dewiggled Pk to our linear Pk (see best_fit.py)
    pk_smooth = bf.smooth(k,parameters['x'], pk_dewiggled(k)) #This is our best-fit linear power spectrum 'smoothed' out 
    Ok = pk_lin(k)/pk_smooth #we divide out linear pk by out best fit smoothed pk to get a pk with only the BAO signature which we can then fit to the data 
    print("BAO signal Ok obtained.")
    
    #plot oscillation signature Ok
    plt.figure()
    plt.plot(k,Ok)
    plt.xlabel('k')
    plt.ylabel('Pk')
    plt.title('BAO signal')
    plt.savefig('Plots/Ok.png')
    return py.interp1d(k,Ok), py.interp1d(k,pk_dewiggled(k)) #We need to turn our templates into an interporatable object
    
def optimize(parameters,k,data,templates,covariance):
    '''calls chi2/smooth and is the function minimized for scipy.optimize(). See best_fit.py for more '''
    model = bf.smooth(k,parameters,templates) #calculates the smoothed pk
    chi2 = bf.get_chi2(model,data,covariance) #calculates Chi^2
    return chi2











