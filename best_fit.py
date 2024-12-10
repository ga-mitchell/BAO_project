import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt

def best_fit_model(data, templates):
    ''' Finds the best fit isotropic model using chi^2 minimization and plots it versus the date with error'''
    
    print("starting best-fit to data")
    #necessary definitions
    k = [data[0][:,0],data[2][:,0]]
    initial_values = [0,0,0,0,0,0,1,0]
    value_labels = ['Bias','A1','A2','A3','A4','A5','Alpha','Sigma_nl']
    print('starting values are:')
    for i in range(0,len(initial_values)):
        print('\t'+value_labels[i]+': '+str(initial_values[i]))

    #best fit minimization
    parameters = op.basinhopping(optimize, initial_values, minimizer_kwargs = {'args': (k,data,templates),'method':'Nelder-Mead'})
    print('\nBest-Fit is: ')
    for i in range(0,len(initial_values)):
        print('\t'+value_labels[i]+': '+str(parameters['x'][i]))
    print('Chi^2 is', parameters['fun'])
    
    #model
    k_model = np.linspace(.01,.3, num=100)
    best_fit = model(k_model,parameters['x'],templates)/smooth(k_model,parameters['x'],templates[1](k_model))
    k_N = data[0][:,0]
    k_S = data[2][:,0]
    fit_S = data[2][:,1]/smooth(k_S,parameters['x'],templates[1](k_S))
    fit_N = data[0][:,1]/smooth(k_N,parameters['x'],templates[1](k_N))

    #plot and save
    plt.figure()
    plt.plot(k_model, best_fit, label = 'Best Fit Model')
    plt.plot(k_N,fit_N, "*",label='NGC')
    plt.plot(k_S,fit_S,"*",label='SGC')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.title('Best Fit Model vs Data')
    plt.savefig('Plots/bestfit_iso.png')
    
    return parameters['x']

def optimize(parameters,k,data,templates):
    '''calls chi2 and model and is the function minimized for scipy.optimize() ''' 
    if parameters[6] > 1.2 or parameters[6] < .8:
        chi2 = 100000000    
    else:
        pk_model_N = model(k[0],parameters,templates)
        pk_model_S = model(k[1],parameters,templates)
        chi2 = get_chi2(pk_model_N,data[0][:,1],np.linalg.inv(data[1])) + get_chi2(pk_model_S,data[2][:,1],np.linalg.inv(data[3])) 
    return chi2

def get_chi2(model,data,covariance):
    ''' Calculates chi^2 given model parameters, the data and the covariance matrices'''
    chi2 = 0
    sub = model - data
    chi2 = np.dot(sub,np.dot(covariance,sub)) #calculate chi^2 using covariance matrices
    return chi2


def model(k, parameters, templates):
    '''Calculates the model for a given set of parameters and k values and templates'''    
    #calculate the smoothed power spectrum

    pk_sm = smooth(k,parameters,templates[1](k))

    #calculate the power spectrum of the BAO signal
    pk_m = pk_sm*(1+((templates[0](k/parameters[6])-1)*np.exp(-(k**2)*(parameters[7]**2)/2)))
    return pk_m

def smooth(k,parameters,pk_dewiggled):
    ''' Calculates the 'smoothed' power-spectrum with wiggles removed plus broadnband terms and bias '''
    broadband = 0
    for j in range(0,5):    
        poly = parameters[1+j]/k**(3-j) 
        broadband += poly
    pk_sm = (parameters[0]**2)*pk_dewiggled + broadband
    return pk_sm
