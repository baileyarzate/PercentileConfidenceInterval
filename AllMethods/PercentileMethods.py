# -*- coding: utf-8 -*-
#%matplotlib inline
#%matplotlib qt
#%config InlineBackend.figure_format = 'svg'
"""
Created on Tue Mar  7 17:21:23 2023

@author: 1556543727C
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)
import matplotlib as mpl
plt.style.use('fivethirtyeight')
mpl.rcParams['axes.linewidth'] = 1.2
#mpl.rcParams['figure.dpi'] = 200
import sys
sys.path.append("C:/Users/1617290819C/OneDrive - United States Air Force/Python, Tolerance Intervals/")
from tolerance.normtolint import normtolint
import scipy.stats
from scipy.special import digamma
import sys

class Rayleigh(object):
    def __init__( self, filename):
        self._name = filename

              
    def p(self, r, m1, s12, m2, s22):
        '''
        From Weil paper, probability distribution of sqrt(x**2 + y**2) where
        x and y independent normal random variables with unequal means and
        variances.
        
        To get independence, fit a linear model to the data, rotate the 
        coordinate axes so that correlation is 0.  
    
        Parameters
        ----------
        r : TYPE  float
            DESCRIPTION.  positive value, distance from  (m1, m2)
        m1 : TYPE  float
            DESCRIPTION. mean of along-track errors
        s12 : TYPE  float
            DESCRIPTION.  variance of along-track errors
        m2 : TYPE  float
            DESCRIPTION. mean of cross-track errors
        s22 : TYPE
            DESCRIPTION. variancce of cross-track errors
    
        Returns
        -------
        returnProb : TYPE float
            DESCRIPTION.  rayleigh probability density at r
    
        '''
        A = np.exp( -(m1**2*s12 + m2**2*s22)/(2*s12*s22))/np.sqrt(s12*s22)
        a = abs(s12-s22)/(4*s12*s22)
        b = m1/s12; c = m2/s22; d = (b**2+c**2)**0.5

        from math import atan2, cos
        from mpmath import besseli  # bessel functions
        if m1 == 0: ksi = np.pi/2
        else: ksi = atan2((m2*s12)/(m1*s22),1)
        
        prob = A*r*np.exp(-r**2 *(s12 + s22)/(4*s12*s22))
        sumProb = float(besseli(0, (a*r**2))*besseli(0, (d*r)))
        sumBess = 0
        sumBess = float(besseli(1,a*r*r)*besseli(2,(d*r))*cos(2*ksi))
        for j in range(1,50):
            addTerm = float(besseli(j,(a*r**2))*besseli(2*j,(d*r))*cos(2*j*ksi))
            #print(addTerm)
            sumBess += addTerm
            if (abs(addTerm)< 0.0001): 
                #print(f'j: {j}, addTerm: {addTerm}')
                break
        returnProb = prob * (sumProb + 2*sumBess)
        return returnProb
            
    
        ## get the data
    def get_data(self, filename):
        tleData = pd.read_csv(filename)
        list(tleData)
    
        #covariance of the bivariate data
        tleCov = tleData.cov()
        print(f"covariance of the data: {tleCov}")
        arr = tleData.rename(columns={'Range': 'y', 'Azimuth' : 'x'})
        
        return arr
    
    def findPercentile(self, pctile, m1, s12, m2, s22):
        # numerical integration p(r)
        from scipy.integrate import romberg
        bot = 0.0
        top = 4.5
        iterCount = 0

        while (top-bot )> 1.e-5:
            iterCount +=1
            r = (top+bot)/2
            integral = romberg(self.p, 0, r, args=(m1, s12, m2, s22), tol=1.e-2)
            #print(integral)
            if integral > pctile:
                top=r
            else:
                bot=r
            if iterCount >= 25:
                print('used 25 iterations in romberg')
                print(bot, top, r)
                return r
        r = (top+bot)/2
        return r

def signal_handler(sig, frame):
    print("\n\n\n\n\n\n\nManual halt of the code executed.")
    sys.exit(0)
    
def exttolint(x, alpha = 0.05, P = 0.99, dist = 'Weibull', ext = 'min', NRdelta = 1e-8):
    x = np.array(x)
    mx = abs(max(x))+1000
    tempind = 0
    lenc = np.where(np.abs(x) > 1000)
    if len(lenc) > 0:
        tempind = 1
        x = x/mx
    n = len(x)
    x = np.log(x)
    delta = np.sqrt((np.mean(x**2)-np.mean(x)**2)*6/np.pi**2)
    xbar = np.mean(x)
    xi = xbar - digamma(1)
    thetaold = [xi,delta]
    diff = [1]
    while ((diff[0]>NRdelta) > 0) or ((diff[1]>NRdelta) > 0):
        f = sum(x*np.exp(x/delta))
        f1 = -sum(x**2 * np.exp(x/delta))/(delta**2)
        g = sum(np.exp(x/delta))
        g1 = -f/(delta**2)
        d = delta + xbar - (f/g)
        d1 = 1-(g*f1-f*g1)/(g**2)
        deltanew = delta - d/d1
        xinew = -deltanew *np.log(n/sum(np.exp(x/deltanew)))
        deltaold = delta
        xiold = xi
        delta = deltanew
        xi = xinew
        if xi == None or delta == None or delta < 0:
            xi = thetaold[0]
            delta = thetaold[1]
            diff = [NRdelta/5]
        else:
            diff = [abs(deltanew - deltaold), abs(xinew - xiold)]
    def lamb(P):
        return np.log(-np.log(P))
    def kt(x1, x2, n):
        return scipy.stats.nct.ppf(1-x1,df=n-1,nc=(-np.sqrt(n)*lamb(x2)))
    lower = xi - delta * kt(alpha, P, n)/np.sqrt(n - 1)
    upper = xi - delta * kt(1 - alpha, 1 - P, n)/np.sqrt(n - 1)
    a = xi
    b = delta
    a = 1/delta
    b = np.exp(xi)
    lower = np.exp(lower)
    upper = np.exp(upper)
    if tempind == 1:
        b = b*mx
        lower = lower*mx
        upper = upper*mx
    return pd.DataFrame({"alpha":[alpha], "P":[P], "shape.1":[a], "shape.2":[b], "1-sided.lower":lower, "1-sided.upper":upper})

def generate_TLE(type1,meanx=0,meany=0,sdx=100,sdy=500,size = 50):
    if type1.lower() == 'normal' or type1.lower() == 'n':
        cross = np.random.normal(meanx,sdx,size)
        along = np.random.normal(meany,sdy,size)
        return np.array([cross,along])
    if type1.lower() == 'cluster' or type1.lower() == 'c':
        cross = np.random.normal(meanx,sdx,int(size*.8))
        along = np.random.normal(meany,sdy,int(size*.8))
        cross = np.hstack((cross,np.random.normal(100,50,int(size*.2)))) #meanx* = 100, sdx*=50
        along = np.hstack((along,np.random.normal(2000,50,int(size*.2)))) #meany* = 2000, sdy* = 50
        return np.array([cross,along])
    if type1.lower() == 'two' or type1.lower() == 't':
        cross = np.random.normal(meanx,sdx,int(size*.5))
        along = np.random.normal(meany,sdy,int(size*.5))
        cross = np.hstack((cross,np.random.normal(500,100,int(size*.5)))) #meanx* = 500, sdx*=100
        along = np.hstack((along,np.random.normal(1500,500,int(size*.5)))) #meany* = 1500, sdy* = 500
        return np.array([cross, along])
    if type1.lower() == 'outliers' or type1.lower() == 'o':
        cross = np.random.normal(meanx,sdx,int(size*.90))
        along = np.random.normal(meany,sdy,int(size*.90))
        cross = np.hstack((cross,np.random.normal(0,500,int(size*.1))))
        along = np.hstack((along,np.random.normal(0,2500,int(size*.1))))
        return np.array([cross,along])

def sample_plots_2D(x, y):
    np.random.seed(3)
    fig,axs = plt.subplots(2,3)
    type1 = ['normal','cluster', 'two', 'outliers']
    fig.set_facecolor('w')
    axs[0,0].set_facecolor('w')
    axs[0,1].set_facecolor('w')
    axs[1,0].set_facecolor('w')
    axs[1,1].set_facecolor('w')
    axs[1,2].set_facecolor('w')
    pos_left = axs[1, 0].get_position()
    pos_right = axs[1, 1].get_position()
    new_pos_left = [pos_left.x0, pos_left.y0-0.1, pos_left.width, pos_left.height]
    new_pos_right = [pos_right.x0, pos_right.y0-0.1, pos_right.width, pos_right.height]
    axs[1, 0].set_position(new_pos_left)
    axs[1, 1].set_position(new_pos_right)
    pos_left = axs[1,1].get_position()
    pos_right = axs[0, 1].get_position()
    axs[0,1].set_position([pos_right.x0+0.1, pos_right.y0, pos_right.width, pos_right.height])
    axs[1,1].set_position([pos_left.x0+0.1, pos_left.y0, pos_left.width, pos_left.height])
    axs[1,2].set_position([0.85,0.1,0.7,0.7])
    axs[0,0].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[0,0].set_ylabel('Along-Track Error (ft)', fontsize = 8)
    axs[0,1].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[0,1].set_ylabel('Along-Track Error (ft)', fontsize = 8)
    axs[1,0].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[1,0].set_ylabel('Along-Track Error (ft)', fontsize = 8)
    axs[1,1].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[1,1].set_ylabel('Along-Track Error (ft)', fontsize = 8)
    axs[1,2].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[1,2].set_ylabel('Along-Track Error (ft)', fontsize = 8)
    axs[0,0].tick_params(labelsize = 8)
    axs[0,1].tick_params(labelsize = 8)
    axs[1,0].tick_params(labelsize = 8)
    axs[1,1].tick_params(labelsize = 8)
    axs[0,0].set_title('Two', fontsize = 8)
    axs[0,1].set_title('Outliers', fontsize = 8)
    axs[1,0].set_title('Cluster', fontsize = 8)
    axs[1,1].set_title('Normal', fontsize = 8)
    xlistn, ylistn = generate_TLE(type1[0],meanx=0,meany=0,sdx=100,sdy=500,size = 50)
    xlistc, ylistc = generate_TLE(type1[1],meanx=0,meany=0,sdx=100,sdy=500,size = 50)
    xlistt, ylistt = generate_TLE(type1[2],meanx=0,meany=0,sdx=100,sdy=500,size = 50)
    xlisto, ylisto = generate_TLE(type1[3],meanx=0,meany=0,sdx=100,sdy=500,size = 50)
    axs[0,0].set_xlim([-max(max(xlistt),max(ylistt)),max(max(xlistt),max(ylistt))])
    axs[0,0].set_ylim([-max(max(xlistt),max(ylistt)),max(max(xlistt),max(ylistt))])
    axs[0,1].set_xlim([-max(max(xlisto),max(ylisto)),max(max(xlisto),max(ylisto))])
    axs[0,1].set_ylim([-max(max(xlisto),max(ylisto)),max(max(xlisto),max(ylisto))])
    axs[1,0].set_xlim([-max(max(xlistc),max(ylistc)),max(max(xlistc),max(ylistc))])
    axs[1,0].set_ylim([-max(max(xlistc),max(ylistc)),max(max(xlistc),max(ylistc))])
    axs[1,1].set_xlim([-max(max(xlistn),max(ylistn)),max(max(xlistn),max(ylistn))])
    axs[1,1].set_ylim([-max(max(xlistn),max(ylistn)),max(max(xlistn),max(ylistn))])
    axs[1,2].set_xlim([-max(max(x),max(y)),max(max(x),max(y))])
    axs[1,2].set_ylim([-max(max(x),max(y)),max(max(x),max(y))])
    axs[0,0].scatter(xlistt,ylistt,label='Original Data',lw=1, marker = '.',color = 'g', s = 2)
    axs[0,1].scatter(xlisto,ylisto,label='Original Data',lw=1, marker = '.',color = 'g', s = 2)
    axs[1,0].scatter(xlistc,ylistc,label='Original Data',lw=1, marker = '.',color = 'g', s = 2)
    axs[1,1].scatter(xlistn,ylistn,label='Original Data',lw=1, marker = '.',color = 'g', s = 2)
    axs[0,2].set_visible(False)
    axs[1,2].scatter(x,y, lw = 1, marker = '.', color = 'r', s = 2)
    axs[1,2].set_aspect('equal', adjustable='box')
    axs[1,2].set_title('Your Data', fontsize = 10)
    plt.show()

def sample_plots_1D(x, ECDF):
    #np.random.seed(3)
    fig,axs = plt.subplots(2,3)
    type1 = ['normal','cluster', 'two', 'outliers']
    fig.set_facecolor('w')
    axs[0,0].set_facecolor('w')
    axs[0,1].set_facecolor('w')
    axs[1,0].set_facecolor('w')
    axs[1,1].set_facecolor('w')
    axs[1,2].set_facecolor('w')
    pos_left = axs[1, 0].get_position()
    pos_right = axs[1, 1].get_position()
    new_pos_left = [pos_left.x0, pos_left.y0-0.1, pos_left.width, pos_left.height]
    new_pos_right = [pos_right.x0, pos_right.y0-0.1, pos_right.width, pos_right.height]
    axs[1, 0].set_position(new_pos_left)
    axs[1, 1].set_position(new_pos_right)
    pos_left = axs[1,1].get_position()
    pos_right = axs[0, 1].get_position()
    axs[0,1].set_position([pos_right.x0+0.1, pos_right.y0, pos_right.width, pos_right.height])
    axs[1,1].set_position([pos_left.x0+0.1, pos_left.y0, pos_left.width, pos_left.height])
    axs[1,2].set_position([0.85,0.1,0.7,0.7])
    axs[0,0].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[0,1].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[1,0].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[1,1].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[1,2].set_xlabel('Cross-Track Error (ft)', fontsize = 8)
    axs[0,0].tick_params(labelsize = 8)
    axs[0,1].tick_params(labelsize = 8)
    axs[1,0].tick_params(labelsize = 8)
    axs[1,1].tick_params(labelsize = 8)
    axs[0,0].set_title('Two', fontsize = 8)
    axs[0,1].set_title('Outliers', fontsize = 8)
    axs[1,0].set_title('Cluster', fontsize = 8)
    axs[1,1].set_title('Normal', fontsize = 8)
    xlistn = generate_data(type1[0],mean = 0, sd = 500, size = 100)
    xlistc = generate_data(type1[1],mean = 0, sd = 500, size = 100)
    xlistt = generate_data(type1[2],mean = 0, sd = 500, size = 100)
    xlisto = generate_data(type1[3],mean = 0, sd = 100, size = 100)
    if ECDF:
        axs[0,0].plot(np.sort(xlistt), np.linspace(1/xlistt.size, 1, xlistt.size, endpoint = False),label='Original Data',lw=1, marker = '.',color = 'g', ms = 2)
        axs[0,1].plot(np.sort(xlisto), np.linspace(1/xlisto.size, 1, xlisto.size, endpoint = False),label='Original Data',lw=1, marker = '.',color = 'g', ms = 2)
        axs[1,0].plot(np.sort(xlistc), np.linspace(1/xlistc.size, 1, xlistc.size, endpoint = False),label='Original Data',lw=1, marker = '.',color = 'g', ms = 2)
        axs[1,1].plot(np.sort(xlistn), np.linspace(1/xlistn.size, 1, xlistn.size, endpoint = False),label='Original Data',lw=1, marker = '.',color = 'g', ms = 2)
        axs[0,2].set_visible(False)
        axs[1,2].plot(np.sort(x), np.linspace(1/x.size, 1, x.size, endpoint = False), lw = 1, marker = '.', color = 'r', ms = 2)
    else:
        axs[0,0].hist(xlistt)
        axs[0,1].hist(xlisto)
        axs[1,0].hist(xlistc)
        axs[1,1].hist(xlistn)
        axs[0,2].set_visible(False)
        axs[1,2].hist(x)
    #axs[1,2].set_aspect('equal', adjustable='box')
    axs[1,2].set_title('Your Data', fontsize = 10)
    plt.show()

def generate_data(type1, mean = 0, sd = 100, size = 50):
    #np.random.seed(int(np.random.uniform()))
    if type1.lower() == 'normal' or type1.lower() == 'n':
        cross = np.random.normal(mean, sd, size)
        return cross
    if type1.lower() == 'cluster' or type1.lower() == 'c':
        cross = np.random.normal(mean, sd, int(size*.8))
        cross = np.hstack((cross,np.random.normal(100,50,int(size*.2)))) #meanx* = 100, sdx*=50
        return cross
    if type1.lower() == 'two' or type1.lower() == 't':
        cross = np.random.normal(mean, sd, int(size*.5))
        cross = np.hstack((cross,np.random.normal(500,100,int(size*.5)))) #meanx* = 500, sdx*=100
        return cross
    if type1.lower() == 'outliers' or type1.lower() == 'o':
        cross = np.random.normal(mean, sd, int(size*.90))
        cross = np.hstack((cross,np.random.normal(0,500,int(size*.1))))
        return cross

#function is never used with real data
def True_perc(type1='normal',perc=0.9, mean = 0, sd = 100):
    if type1 == 'normal':
        x = generate_data(type1, mean = mean, sd = sd, size = 500_000)
        true_normal = np.percentile(x,(perc)*100) 
        return true_normal
    if type1 == 'two':
        x = generate_data(type1, mean = mean, sd = sd, size = 500_000)
        true_two = np.percentile(x,(perc)*100) 
        return true_two
    if type1 == 'cluster':
        x = generate_data(type1, mean = mean, sd = sd, size = 500_000) 
        true_cluster = np.percentile(x,(perc)*100) 
        return true_cluster
    if type1 == 'outliers':
        x = generate_data(type1, mean = mean, sd = sd, size = 500_000)  
        true_outliers = np.percentile(x,(perc)*100) 
        return true_outliers

import matplotlib.pyplot as plt
#import statsmodels.api as sm

def bounds(data1, data2 = None, DataType = 'normal', Method = 'tolerance', perc = .9, CLevel = 90):
    data1 = np.array(data1)
    try:
        data2 = np.array(data2)
        if len(data1) != len(data2):
            return "Both sets of data must have the same length to compute data."
    except:
        data2 = None
    try:
        radial_err = np.sqrt((data1-np.mean([0]))**2+ (data2-np.mean([1]))**2)
    except:
        radial_err = data1
    
    if Method.lower() == 'tolerance' or Method.lower() == 't' or Method.lower() == 'binomial' or Method.lower() == 'b':
        tol_lower,tol_upper = binom_percentile(radial_err,perc,1-(CLevel/100))
        return sorted([tol_lower,tol_upper])
                
    if Method.lower() == 'binomial smooth' or Method.lower() == 'bs':
        upper_sm = binom_percentile_smooth(radial_err,perc,1-(CLevel/100))
        return upper_sm
                
    if Method.lower() == 'vp':
        lower_vp,upper_vp = vp(data1,perc,1-(CLevel/100))
        return lower_vp,upper_vp

    if Method.lower() == 'binomial 2' or Method.lower() == 'b2':
        lower_binomci,upper_binomci = binom2(radial_err, perc, 1-(CLevel/100))
        return lower_binomci,upper_binomci
        
    if Method.lower() == 'weibull' or Method.lower() == 'w':
        raderrp = radial_err + abs(min(radial_err)) + 2
        lower_weibull = exttolint(raderrp, alpha = 1-(CLevel/100), P = 1-perc, dist = 'Weibull').iloc[:,4][0]
        upper_weibull = exttolint(raderrp, alpha = 1-(CLevel/100), P = perc, dist = 'Weibull').iloc[:,5][0]
        lower_weibull = lower_weibull - abs(min(radial_err)) - 2
        upper_weibull = upper_weibull - abs(min(radial_err)) - 2
        return sorted([lower_weibull,upper_weibull])
    if Method.lower() == 'rayleigh' or Method.lower() == 'r':
        ray = Rayleigh('data')
        dataDict = {'Range': data1, 'Azimuth':data2}
        raderr = (data1**2+data2**2)**0.5
        tleData = pd.DataFrame(data=dataDict)
        from scipy.stats import rayleigh
        rayleighFit = rayleigh.fit(raderr)[1]
        from scipy.linalg import cholesky, pinv
        Ln = cholesky(tleData.cov()).T
        tArr = (pinv(Ln) @ tleData.T).T
        tArr = tArr.rename(columns={0: "x", 1: "y"})
        m1n = np.mean(tArr.x) 
        m2n = np.mean(tArr.y)
        s12n = np.var(tArr.x)
        s22n = np.var(tArr.y)
        #s2n = np.std(tArr.y)
        ray_norm_upper = ray.findPercentile([perc], m1n, s12n, m2n, s22n)*rayleighFit
        return ray_norm_upper

def vp(x, percentile,alpha,iterations = 1000):
    mean_radial_error = np.mean(x)
    std_radial_error = np.std(x,ddof=1)
    pop_r = np.random.normal(mean_radial_error, std_radial_error, size = 1000)
    n = length(x)
    percentile_radial_error = np.percentile(pop_r,percentile*100)
    boot = np.random.choice(pop_r,size =n*iterations, replace=True).reshape((n,iterations))
    mean_boot = np.mean(boot,axis=0)
    std_boot = np.std(boot,ddof=0,axis=0)
    lam = (percentile_radial_error -mean_boot)/std_boot    
    result_lower_conf_CEP = (np.percentile(lam, 100*alpha/2) *std_radial_error) + (mean_radial_error)
    result_upper_conf_CEP = (np.percentile(lam, 100*((1-alpha) + 0.05/2)) *std_radial_error) + (mean_radial_error)
    return(result_lower_conf_CEP,result_upper_conf_CEP)

def binom_percentile(x, percentile,alpha):
    n = len(x)
    q = int(stats.binom.ppf(alpha,n,1-percentile))
    if q == 0:
        res = np.max(x)
    else:
        try:
            res = np.sort(x)[n-q]
        except:
            res = np.inf
    q = int(stats.binom.ppf(alpha,n,percentile))
    if q == 0:
        low = np.min(x)
    else:
        try:
            low = np.sort(x)[q]
        except:
            low = 0
    return [low,res]

def binom_percentile_smooth(x,percentile,alpha):
    #n = x.shape[0]
    #r = (x['Range']**2+x['Azimuth']**2)**.5
    n = len(x)
    r = x
    q = int(stats.binom.ppf(1-alpha,n,percentile))
    res = np.sort(r)[q-1]
    w = [0,.5,.5]
    if q < len(r):
        res = (w[0]*np.sort(r)[q-2] + w[1]*res + w[2]*np.sort(r)[q])
    return(res) 

def binom2(x, percentile = 0.9, alpha = 0.05):
    try:
        n = len(x)
    except:
        n = 1
    bp = list(range(1,n+1))
    bp = pd.DataFrame(stats.binom.pmf(k = bp, n = n , p = percentile))
    ordered_bp_val = bp.sort_values(by = [0],ascending=False)
    ordered_bp_ind = np.array(ordered_bp_val.index)
    ordered_bp_val = np.array(ordered_bp_val.iloc[:,0])
    coverage = 0
    indices = list(range(n))
    for i in range(n):
        coverage += ordered_bp_val[i]
        indices[i] = ordered_bp_ind[i]
        if coverage >= 1-alpha:
            break
    ind_in = ordered_bp_ind[0:i+1]
    y = np.sort(x)
    return [y[min(ind_in)], y[max(ind_in)]]

def Plot(data1, data2 = None, DataType = 'normal', Method = 'tolerance', perc = 90, CLevel = 90, ECDF = True):
    #upper_bound = bounds(data1, DataType = DataType, Method = Method, perc = perc, CLevel = CLevel)
    if data2 is not None:
        upper_bound = bounds(data1,data2,DataType = DataType, Method = Method, perc = perc, CLevel = CLevel)
        def True_perc(type1='normal',perc=0.9,meanx=0,meany=0,sdx=100,sdy=500):
            if type1.lower() == 'normal' or type1.lower() == 'n':
                xlist1, ylist1 = generate_TLE(type1,meanx = meanx,meany = meany,sdx=sdx,sdy=sdy,size = 500_000)
                raderr = (np.sqrt((xlist1-meanx)**2 + (ylist1-meany)**2))
                true_raderr_normal = np.percentile(raderr,(perc)*100) 
                return true_raderr_normal
            if type1.lower() == 'two' or type1.lower() == 't':
                xlist1, ylist1 = generate_TLE(type1.lower(),meanx = meanx,meany = meany,sdx=sdx,sdy=sdy,size = 500_000)
                raderr = (np.sqrt((xlist1-meanx)**2 + (ylist1-meany)**2))
                true_raderr_two = np.percentile(raderr,(perc)*100) 
                return true_raderr_two
            if type1.lower() == 'cluster' or type1.lower() == 'c':
                xlist1, ylist1 = generate_TLE(type1.lower(),meanx = meanx,meany = meany,sdx=sdx,sdy=sdy,size = 500_000) 
                raderr = (np.sqrt((xlist1-meanx)**2 + (ylist1-meany)**2))
                true_raderr_cluster = np.percentile(raderr,(perc)*100) 
                return true_raderr_cluster
            if type1.lower() == 'outliers' or type1.lower() == 'o':
                xlist1, ylist1 = generate_TLE(type1.lower(),meanx = meanx,meany = meany,sdx=sdx,sdy=sdy,size = 500_000)  
                raderr = (np.sqrt((xlist1-meanx)**2 + (ylist1-meany)**2))
                true_raderr_outliers = np.percentile(raderr,(perc)*100) 
                return true_raderr_outliers
    
        #true_raderr = True_perc(type1 = DataType, perc = perc)    
    
        if Method.lower() == 'avg' or Method.lower() == 'all':
            upper_bound1 = bounds(data1,data2,DataType = DataType, Method = 'tolerance', perc = perc, CLevel = CLevel)[1]
            upper_bound2 = bounds(data1,data2,DataType = DataType, Method = 'binomial smooth', perc = perc, CLevel = CLevel)
            upper_bound3 = bounds(data1,data2,DataType = DataType, Method = 'vp', perc = perc, CLevel = CLevel)[1]
            upper_bound4 = bounds(data1,data2,DataType = DataType, Method = 'binomial 2', perc = perc, CLevel = CLevel)[1]
            upper_bound5 = bounds(data1,data2,DataType = DataType, Method = 'weibull', perc = perc, CLevel = CLevel)[1]
            upper_bound6 = bounds(data1,data2,DataType = DataType, Method = 'rayleigh', perc = perc, CLevel = CLevel)
            AVG_UPP_BOUND = (upper_bound1+upper_bound3+upper_bound4+upper_bound5)/4
            
            lower_bound1 = bounds(data1,data2,DataType = DataType, Method = 'tolerance', perc = perc, CLevel = CLevel)[0]
            lower_bound3 = bounds(data1,data2,DataType = DataType, Method = 'vp', perc = perc, CLevel = CLevel)[0]
            lower_bound4 = bounds(data1,data2,DataType = DataType, Method = 'binomial 2', perc = perc, CLevel = CLevel)[0]
            lower_bound5 = bounds(data1,data2,DataType = DataType, Method = 'weibull', perc = perc, CLevel = CLevel)[0]
            AVG_LOW_BOUND = (lower_bound1+lower_bound3+lower_bound4+lower_bound5)/4
            
            
        t = np.linspace(0,2*np.pi,1000)
        fig,ax = plt.subplots(dpi = 150)
        ax.set_aspect('equal')
        fig.set_facecolor('w')
        ax.set_facecolor('w')
        ax.set_xlabel('Cross-Track Error (ft)', fontsize = 8)
        ax.set_ylabel('Along-Track Error (ft)', fontsize = 8)
        #true_raderr_normal = True_perc()
        #ax.plot(true_raderr_normal*np.cos(t),true_raderr_normal*np.sin(t),ls='--',color='cyan',label = 'True 90th percentile', lw=1)
    
        
        ax.scatter(data1,data2,label='Original Data',lw=1, marker = '.',color = 'g', s = 2)
        #ax.set_title(f'{DataType}', fontsize = 8)
        #ax.plot(true_raderr*np.cos(t),true_raderr*np.sin(t),ls='-',color='cyan',label = 'True Limit', lw=1)
        
        if Method.lower() == 'tolerance' or Method.lower() == 't':
            ax.plot(upper_bound[1]*np.cos(t),upper_bound[1]*np.sin(t),ls='--',color='orange',label = 'Upper Tolerance Limit', lw=1)
            ax.plot(upper_bound[0]*np.cos(t),upper_bound[0]*np.sin(t),ls='--',color='orange',label = 'Lower Tolerance Limit', lw=1)
    
        if Method.lower() == 'binomial smooth' or Method.lower() == 'bs':
            ax.plot(upper_bound*np.cos(t),upper_bound*np.sin(t),ls='--',color='orange',label = 'Upper Binomial Smooth Limit', lw=1)
            
        if Method.lower() == 'binomial 2' or Method.lower() == 'b2':
            ax.plot(upper_bound[1]*np.cos(t),upper_bound[1]*np.sin(t),ls='--',color='orange',label = 'Upper Binomial #2 Limit', lw=1)
            ax.plot(upper_bound[0]*np.cos(t),upper_bound[0]*np.sin(t),ls='--',color='orange',label = 'Lower Binomial #2 Limit', lw=1)
           
        if Method.lower() == 'vp':
            ax.plot(upper_bound[1]*np.cos(t),upper_bound[1]*np.sin(t),ls='--',color='orange',label = 'Upper VP Limit', lw=1)
            ax.plot(upper_bound[0]*np.cos(t),upper_bound[0]*np.sin(t),ls='--',color='orange',label = 'Lower VP Limit', lw=1)
        
        if Method.lower() == 'w' or Method.lower() == 'weibull':
            ax.plot(upper_bound[1]*np.cos(t),upper_bound[1]*np.sin(t),ls='--',color='orange',label = 'Upper Weibull Limit', lw=1)
            ax.plot(upper_bound[0]*np.cos(t),upper_bound[0]*np.sin(t),ls='--',color='orange',label = 'Lower Weibull Limit', lw=1)
            
        if Method.lower() == 'r' or Method.lower() == 'rayleigh':
            ax.plot(upper_bound*np.cos(t),upper_bound*np.sin(t),ls='--',color='orange',label = 'Upper Rayleigh Limit', lw=1)
            
        if Method.lower() == 'avg':
            ax.plot(AVG_UPP_BOUND*np.cos(t),AVG_UPP_BOUND*np.sin(t),ls='-',color='black',label = 'Upper Average Limit', lw=1)
            ax.plot(AVG_LOW_BOUND*np.cos(t),AVG_LOW_BOUND*np.sin(t),ls='-',color='black',label = 'Lower Average Limit', lw=1)
        
        if Method.lower() == 'all':
            lowbool = input('Do you want to see:\n\t(U) Upper bounds\n\t(L) Lower bounds\n\t(B) Both Bounds\nDesired Bounds: ')
            if lowbool.lower() == 'u' or lowbool.lower() == 'b':
                ax.plot(upper_bound1*np.cos(t),upper_bound1*np.sin(t),ls='--',color='red',label = 'Upper Tolerance Limit', lw=1)
                ax.plot(upper_bound2*np.cos(t),upper_bound2*np.sin(t),ls='--',color='green',label = 'Upper Binomial Smooth Limit', lw=1)
                ax.plot(upper_bound3*np.cos(t),upper_bound3*np.sin(t),ls='--',color='orange',label = 'Upper VP Limit', lw=1)
                ax.plot(upper_bound4*np.cos(t),upper_bound4*np.sin(t),ls='--',color='blue',label = 'Upper Binomial #2 Limit', lw=1)
                ax.plot(upper_bound5*np.cos(t),upper_bound5*np.sin(t),ls='--',color='violet',label = 'Upper Weibull Limit', lw=1)
                ax.plot(upper_bound6*np.cos(t),upper_bound6*np.sin(t),ls='--',color='pink',label = 'Upper Rayleigh Limit', lw=1)
                ax.plot(AVG_UPP_BOUND*np.cos(t),AVG_UPP_BOUND*np.sin(t),ls='--',color='black',label = 'Upper Average Limit', lw=1)
            if lowbool.lower() == 'l' or lowbool.lower() == 'b':
                ax.plot(lower_bound1*np.cos(t),lower_bound1*np.sin(t),ls='-.',color='red',label = 'Lower Tolerance Limit', lw=1)
                ax.plot(lower_bound3*np.cos(t),lower_bound3*np.sin(t),ls='-.',color='orange',label = 'Lower VP Limit', lw=1)
                ax.plot(lower_bound4*np.cos(t),lower_bound4*np.sin(t),ls='-.',color='blue',label = 'Lower Binomial #2 Limit', lw=1)
                ax.plot(lower_bound5*np.cos(t),lower_bound5*np.sin(t),ls='-.',color='violet',label = 'Lower Weibull Limit', lw=1)
                ax.plot(AVG_UPP_BOUND*np.cos(t),AVG_UPP_BOUND*np.sin(t),ls='-.',color='black',label = 'Lower Average Limit', lw=1)
        ax.legend(loc = 0,title = "Limits", bbox_to_anchor=(1.04, 1), prop={'size': 8})
        # ax.legend(loc = 'lower left', prop={'size': 5})
        ax.tick_params(labelsize = 7)
        
        if Method.lower() != 'all' and Method.lower() != 'avg':
            return upper_bound
        elif Method.lower() == 'avg':
            return [AVG_LOW_BOUND,AVG_UPP_BOUND]
        elif Method.lower() == 'all':
            return [lower_bound1,upper_bound1],upper_bound2,[lower_bound3,upper_bound3],[lower_bound4,upper_bound4],[AVG_LOW_BOUND,AVG_UPP_BOUND],[lower_bound5,upper_bound5],upper_bound6
    if data2 is None:
        # def True_perc(type1 = 'normal',perc = 0.9,mean = 0, sd = 100):
        #     if type1.lower() == 'normal' or type1.lower() == 'n':
        #         xlist1 = generate_data(type1.lower(), mean = mean, sd = sd, size = 500_000)
        #         true_perc_normal = np.percentile(xlist1,(perc)*100) 
        #         return true_perc_normal
        #     if type1.lower() == 'two' or type1.lower() == 't':
        #         xlist1 = generate_data(type1.lower(), mean = mean, sd = sd, size = 500_000)
        #         true_perc_two = np.percentile(xlist1,(perc)*100) 
        #         return true_perc_two
        #     if type1.lower() == 'cluster' or type1.lower() == 'c':
        #         xlist1 = generate_data(type1.lower(), mean = mean, sd = sd, size = 500_000)
        #         true_perc_cluster = np.percentile(xlist1,(perc)*100) 
        #         return true_perc_cluster
        #     if type1.lower() == 'outliers' or type1.lower() == 'o':
        #        xlist1 = generate_data(type1.lower(), mean = mean, sd = sd, size = 500_000)
        #        true_perc_outliers = np.percentile(xlist1,(perc)*100) 
        #        return true_perc_outliers
        
        if Method.lower() == 'avg' or Method.lower() == 'all':
            upper_bound1 = bounds(data1,DataType = DataType, Method = 'tolerance', perc = perc, CLevel = CLevel)[1]
            upper_bound2 = bounds(data1,DataType = DataType, Method = 'binomial smooth', perc = perc, CLevel = CLevel)
            upper_bound3 = bounds(data1,DataType = DataType, Method = 'vp', perc = perc, CLevel = CLevel)[1]
            upper_bound4 = bounds(data1,DataType = DataType, Method = 'binomial 2', perc = perc, CLevel = CLevel)[1]
            upper_bound5 = bounds(data1,DataType = DataType, Method = 'weibull', perc = perc, CLevel = CLevel)[1]
            # upper_bound6 = bounds(data1,data2,DataType = DataType, Method = 'rayleigh', perc = perc, CLevel = CLevel)[1]
            AVG_UPP_BOUND = (upper_bound1+upper_bound2+upper_bound5)/3
            lower_bound1 = bounds(data1,DataType = DataType, Method = 'tolerance', perc = perc, CLevel = CLevel)[0]
            lower_bound3 = bounds(data1,DataType = DataType, Method = 'vp', perc = perc, CLevel = CLevel)[0]
            lower_bound4 = bounds(data1,DataType = DataType, Method = 'binomial 2', perc = perc, CLevel = CLevel)[0]
            lower_bound5 = bounds(data1,DataType = DataType, Method = 'weibull', perc = perc, CLevel = CLevel)[0]
            AVG_LOW_BOUND = (lower_bound1+lower_bound4+lower_bound5)/3
        else:
            upper_bound = bounds(data1, DataType = DataType, Method = Method, perc = perc, CLevel = CLevel)
            
            
        #t = np.linspace(0,2*np.pi,1000)
       
    
        fig, ax = plt.subplots()
        if ECDF:
            ax.plot(np.sort(data1), np.linspace(1/data1.size, 1, data1.size, endpoint = False), label='Empirical CDF',lw=1, marker = '.',color = 'g')
            ax.set_yticks(np.arange(0,1.1,0.1))
            ax.set_xlim([min(data1)*1.25,max(data1)*1.25])
        if not ECDF:
            ax.hist(data1,label='Original Data',lw=1,color = 'g')
            
        #true_raderr = True_perc(type1 = DataType, perc = perc)    
        #ax.axvline(true_raderr,ls='-',color='cyan',label = 'True Limit', lw=1)
            
        ax.set_title(DataType, fontsize = 8)
        
        if Method.lower() == 'tolerance' or Method.lower() == 't':
           ax.axvline(upper_bound[1],ls='--',color='orange',label = 'Upper Tolerance Limit', lw=1)
           ax.axvline(upper_bound[0],ls='--',color='orange',label = 'Lower Tolerance Limit', lw=1)
    
        if Method.lower() == 'binomial smooth' or Method.lower() == 'bs':
            ax.axvline(upper_bound,ls='--',color='orange',label = 'Upper Binomial Smooth Limit', lw=1)
            
        if Method.lower() == 'binomial 2' or Method.lower() == 'b2':
            ax.axvline(upper_bound[1],ls='--',color='orange',label = 'Upper Binomial #2 Limit', lw=1)
            ax.axvline(upper_bound[0],ls='--',color='orange',label = 'Lower Binomial #2 Limit', lw=1)
           
        if Method.lower() == 'vp':
            ax.axvline(upper_bound[1],ls='--',color='orange',label = 'Upper VP Limit', lw=1)
            ax.axvline(upper_bound[0],ls='--',color='orange',label = 'Lower VP Limit', lw=1)
        
        if Method.lower() == 'w' or Method.lower() == 'weibull':
            ax.axvline(upper_bound[1],ls='--',color='orange',label = 'Upper Weibull Limit', lw=1)
            ax.axvline(upper_bound[0],ls='--',color='orange',label = 'Lower Weibull Limit', lw=1)
            
        if Method.lower() == 'avg':
            ax.axvline(AVG_UPP_BOUND,ls='-',color='black',label = 'Upper Average Limit', lw=1)
            ax.axvline(AVG_LOW_BOUND,ls='-',color='black',label = 'Lower Average Limit', lw=1)
            
        if Method.lower() == 'all':
            lowbool = input('Do you want to see:\n\t(U) Upper bounds\n\t(L) Lower bounds\n\t(B) Both Bounds\nDesired Bounds: ')
            if lowbool.lower() == 'u' or lowbool.lower() == 'b':
                ax.axvline(upper_bound1,ls='-',color='orange',label = 'Upper Tolerance Limit', lw=0.7)
                ax.axvline(upper_bound2,ls='-',color='green',label = 'Upper Binomial Smooth Limit', lw=0.7)
                ax.axvline(upper_bound3,ls='-',color='red',label = 'Upper VP Limit', lw=0.7)
                ax.axvline(upper_bound4,ls='-',color='blue',label = 'Upper Binomial #2 Limit', lw=0.7)
                ax.axvline(upper_bound5,ls='-',color='violet',label = 'Upper Weibull Limit', lw=0.7)
                ax.axvline(AVG_UPP_BOUND,ls='-',color='black',label = 'Upper Average Limit', lw=0.7)
            if lowbool.lower() == 'l' or lowbool.lower() == 'b':
                ax.axvline(lower_bound1,ls='--',color='orange',label = 'Lower Tolerance Limit', lw=0.7)
                ax.axvline(lower_bound3,ls='--',color='red',label = 'Lower VP Limit', lw=0.7)
                ax.axvline(lower_bound4,ls='--',color='blue',label = 'Lower Binomial #2 Limit', lw=0.7)
                ax.axvline(lower_bound5,ls='--',color='violet',label = 'Lower Weibull Limit', lw=0.7)
                ax.axvline(AVG_LOW_BOUND,ls='--',color='black',label = 'Lower Average Limit', lw=0.7)
            
        ax.legend(loc = 0,bbox_to_anchor=(1.04, 1), prop={'size': 8})
        ax.tick_params(labelsize = 7)
        
        if Method.lower() != 'all' and Method.lower() != 'avg':
            return upper_bound
        elif Method.lower() == 'avg':
            return [AVG_LOW_BOUND,AVG_UPP_BOUND]
        elif Method.lower() == 'all':
            return [lower_bound1,upper_bound1],upper_bound2,[lower_bound3,upper_bound3],[lower_bound4,upper_bound4],[AVG_LOW_BOUND,AVG_UPP_BOUND],[lower_bound5,upper_bound5]

def main(data1, data2 = None, ECDF = True):
    #Handling Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    sample_size = len(data1)
    if data2 is None:    
        R1 = 0
        while True:
            R1 = input('What percentile are you looking for, 50 or 90? ')
            R1 = float(R1)
            if R1 <= 0:
                print('The {R1}th percentile upper bound is undefined\n\n Consider having percentile value greater than 0')
                sys.exit(0)
            if R1 >= 100:
                print(f'The {R1}th percentile upper bound is undefined\n\n Consider having percentile value less than 100')
                sys.exit(0)
            R1 = R1/100
            CLevel = float(input("What confidence level are you looking for? (0-100): "))
            if CLevel <= 0:
                print('The 90th percentile upper bound is: 0\n\n Consider having confidence level greater than 0')
                sys.exit(0)
            if CLevel >= 100:
                print(f'The 90th percentile upper bound is: {np.inf}\n\n Consider having confidence level less than 100')
            
            if R1 == .50 or R1 == .90:
                break
            else:
                R3 = input(f"Your chosen percentile is {R1}. Are you sure you want to continue? Results haven\'t been tested for this percentile. (y or n): ")
                if R3.lower() == 'y' or R3.lower() == 'yes':
                    break
                else:
                    continue
    
        while True:
            R2 = input(f'Is your sample size: {sample_size}, (y or n)? ')
            if R2.lower() == 'n' or R2.lower() == 'no':
                R2 = int(input('What\'s your sample size? '))
                break
            elif R2.lower() == 'y' or R2.lower() == 'yes':
                R2 = sample_size
                break
        
        sample_plots_1D(data1, ECDF)
        while True:
            DataType = input('What type of scatterplot does your data most resemble? Plots of sample data are shown: Options are: \n\tnormal (n)\n\toutliers(o)\n\tcluster(c)\n\ttwo (t)\n\tother\nData Type: ')
            if (DataType.lower() == 'normal' or DataType.lower() == 'n' or DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'cluster'
                 or DataType.lower() == 'c' or DataType.lower() == 'outliers' or DataType.lower() == 'o' or DataType.lower() == 'other'):
                break
            else:
                print('Data Type must be a valid input. Use the letter in the parenthesis or type \'other\'')
                continue
        if R1 == .90:
            if R2 == 2:
                if DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend using the Tolerance method.")
                elif DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend using the VP method.")
                elif DataType.lower() == 'outliers' or DataType.lower() == 'o' or DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'other':
                    print("We don't recommend using any of these methods.")
            elif 3 <= R2 <= 20:
               print("We recommend using the VP method.")
            elif R2 == 21:
                if DataType.lower() == 'normal' or DataType.lower() == 'n' or DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print("We recommend using the VP method.")
                elif DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend using the Tolerance method.")
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
            elif 22 <= R2 <= 23:
                if DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend using the Binomial Smooth method")
                elif DataType.lower == 'outliers' or DataType.lower == 'o':
                    print("We recommend using the VP method")
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
            elif 24 <= R2 <= 26:
                if DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend using the Tolerance method")
                elif DataType.lower == 'outliers' or DataType.lower == 'o':
                    print("We recommend using the VP method")
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
            elif 27 <= R2 <= 29:
                if DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend using the Tolerance method")
                elif DataType.lower == 'outliers' or DataType.lower == 'o' or DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend using the VP method")
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
            elif R2 >= 30:
                if DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print("We recommend using the Binomial #2 method")
                elif DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print('We recommend using the VP method.')
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
        elif R1 == .50:
            if R2 < 10:
                print("We recommend using the VP method")
            elif 11 <= R2 <= 50:
                if DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print('We recommend using the VP method.')  
                elif DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print('We recommend using the Binomial #2 method.')  
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
            elif R2 > 50:
                if DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print('We recommend using the VP method.')  
                elif DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print('We recommend using the Binomial #2 method.')  
                elif DataType.lower() == 'cluster' or DataType.lower() == 'c':
                    print('We recommend using the Binomial Smooth method.')  
                elif DataType.lower() == 'other':
                    print('\tPlotting all methods may be useful here.')
        else:
            print("Warning:\n    Untested percentiles are being used. Bounds may overestimate or underestimate significantly.")
            print('\n We recommend using the tolerance or Binomial #2 method.')
            print('\n We also recommend looking at all methods here too.')
        
        while True:
            Method = input("What method would you like to use? Options are:\n\t(VP) Vysochanskij–Petunin\n\t(T) Tolerance \n\t(W) Weibull\n\t(B2) Binomial #2\n\t(BS) Binomial Smooth\n\t(ALL) All methods\n\t(AVG) Average of all methods [Not recommended]\n Desired Method: ")
            if (Method.lower() == 'tolerance' or Method.lower() == 't' or Method.lower() == 'vp' or Method.lower() == 'binomial 2'
                or Method.lower() == 'b2' or Method.lower() == 'binomial smooth' or Method.lower() == 'bs'
                or Method.lower() == 'all' or Method.lower() == 'weibull' or Method.lower() == 'w' or Method.lower() == 'avg'):
                break
            else:
                print("Invalid method inputted.\n")
                continue
        upperBound = Plot(data1, DataType = DataType.lower(), Method = Method.lower(), perc = R1, CLevel = CLevel)
        #raderr = np.sqrt(data1**2 + data2**2)
        if Method.lower() != 'all':
            for i in range(length(upperBound)):
                try:
                    if upperBound[i] < 0:
                        upperBound[i] = abs(upperBound[i])
                except:
                    if upperBound < 0:
                        upperBound = abs(upperBound)
                    if R1 < .01:
                        upperBound = np.min(data1)
            try:
                upperBound = sorted(upperBound)
            except:
                upperBound = upperBound
                upperBound = [0, upperBound]
            try:
                print(f'\nThe {int(R1*100)}th percentile {int(CLevel)}% confidence interval for the data relative to the origin using the {Method} method is: {np.round(upperBound,4)}')
            except:
                print(f'\nThe {R1*100}th percentile {CLevel} confidence interval for the data relative to the origin using the {Method} method is: {np.round(upperBound,4)}')
        else:
            upperBound= list(upperBound)
            for j in range(length(upperBound)):
                for i in range(length(upperBound[j])):
                    try:
                        if upperBound[j][i] < 0:
                            upperBound[j][i] = abs(upperBound[j][i])
                    except:
                        if upperBound[j] < 0:
                            upperBound[j] = abs(upperBound[j])   
                        if R1 < .01:
                            upperBound[j] = np.min(data1)
            upperBound = [sorted(u) if u is list else u for u in upperBound]
            
            print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the data relative to the origin using the Tolerance Method is: {np.round(upperBound[0],4)}')
            print(f'\nThe {(R1*100)}th percentile {(CLevel)}% upper bound for the data relative to the origin using the Binomial Smooth is: {np.round(upperBound[1],4)}')
            print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the data relative to the origin using the VP is: {np.round(upperBound[2],4)}')
            print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the data relative to the origin using the Binomial #2 Method is: {np.round(upperBound[3],4)}')
            print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the data relative to the origin using the Weibull Method is: {np.round(upperBound[4],4)}')
            print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the data relative to the origin using the average of all the methods is: {np.round(upperBound[5],4)}')
    elif data2 is not None:
        def sample_size_confirmation(x, y):
            if len(x) != len(y):
                print('Both datasets must have the same length. Fix, and then rerun the code.')
                sys.exit(0)
        sample_size_confirmation(data1, data2)
        sample_size = len(data1)
        R1 = 0
        while True:
            R1 = input('What percentile are you looking for, 50 or 90? ')
            R1 = float(R1)
            if R1 <= 0:
                print('The {R1}th percentile upper bound is undefined\n\n Consider having percentile value greater than 0')
                sys.exit(0)
            if R1 >= 100:
                print(f'The {R1}th percentile upper bound is undefined\n\n Consider having percentile value less than 100')
                sys.exit(0)
            R1 = R1/100
            CLevel = float(input("What confidence level are you looking for? (0-100): "))
            if CLevel <= 0:
                print('The 90th percentile upper bound is: 0\n\n Consider having confidence level greater than 0')
                sys.exit(0)
            if CLevel >= 100:
                print(f'The 90th percentile upper bound is: {np.inf}\n\n Consider having confidence level less than 100')
            
            if R1 == .50 or R1 == .90:
                break
            else:
                R3 = input(f"Your chosen percentile is {R1}. Are you sure you want to continue? Results haven\'t been tested for this percentile. (y or n): ")
                if R3.lower() == 'y' or R3.lower() == 'yes':
                    break
                else:
                    continue
    
        while True:
            R2 = input(f'Is your sample size: {sample_size}, (y or n)? ')
            if R2 == 'n' or R2 == 'no':
                R2 = int(input('What\'s your sample size? '))
                break
            elif R2 == 'y' or R2 == 'yes':
                R2 = sample_size
                break
        
        sample_plots_2D(data1,data2)
        while True:
            DataType = input('What type of scatterplot does your data most resemble? Plots of sample data are shown: Options are: \n\tnormal (n)\n\toutliers(o)\n\tcluster(c)\n\ttwo (t)\n\tother\nData Type: ')
            if (DataType.lower() == 'normal' or DataType.lower() == 'n' or DataType.lower() == 'two' or DataType.lower() == 't' or DataType.lower() == 'cluster'
                 or DataType.lower() == 'c' or DataType.lower() == 'outliers' or DataType.lower() == 'o' or DataType.lower() == 'other'):
                break
            else:
                print('Data Type must be a valid input. Use the letter in the parenthesis or type \'other\'')
                continue
        if R1 == .90:
            if 2 <= R2 <=10:
                if DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend using the Rayleigh method")
                else:
                    print("We recommend using the VP method")
            elif 11 <= R2 <= 14:
                if DataType.lower() != 'other':
                    print('We recommend:')
                    print('    If you want higher confidence with less precision, use the VP method')
                    print('    If you want more precision but with lower confidence, use the Tolerance method')
                if DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
            elif 15 <= R2 <= 20:
                if DataType.lower() == 'cluster' or DataType.lower() == 'c' or DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the VP method')
                    print('    If you want more precision but with lower confidence, use the Tolerance method')
                elif DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the VP method')
                    print('    If you want more precision but with lower confidence, use the Weibull method')
                elif DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print("    We recommend using the Weibull method")
                elif DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
            elif 21 <= R2 <= 30:
                print('\n')
                if DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the VP method')
                    print('    If you want more precision with lower confidence, use the Weibull method')
                if DataType.lower() == 'outliers' or DataType.lower() == 'o':
                        print("We recommend: ")
                        print('    If you want higher confidence with less precision, use the VP method')
                        print('    If you want more precision but with lower confidence, use the Weibull method')
                elif DataType.lower() == 'cluster' or DataType.lower() == 'c':
                    print("    We recommend using the Binomial #2 method")
                elif DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend: ")
                    print('    Using the Binomial Smooth method.')
                    print('    If you want the results to be more precise but with lower confidence, use the Binomial #2 method')
                elif DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
            elif 31 <= R2 <= 35:
                print('\n')
                if DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the VP method')
                    print('    If you want more precision with lower confidence, use the Weibull method')
                elif DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print("    We recommend using the Weibull method")
                elif DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the Tolerance method')
                    print('    If you want more precision but with lower confidence, use the Binomial #2 method')
                elif DataType.lower() == 'cluster' or DataType.lower() == 'c':
                    print("    We recommend using the Binomial #2 method")
                elif DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
            elif R2 > 35:
                print('\n')
                if DataType.lower() == 'normal' or DataType.lower() == 'n':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the VP method')
                    print('    If you want more precision with lower confidence, use the Weibull method')
                elif DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print("We recommend using the Weibull method")
                elif DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend using the Binomal Smooth method")
                elif DataType.lower() == 'cluster' or DataType.lower() == 'c':
                    print("We recommend using the Binomal #2 method")
                elif DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
                    
        elif R1 == .50:
            if 2 <= R2 <= 4:
                print("We recommend using the VP method")
            elif 5 <= R2 <= 10:
                print('\n')
                if R2 == 5:
                    if DataType.lower() == 'normal' or DataType.lower() == 'n':
                        print("We recommend: ")
                        print('    If you want higher confidence with less precision, use the VP method')
                        print('    If you want more precision with lower confidence, use the Binomial #2 method')
                else:
                    if DataType.lower() == 'normal' or DataType.lower() == 'n':
                        print("We recommend: ")
                        print('    If you want higher confidence with less precision, use the VP method')
                        print('    If you want more precision with lower confidence, use the Weibull method')
                if DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend: using the Weibull method")
                if DataType.lower() == 'cluster' or DataType.lower() == 'c':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the Weibull method')
                    print('    If you want more precision with lower confidence, use the Binomial #2 method')
                if DataType.lower() == 'outliers' or DataType.lower() == 'o':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the Binomial Smooth method')
                    print('    If you want more precision with lower confidence, use the Binomial #2 method')
                if DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
            elif R2 > 10:
                if R2 < 25:
                    if DataType.lower() == 'normal' or DataType.lower() == 'n':
                        print("We recommend: ")
                        print('    If you want higher confidence with less precision, use the VP method')
                        print('    If you want more precision with lower confidence, use the Weibull method')
                    print('Untested option here, use b2 for cluster.')
                elif R2 >= 25:
                    if DataType.lower() == 'normal' or DataType.lower() == 'n':
                        print("We recommend: using the VP method")
                if DataType.lower() == 'two' or DataType.lower() == 't':
                    print("We recommend: using the Weibull method")
                if DataType.lower() == 'outliers' or DataType.lower() == 'o' or DataType.lower() == 'cluster' or DataType.lower() == 'c':
                    print("We recommend: ")
                    print('    If you want higher confidence with less precision, use the Binomial Smooth method')
                    print('    If you want more precision with lower confidence, use the Binomial #2 method')
                if DataType.lower() == 'other':
                    print('\tWe recommend using the Tolerance Method for the most accuracy, but least precision.')
                    print('\tPlotting all methods may be useful here.')
        else:
            print("Warning:\n    Untested percentiles are being used. Bounds may overestimate or underestimate significantly.")
            if R2 < 25:
                print("We recommend using VP or Weibull method")
                print('\tPlotting all methods may be useful here.')
            elif R2 >= 25:
                print("We recommend using the Binomial Smooth or Binomial #2 method")
                print('\tPlotting all methods may be useful here.')
        
        while True:
            Method = input("What method would you like to use? Options are:\n\t(VP) Vysochanskij–Petunin\n\t(T) Tolerance \n\t(W) Weibull\n\t(B2) Binomial #2\n\t(BS) Binomial Smooth\n\t(R) Rayleigh\n\t(ALL) All methods\n\t(AVG) Average of all methods [Not recommended]\n Desired Method: ")
            if (Method.lower() == 'tolerance' or Method.lower() == 't' or Method.lower() == 'vp' or Method.lower() == 'binomial 2'
                or Method.lower() == 'b2' or Method.lower() == 'binomial smooth' or Method.lower() == 'bs'
                or Method.lower() == 'all' or Method.lower() == 'weibull' or Method.lower() == 'w' or Method.lower() == 'avg'
                or Method.lower() == 'r' or Method.lower() == 'rayleigh'):
                break
            else:
                print("Invalid method inputted.\n")
                continue
        upperBound = Plot(data1, data2, DataType = DataType.lower(), Method = Method.lower(), perc = R1, CLevel = CLevel)
        raderr = np.sqrt(data1**2 + data2**2)
        if Method.lower() != 'all':
            for i in range(length(upperBound)):
                try:
                    if upperBound[i] < 0:
                        upperBound[i] = abs(upperBound[i])
                except:
                    if upperBound < 0:
                        upperBound = abs(upperBound)
                    if R1 < .01:
                        upperBound = np.min(raderr)
            try:
                upperBound = sorted(upperBound)
            except:
                upperBound = upperBound
                upperBound = [0, upperBound]
            try:
                print(f'\nThe {int(R1*100)}th percentile {int(CLevel)}% confidence interval for the radial error relative to the origin using the {Method} method is: {np.round(upperBound,4)}')
            except:
                print(f'\nThe {R1*100}th percentile {CLevel} confidence interval for the radial error relative to the origin using the {Method} method is: {np.round(upperBound,4)}')
        else:
            upperBound= list(upperBound)
            for j in range(length(upperBound)):
                for i in range(length(upperBound[j])):
                    try:
                        if upperBound[j][i] < 0:
                            upperBound[j][i] = abs(upperBound[j][i])
                    except:
                        if upperBound[j] < 0:
                            upperBound[j] = abs(upperBound[j])   
                        if R1 < .01:
                            upperBound[j] = np.min(raderr)
            upperBound = [sorted(u) if u is list else u for u in upperBound]
            
            try:
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the Tolerance Method is: {np.round(upperBound[0],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% upper bound for the radial error relative to the origin using the Binomial Smooth is: {np.round(upperBound[1],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the Binomial #2 Method is: {np.round(upperBound[3],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the Weibull Method is: {np.round(upperBound[5],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the VP Method is: {np.round(upperBound[2],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the average of all the methods is: {np.round(upperBound[4],4)}')
            except:
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the Tolerance Method is: {np.round(upperBound[0],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% upper bound for the radial error relative to the origin using the Binomial Smooth is: {np.round(upperBound[1],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the Binomial #2 Method is: {np.round(upperBound[3],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the Weibull Method is: {np.round(upperBound[5],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the VP Method is: {np.round(upperBound[2],4)}')
                print(f'\nThe {(R1*100)}th percentile {(CLevel)}% confidence interval for the radial error relative to the origin using the average of all the methods is: {np.round(upperBound[4],4)}')