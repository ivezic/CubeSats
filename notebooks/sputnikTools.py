import numpy as np 
import matplotlib.pyplot as plt


### numerical model specifications and physical constants 
def getPhysConstants():
    physConst = {}
    physConst['RE'] = 6371.0             # Earth's radius in km
    physConst['sigSB'] = 5.6703*(10**-8) # Stefan-Boltzmann constant
    physConst['KelToCel'] = 273.15       # 0 C in Kelvin
    return physConst   


def getInputParamsDefault():
    params = {}
    physConst = getPhysConstants()
    
    ## environment (radiation from Sun and Earth)
    params['Fsun'] = 1372   # solar flux, W/m2 
    params['FIR'] = 240     # Earth's IR flux, W/m2 
    params['rhoE'] = 0.3    # Earth's reflectivity
     
    ## orbit
    params['h'] = 550.0     # in km, orbit's altitude
    params['PorbMin'] = 90  # orbital period in minutes
    params['etaP'] = 0.33   # fraction of orbit in eclipse 
    params['falb'] = 0.62   # correction for albedo variation
                            # for polar orbit facing Sun: 0.06
    # view factor
    RE = physConst['RE']
    params['fE'] = (RE/(RE+params['h']))**2
    
    ## satellite 
    # area
    params['Atot'] = 0.10   # in m2, as 2U CubeSat
    params['etaS'] = 0.25   # effective area ratio, sphere, point source
    params['etaE'] = 0.50   # effective area ratio, sphere, hemisphere 
    # use instead the version with a correction for incomplete hemisphere
    # it's basically viewing factor for two spheres multiplied by fE
    params['etaE'] = 0.50 * (1-np.sqrt(1-params['fE'])) / params['fE']  
    
    # surface emissivity
    params['alphaS'] = 0.86 # black anodized Al
    params['epsT'] = 0.86   # black anodized Al
    params['alphaIR'] = params['epsT']  # usual ansatz
    # thermal inertia
    params['mass'] = 2.0    # kg (n.b. implies density)
    params['C'] = 921.0     # J/kg/K, aluminum
    # battery charging/power dissipation
    params['etaCell'] = 0.2 # fraction of energy for charging
    
    return params


def getInputParamsHot(params=""):
    
    if (params==""):
        params = getInputParamsDefault()

    ## modify for the hot case 
    # environment (radiation from Sun and Earth)
    params['Fsun'] = 1422   # solar flux, W/m2 
    params['FIR'] = 260     # Earth's IR flux, W/m2 
    params['rhoE'] = 0.35   # Earth's reflectivity
    ## orbit
    params['etaP'] = 0.0    # no eclipse in beta=90 (terminator) orbit  
    params['falb'] = 0.06   # correction for albedo in hot orbit 
    
    return params


def getInputParamsCold(params=""):

    if (params==""):
        params = getInputParamsDefault()

    ## modify for the cold case 
    # environment (radiation from Sun and Earth)
    params['Fsun'] = 1322   # solar flux, W/m2 
    params['FIR'] = 220     # Earth's IR flux, W/m2 
    params['rhoE'] = 0.25   # Earth's reflectivity     
 
    return params
 


### heat flux and equilibrium temperature evaluations 

def getQsun(p):
    return p['etaS']*p['Atot']*p['alphaS']*p['Fsun']

def getQref(p):
    return p['fE']*p['etaE']*p['Atot']*p['falb']*p['rhoE']*p['alphaS']*p['Fsun']

def getQIR(p):
    return p['fE']*p['etaE']*p['Atot']* p['alphaIR']*p['FIR'] 

def getQdissip(p):
    return p['etaCell']*(1-p['etaP'])*(getQsun(p)+getQref(p))
  
def getQin(p):
    QinSun = (1-p['etaCell']*p['etaP'])*(getQsun(p)+getQref(p)) + getQIR(p)
    QinEclipse = getQIR(p) + getQdissip(p)
    return QinSun, QinEclipse

def getAllHeatQ(p):
    return getQsun(p), getQref(p), getQIR(p), getQdissip(p)

## equilibrium temperatures
def getTeq(Qin,p):
    physConst = getPhysConstants()
    return (Qin/(p['Atot']*p['epsT']*physConst['sigSB']))**0.25


# return the view factor divided by fE=(RE/(RE+altitude))^2  
# here fE = 1/h^2 
def getF12h2(h,beta):
    rad2deg = 180/np.pi
    brad = beta/rad2deg
    if (brad<=np.arccos(1/h)):
        return np.cos(brad)
    x = np.sqrt(h**2-1)
    y = -x / np.tan(brad)
    if (y>1):
        return 0
    z = np.sqrt(1-y**2)
    t1 = (np.cos(brad)*np.arccos(y)-x*np.sin(brad)*z) 
    t2 = np.arctan(np.sin(brad)*z/x) 
    # print(x,y,z,t1,t2)
    return (t1+t2*h**2)/np.pi

def getEffectiveAreas2U(h, beta):
    if ((beta<0)or(beta>180)):
        return 0 
    if (beta>90):
        beta = 180-beta
    # small side towards Earth
    s1 = getF12h2(h, beta)
    # the "other" small side 
    s2 = getF12h2(h, 180-beta)
    # long 2U side "towards" Earth
    s3 = 2 * getF12h2(h, beta+90)
    # long 2U side "away from" Earth
    s4 = 2 * getF12h2(h, 180 - (beta+90))
    # 2 long 2U sides "perpendicular to" Earth
    s5 = 2 * 2 * getF12h2(h, 90)
    # print(h,beta,s1, s2, s3, s4, s5)
    return (s1+s2+s3+s4+s5)/10 

def getEffectiveAreas1U(h, beta):
    if ((beta<0)or(beta>180)):
        return 0 
    if (beta>90):
        beta = 180-beta
    # small side towards Earth
    s1 = getF12h2(h, beta)
    # the "other" small side 
    s2 = getF12h2(h, 180-beta)
    # small side "towards" Earth
    s3 = getF12h2(h, beta+90)
    # small side "away from" Earth
    s4 = getF12h2(h, 180 - (beta+90))
    # 2 small sides "perpendicular to" Earth
    s5 = 2 * getF12h2(h, 90)
    # print(h,beta,s1, s2, s3, s4, s5)
    return (s1+s2+s3+s4+s5)/6



## for compatibility with old code 
        
def doOneCase(modelTitleText,alpha,epsilon, params="", verbose=False):
    physConst = getPhysConstants()
    titlePrint(modelTitleText)
    if (params == ""):
        print('using default parameters from getInputParamsDefault()')
        params = getInputParamsDefault()
    params['alphaS'] = alpha
    params['epsT'] = epsilon
    params['alphaIR'] = params['epsT'] 
    
    # heat fluxes
    Q_sun_flux, Q_albedo_flux, Q_IR_flux, Q_int = getAllHeatQ(params)
    if (verbose):
        print('Q (sun, alb, IR, diss):', Q_sun_flux, Q_albedo_flux, Q_IR_flux, Q_int)
    
    # equilibrium solutions 
    Q_hot, Q_cold = getQin(params)
    print('Qsun=', Q_hot, ' Qeclipse=', Q_cold) 
    Temp_hot = getTeq(Q_hot,params)
    Temp_cold = getTeq(Q_cold,params)
    if (verbose):
        printEqTemp('Hot', Temp_hot, physConst['KelToCel'])
        printEqTemp('Cold', Temp_cold, physConst['KelToCel'])
    
    # solve for time variation
    t0H = getTimeConstant(params,Temp_hot)
    t0C = getTimeConstant(params,Temp_cold)
    xH = (1-params['etaP'])*params['PorbMin']*60/t0H
    xC = params['etaP']*params['PorbMin']*60/t0C
    Tmin, Tmax, tau0C, tauFC, tau0H, tauFH = solveBistableEquilibrium(xC, xH, Temp_cold, Temp_hot, 1000)

    # generate temperature arrays
    timeA, TempsA = getFullTempArray(params, Tmin, Tmax, Temp_cold, Temp_hot, t0C, t0H)
    if (verbose):
        print('doOneCase: temp. range =', np.min(TempsA), np.max(TempsA))
        print('     in Celsius: range =', np.min(TempsA-273.15), np.max(TempsA-273.15))

    return timeA, TempsA 

def getTimeConstant(params,Teq):
    physConst = getPhysConstants()
    return (params['mass']*params['C']/physConst['sigSB']/params['Atot']/params['epsT']/Teq**3)

def getFullTempArray(params, Tmin, Tmax, Temp_cold, Temp_hot, t0C, t0H):
    
    # generate temperature arrays
    if (Tmin >= Temp_hot):
        print('adjusting Tmin=', Tmin, 'to: Thot=', Temp_hot)
        Tmin = 0.998*Temp_hot
    if (Tmin <= Temp_cold):
        print('adjusting Tmin=', Tmin, 'to: Tcold=', Temp_cold)
        Tmin = 1.001*Temp_cold
    if (Tmax >= Temp_hot):
        print('adjusting Tmax=', Tmax, 'to: Thot=', Temp_hot)
        Tmax = 0.999*Temp_hot 
    TempsC = np.linspace(Tmax, Tmin, 100)
    timeC = getTempSolution(TempsC, t0C, Temp_cold)
    TempsH = np.linspace(Tmin, Tmax, 200)
    timeH = getTempSolution(TempsH, t0H, Temp_hot)
    timeH[-1] = (1-params['etaP'])*params['PorbMin']*60

    # concatenate both solutions
    timeOffset = params['etaP']*params['PorbMin']*60 
    timeHot = (1-params['etaP'])*params['PorbMin']*60 
    timeHshifted = timeH/np.max(timeH)*timeHot + timeOffset
    timeA = np.concatenate((timeC/np.max(timeC)*timeOffset, timeHshifted), axis=None)
    TempsA = np.concatenate((TempsC, TempsH), axis=None)
    
    return timeA, TempsA


def getNumSolution(params, Tstart, Tc=0.0, Pc=0.0):
    
    orbPeriodMin = params['PorbMin']
    etaP = params['etaP']
    t_final_min = orbPeriodMin*etaP   # eclipse duration in minutes
    t_step = 1.0                  # integration step in seconds 

    # Tstart = TempsBlack[0]
    Qs, Qe = getQin(params)
    epsT = params['epsT'] 
    Atot = params['Atot']  
    C = params['C'] 
    mass = params['mass'] 
    
    physConst = getPhysConstants()
    sigmaSB = physConst['sigSB']

    if (etaP>0):
        facSun = (1-etaP)/etaP
    else: 
        facSun = 1.0
        
    # need to enforce cyclic boundary condition with iterations
    # the duration of "sunshine", in units of eclipse time (=t_final_min)
    for k in range(0,25):
        time1, Temps1 = nonSteadyStateTemp(Tstart,t_final_min,t_step,Qe,C,mass,sigmaSB,epsT,Atot,Tc,Pc)
        tmax = facSun*t_final_min
        time2, Temps2 = nonSteadyStateTemp(Temps1[-1],tmax,t_step,Qs,C,mass,sigmaSB,epsT,Atot,Tc,Pc)
        Tstart = 0.5*(Tstart+Temps2[-1])
       
    time = np.concatenate((time1, (time2+time1[-1])), axis=None)
    Temperature = np.concatenate((Temps1, Temps2), axis=None)

    return time, Temperature 


# given x and a tau0 array, find tau for each tau0
def getTauSolution(x, tau0, Nsteps=100):
    tauFinal = 0*tau0
    for i in range(0,np.size(tau0)):
        thisTau0 = tau0[i]
        tauGrid = np.linspace(thisTau0, 1, Nsteps)
        notdone = True
        for j in range(1,np.size(tauGrid)):
            if (notdone):
                thisTau = tauGrid[j]
                thisX = getXforTempSolution(thisTau, thisTau0)
                if (thisX >= x): 
                    tauFinal[i] = thisTau
                    notdone = False 
    return tauFinal 
     
# given tau and tau0, return x=t/t0
def getXforTempSolution(tau, tau0):
    x1 = 0.5*(np.arctan(tau)-np.arctan(tau0))
    x2 = 0.25*(np.log((tau+1)/(tau0+1)) - np.log((tau-1)/(tau0-1)))
    return x1+x2


# given an array of temperatures, Teq and t0, return model time grid
def getTempSolution(T, t0, Teq):
    tau = T/Teq
    # print('getTempSolution: Teq, t0, tau0, tauFinal=', Teq, t0, tau[0], tau[-1])
    t1 = 0.5*(np.arctan(tau)-np.arctan(tau[0]))
    t2 = 0.25*(np.log((tau+1)/(tau[0]+1)) - np.log((tau-1)/(tau[0]-1)))
    return t0*(t1+t2)


def solveBistableEquilibrium(xC, xH, TeqCold, TeqHot, Nsteps=100):
    C1 = TeqCold / TeqHot
    tau0Carr = np.linspace(1.01,3.0, Nsteps)
    tauFinalCarr = getTauSolution(xC, tau0Carr)
    tau0Harr = C1*tauFinalCarr
    tauFinalHarr = getTauSolution(xH, tau0Harr)
    tau0C2 = tauFinalHarr/C1
    ## now solve for tauFinalCarr(tau0Carr) = tauFinalCarr(tau0C2) 
    ## to get tau0C and tauFinalC, and then tau0H, tauFinalH, and
    ## finally TminEq and TmaxEq
    # by construction, tau0Carr and tau0C2 are increasing 
    # and at first point tau0C2[0] > tau0Carr[0]
    notdone = True
    for i in range(1,np.size(tau0C2)):
        if (notdone):
            if (tau0Carr[i] >= tau0C2[i]): 
                tau0C = tau0Carr[i]
                tauFinalC = tauFinalCarr[i]
                notdone = False 
    tau0H = C1 * tauFinalC
    tauFinalH = C1 * tau0C
    TminEq = tauFinalC * TeqCold
    TmaxEq = tau0C * TeqCold
    return TminEq, TmaxEq, tau0C, tauFinalC, tau0H, tauFinalH


     
############# old code from Haley Stewart (University of Washington)   #################################
def nonSteadyStateTemp(Tinitial,t_final_min,t_step,Qheating,c,mass,sigma,emissivity,Area_sphere,Tc,Pc):
    T = Tinitial # set the initial temp to the hot eq temp
    Temps = []
    Temps.append(T) # include initial temp in list
    time = []
    time.append(0) # python counts from zero
    # need smaller time step
    t_final = t_final_min*60 # convert minutes to seconds
    for i in range(1, int((t_final/t_step) + 1), 1): # sampling every t_step seconds 
        T = nextTemp(T, t_step, Qheating, c, mass, sigma, emissivity, Area_sphere,Tc,Pc)
        Temps.append(T)
        time.append(i*t_step)
    return np.array(time), np.array(Temps)


def nextTemp(T, t_step, Qheating, c, mass, sigma, emissivity, Area_sphere,Tc,Pc):
    if ((T<Tc)):
        QheatingTotal = Qheating + Pc
    else:
        QheatingTotal = Qheating
    dTdt = (1/(c*mass))*(QheatingTotal-(sigma*(T**4)*emissivity*Area_sphere))
    return T + dTdt*t_step
#########################################################################################################




### PLOTTING 
def TempsPlot2(time1, temp1, c1, time2, temp2, c2, time3, temp3, c3, outfile="", title=""):
    physConst = getPhysConstants()
    KelToCel = physConst['KelToCel']
    
    # Plot the Temperature wrt time
    battTmin =  0.0
    battTmax = 40.0
    tempmax = []
    tempmin = []
    for int in range(0,len(time1)):
        tempmax.append(battTmax)
        tempmin.append(battTmin)
    timeMin = np.array(time1)/60
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(wspace=0.22, left=0.14, right=0.98, bottom=0.15, top=0.97)

    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.plot(np.array(time1)/60,np.array(temp1)-KelToCel,label=c1, lw=3)
    plt.plot(np.array(time2)/60,np.array(temp2)-KelToCel,label=c2, lw=3)
    if (c3 != ""):
        plt.plot(np.array(time3)/60,np.array(temp3)-KelToCel,label=c3, lw=3)
    plt.plot(np.array(time1)/60,tempmax, lw=1, c='black')
    plt.plot(np.array(time1)/60,tempmin, lw=1, c='black')
    ax.fill_between(timeMin, tempmin, tempmax, alpha=0.1) 
    plt.rc('legend',fontsize=12)  
    plt.legend(loc=1)
    plt.title(title)
    plt.xlabel('Time (minute)', fontsize=18)
    plt.ylabel('Temperature (C)', fontsize=18)
    yLimB, yLimT = plt.ylim()
    plt.ylim(yLimB-5, 60.0)

    if (outfile == ""):
        name = 'figures/TempsVsOperatingTemp.png'
    else:
        name = 'figures/TempsPlot_' + outfile + '.png'
    plt.savefig(name)
    plt.close("all")
    return 

def TempsPlotCompare(timeA, TempsA, timeN, TempsN, labelText, label2, outfile="", title=""):
    physConst = getPhysConstants()
    KelToCel = physConst['KelToCel']
        
    # Plot the Temperature wrt time
    battTmin =  0.0
    battTmax = 40.0
    tempmax = []
    tempmin = []
    for int in range(0,len(timeA)):
        tempmax.append(battTmax)
        tempmin.append(battTmin)
    timeMin = np.array(timeA)/60
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(wspace=0.22, left=0.16, right=0.98, bottom=0.16, top=0.97)

    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.plot(np.array(timeA)/60,np.array(TempsA)-KelToCel,label=labelText, lw=3, c='b')
    plt.plot(np.array(timeN)/60,np.array(TempsN)-KelToCel,label=label2, lw=2, c='r', linestyle='dashed')
    plt.plot(np.array(timeA)/60,tempmax, lw=1, c='black')
    plt.plot(np.array(timeA)/60,tempmin, lw=1, c='black')
    ax.fill_between(timeMin, tempmin, tempmax, alpha=0.1) 

    plt.rc('legend',fontsize=15)  
    plt.legend(loc=1)
    plt.title(title)
    plt.xlabel('Time (minute)', fontsize=22)
    plt.ylabel('Temperature (C)', fontsize=22)
    yLimB, yLimT = plt.ylim()
    plt.ylim(yLimB-5, 60.0)
     

    if (outfile == ""):
        name = 'figures/TempsVsOperatingTemp.png'
    else:
        name = 'figures/TempsPlotCompare_' + outfile + '.png'
    
    plt.savefig(name)
    plt.close("all")
    return 


def TempsPlot(time, temp1, coating1, temp2, coating2, temp3, coating3, outfile="", title=""):    
    TempsPlot2(time, temp1, coating1, time, temp2, coating2, time, temp3, coating3, outfile, title)
    return 

def titlePrint(coating):
    print()
    print(coating + ' Sphere, Temperature Extremes')
    print('-------------------------------------------------------')

def printEqTemp(case, Temp, KelToCel):
    print(case +' Eq. Temp: ' + str('%.2f'%(Temp)) + 'K = ' + str('%.2f'%(Temp-KelToCel)) + u'\u2103')






