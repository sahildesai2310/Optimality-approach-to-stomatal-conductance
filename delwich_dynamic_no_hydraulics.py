"""
Solving the Delwiche Model in steady state without any hydraulics

@author: sahildesai (sad269@cornell.edu)
"""
import numpy as np
from scipy import integrate as integ


params={}
# Fixed parameter Values
params['A2']=6.3e-4
params['A3']=6.3e-3
params['A4']=3.1e-2
params['A5']=2.8e-3
params['Vg0']=4.2e-6
params['Vs0']=4.2e-5
params['Vm0']=1e-4
params['pim0']=17e5
params['epsilonm']=50e5
params['L']=1e-11
params['l']=100
params['qsi']=4.0
params['n']=70
params['D']=26
params['delta']=0.5
params['d']=2e-2
params['r']=0.5e-2
params['b']=1e-2
params['cm']=23e-6
params['pim']=0
params['v']=1e3
# Hypothetical plant types
#"""
#Set 1
params['epsilong']=50e5
params['epsilons']=50e5
params['pig0']=20e5
params['pis0']=15e5
params['bg']=1.4e-8
params['bs']=-2.1e-8
params['b0']=0e-3
#"""
"""
#Set 2
params['epsilong']=50e5
params['epsilons']=50e5
params['pig0']=20e5
params['pis0']=15e5
params['bg']=1.4e-8
params['bs']=-2.1e-8
params['b0']=0e-3
"""
"""
#Set 3
params['epsilong']=50e5
params['epsilons']=50e5
params['pig0']=20e5
params['pis0']=15e5
params['bg']=1.4e-8
params['bs']=-2.1e-8
params['b0']=0e-3
"""
"""
#Set 4
params['epsilong']=50e5
params['epsilons']=50e5
params['pig0']=20e5
params['pis0']=15e5
params['bg']=1.4e-8
params['bs']=-2.1e-8
params['b0']=0e-3
"""

# Parameters Varied
params['psir']=-3e5
params['cta']=4.6e-6
params['Rp']=0.4e10
params['pig']=5e5 # Value of pi_g_telda at steady state
params['Ct']=(1/1.6)*1e-8
params['Rr']=1.5e9

def delwiche_dynamic(P_sol, t, params):
    
    Pg=P_sol[0]
    Ps=P_sol[1]
    Pm=P_sol[2]   
    
    
    A2=params['A2']
    A3=params['A3']
    A4=params['A4']
    A5=params['A5']
    Vg0=params['Vg0']
    Vs0=params['Vs0']
    Vm0=params['Vm0']
    pim0=params['pim0']
    epsilonm=params['epsilonm']
    L=params['L']
    l=params['l']
    qsi=params['qsi']
    n=params['n']
    D=params['D']
    delta=params['delta']
    d=params['d']
    r=params['r']
    b=params['b']
    cm=params['cm']
    pim=params['pim']
    v=params['v']
    
    epsilong=params['epsilong']
    epsilons=params['epsilons']
    pig0=params['pig0']
    pis0=params['pis0']
    bg=params['bg']
    bs=params['bs']
    b0=params['b0']
    
    psir=params['psir']
    cta=params['cta']
    Rp=params['Rp']
    pig=params['pig']*(1-np.exp(-7.8*0.01*t/60))
       
    k=1/(1+Rp*L/A5*(A3+A4))
    
    a1=-epsilong*L*A2/(2*Vg0)*(1+pig0/epsilong)
    
    a2=epsilong*L*A2/(2*Vg0)*(1+pis0/epsilons)
    
    a3=epsilong*L*A2/(2*Vg0)*pig
    
    a4=epsilong*L*A2/(2*Vg0)*(pig0-pis0)
    
    a5=epsilons*L*A2/(2*Vs0)*(1+pig0/epsilong)
    
    a6=-epsilons*L*A2/(2*Vs0)*(1+pis0/epsilons)+epsilons*L**2\
    *A3**2*k*Rp/(Vs0*A5)*(1+pis0/epsilons)\
    -epsilons*L*A3/Vs0*(1+pis0/epsilons)
    
    a7=epsilons*L**2*A3*A4*k*Rp/(Vs0*A5)*(1+pim0/epsilonm)
    
    a8=-epsilons*L*A2/(2*Vs0)*pig-epsilons*L**2*A3*A4*k*Rp/(Vs0*A5)*pim
    
    a9=epsilons*L*A2/(2*Vs0)*(pis0-pig0)+\
    epsilons*L*A3/Vs0*(k*psir-k*Rp*L*A3/A5*pis0-k*Rp*L*A4/A5*pim0+pis0)
    
    a10=epsilonm*L**2*A3*A4*k*Rp/(Vm0*A5)*(1+pis0/epsilons)
    
    a11=epsilonm*L**2*A4**2*k*Rp/(Vm0*A5)*(1+pim0/epsilonm)\
    -epsilonm*L*A4/(Vm0)*(1+pim0/epsilonm)
    
    a12=(-epsilonm*L**2*A4**2*k*Rp/(Vm0*A5)+epsilonm*L*A4/Vm0)*pim
    
    a13=epsilonm*L*A4/Vm0*(k*psir-k*Rp*L*A3/A5*pis0-k*Rp*L*A4/A5*pim0+pim0)
    
    a=max(b0+bg*Pg+bs*Ps,0)
    Ast=np.pi*a*b
    
    if Ast==0:
        Rt=np.inf
    else:
        Rt=1/D*(delta+(d+r)/(n*Ast)+qsi*np.sqrt(l/v))
    f1=-epsilons*L*A3*k*Rp/(Vs0*Rt)*(cm-cta)
    f2=-epsilonm*L*A4*k*Rp/(Vm0*Rt)*(cm-cta)
    
    dPgdt=a1*Pg+a2*Ps+a3+a4
    dPsdt=a5*Pg+a6*Ps+a7*Pm+a8+a9+f1
    dPmdt=a10*Ps+a11*Pm+a12+a13+f2
    return([dPgdt,dPsdt,dPmdt])


P_sol=[00000,00000,00000]

time=np.linspace(0,5400,1000)
Y=integ.odeint(delwiche_dynamic,P_sol,time,(params,))
Pg=Y[:,0]
Ps=Y[:,1]
Pm=Y[:,2]

a=np.maximum(params['b0']+params['bg']*Pg[-1]+params['bs']*Ps[-1],0)
a1 = np.maximum(params['b0']+params['bg']*Pg+params['bs']*Ps,0)
Ast=np.pi*a*params['b']
Ast1 = np.pi*a1*params['b']
print(Ast) # Aperture in mm^2
print('Pg, Ps, Pm: ')
print(Y[-1]/100000) # Pressure is in bar


pig_t = 5e5*(1-np.exp(-7.8*0.01*time/60))
#print(pig_t)

# Calculatig the potentials
psi_g = -params['pig0'] +Pg
psi_s = -params['pis0']+Ps
#print(psi_g)

import plotly.offline as pyof
import plotly.graph_objs as go

trace11 = go.Scatter(x=time/60, y=pig_t/100000, marker={'color': 'black', 'symbol': 104, 'size': 10}, 
                    mode="lines")
data=go.Data([trace11])
figure=go.Figure(data=data)
figure.update_layout(
    xaxis_title="Time (min)",
    yaxis_title="pi_g (bar)",
    )
pyof.plot(figure)


trace21 = go.Scatter(x=time/60, y=Pg/100000, marker={'color': 'blue', 'symbol': 104, 'size': 10}, 
                    mode="lines", name='Guard Cells Water potential (bar)')
trace22 = go.Scatter(x=time/60, y=Ps/100000, marker={'color': 'green', 'symbol': 104, 'size': 10}, 
                    mode="lines", name='Subsidiary Cells Water potential (bar)')
trace23 = go.Scatter(x=time/60, y=Pm/100000, marker={'color': 'black', 'symbol': 104, 'size': 10}, 
                    mode="lines", name='Mesophyll Water potential (bar)')
data=go.Data([trace21,trace22,trace23])
figure=go.Figure(data=data)
figure.update_layout(
    xaxis_title="Time (min)",
    yaxis_title="Pressure (bar)",
    )
pyof.plot(figure)

trace31 = go.Scatter(x=time/60, y=Ast1, marker={'color': 'blue', 'symbol': 104, 'size': 10}, 
                    mode="lines", name='Aperture Area (mm^2)')
data=go.Data([trace31])
figure=go.Figure(data=data)
figure.update_layout(
    xaxis_title="Time (min)",
    yaxis_title="Aperture (mm^2)",
    )
pyof.plot(figure)

"""
trace41 = go.Scatter(x=time/60, y=Ast1, marker={'color': 'blue', 'symbol': 104, 'size': 10}, 
                    mode="lines", name='Aperture Area (mm^2)')
trace42 = go.Scatter(x=time/60, y=Ast1, marker={'color': 'blue', 'symbol': 104, 'size': 10}, 
                    mode="lines", name='Aperture Area (mm^2)')
data=go.Data([trace41,trace42])
figure=go.Figure(data=data)
figure.update_layout(
    xaxis_title="Time (min)",
    yaxis_title="Water Potential",
    )
pyof.plot(figure)


"""
trace41 = go.Scatter(x=time/60, y=psi_g/100000, marker={'color': 'black', 'symbol': 104, 'size': 10}, 
                    mode="lines")
trace42 = go.Scatter(x=time/60, y=psi_s/100000, marker={'color': 'black', 'symbol': 104, 'size': 10}, 
                    mode="lines")
data=go.Data([trace41,trace42])
figure=go.Figure(data=data)
figure.update_layout(
    xaxis_title="Time (min)",
    yaxis_title="Water Potential (bar)",
    )
pyof.plot(figure)

