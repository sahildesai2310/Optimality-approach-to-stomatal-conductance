"""
Solving the Delwiche Model in dynamic state with hydraulics

@author: sahildesai (sad269@cornell.edu)
"""
import numpy as np
from scipy import integrate as integ
from numpy.linalg import inv


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
""
#Set 1
params['epsilong']=50e5
params['epsilons']=50e5
params['pig0']=20e5
params['pis0']=15e5
params['bg']=1.4e-8
params['bs']=-2.1e-8
params['b0']=0e-3

# Parameters Varied
params['psir']=-3e5
params['cta']=4.6e-6
params['Rp']=0.4e10
params['pig']=5e5 # Value of pi_g_telda at steady state
params['Ct']=0.625e-8
params['Rr']=1.5e9


def delwiche_SS_hydraulics(P_sol, params):
    
    Pg=P_sol[0]
    Ps=P_sol[1]
    Pm=P_sol[2]  
    psit = P_sol[3]
    
    
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
    pig=params['pig']
    Rr=params['Rr'] #Added
    Ct=params['Ct'] #Added
    
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
    
    a14=(1/Rp)*(1+pim0/epsilonm)
    
    a15=-1/Rr - 1/Rp
    
    a16=pim0/Rp-psir/Rr
    
    a=max(b0+bg*Pg+bs*Ps,0)
    Ast=np.pi*a*b
    if Ast==0:
        Rt=np.inf
    else:
        Rt=1/D*(delta+(d+r)/(n*Ast)+qsi*np.sqrt(l/v))
    f1=-epsilons*L*A3*k*Rp/(Vs0*Rt)*(cm-cta)
    f2=-epsilonm*L*A4*k*Rp/(Vm0*Rt)*(cm-cta)
    
    A = np.array([[a1,a2,0, 0],[a5,a6,a7,0],[0,a10,a11,0],[0,0,a14,a15]])
    mat_b = np.array([[-a3-a4],[-a8-a9-f1],[-a12-a13-f2],[a16]])  
   
    return(A,mat_b)
    
res_Pg = res_Ps = res_Pm = res_psit = 10**9
tol = 0.00001
P_sol=[1000000,1000000,1000000,1000000]
iter_count=0

while (res_Pg>tol and res_Ps>tol and res_Pm>tol and res_psit>tol):
   
    P_sol_old= P_sol # Saving the previous iteration value as old solution
    iter_count=iter_count+1
    print(iter_count)
    
 
    A,mat_b=delwiche_SS_hydraulics(P_sol, params)
 
    Ainv = inv(A)
    P_sol = np.dot(Ainv,mat_b) # Pressure solution for current iteration

    
    # Residual of pressures
    res_Pg = np.absolute(P_sol[0,0]-P_sol_old[0])
    res_Ps = np.absolute(P_sol[1,0]-P_sol_old[1])
    res_Pm = np.absolute(P_sol[2,0]-P_sol_old[2])
    res_psit = np.absolute(P_sol[3,0]-P_sol_old[3])
  
    Pg=P_sol[0,0]
    Ps=P_sol[1,0]
    Pm=P_sol[2,0] 
    psit=P_sol[3,0] 
    
    P_sol = np.array([Pg,Ps,Pm,psit])
    
    print('Pg, Ps, Pm: ')
    print(P_sol[0:3]/100000) # Pressure is in bar
    print('psi_t: ')
    print(P_sol[3]/100000)
    print(res_Pg,res_Ps,res_Pm,res_psit)
    print('')