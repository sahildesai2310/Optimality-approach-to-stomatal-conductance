
#This file will solve the Delwiche model with the hydraulics of SPAC model

#Delwiche model only deals with the guard cell, subsidiary cell and mesophyll cell
#turgor

# Pennman Monteith is not used in this model.
# This uses a simple diffusion equation to define the flux

#SPAC adds the stem water potential and provides a continumm approach from the
#root level to the cellular level

#@author: Sahil Desai (sad269@cornell.edu)
"""
# Choice of solver
using JuMP
using Juniper
using Ipopt
using Cbc

optimizer = Juniper.Optimizer
params = Dict{Symbol,Any}()
params[:nl_solver] = with_optimizer(Ipopt.Optimizer, print_level=0)
params[:mip_solver] = with_optimizer(Cbc.Optimizer, logLevel=0)

m = Model(with_optimizer(optimizer, params))
"""
#"""
using JuMP
#using Juniper
using Ipopt
#using Cbc

optimizer = Ipopt.Optimizer
m = Model(with_optimizer(optimizer, ))
#"""
# Definition of the parameters from Delwiche model
A2=6.3*0.0001;
A3=6.3*0.001;
A4=3.1*0.01;
A5=2.8*0.001;

Vg0=4.2*0.000001;
Vs0=4.2*0.00001;
Vm0=1*0.0001;

pim0=17*(10^5);
pig0=20*(10^5);
pis0=20*(10^5);

epsilonm=50*(10^5);
epsilong=23*(10^5);
epsilons=10*(10^5);

L=1*0.00000000001;
l=100;
qsi=4.0;
n=70;
D=26;
delta=0.5;
d=2*0.01;
r=0.5*0.01;
b=1*0.01;

b0=-4*0.001;
#bg=1.6*0.00000001;
#bs=-2.0*0.00000001;


psir=-3*10^5;
v=1*(10^3);

cm=23*0.000001;
cta=4.6*0.000001;

Rr=1.5*(10^9);
Rt= 0.4*(10^10);
Rp=(Rr+Rt);
Ct = 0.625*10^-8;

expr2=0;


# Definition of the variables
@variable(m, dPgdt)
@variable(m, dPsdt)
@variable(m, dPmdt)
@variable(m, dpsitdt)
@variable(m, J2)
@variable(m, J3)
@variable(m, J4)
@variable(m, E)
@variable(m, psig)
@variable(m, psis)
@variable(m, psim)
@variable(m, psit)
@variable(m, psicw)
@variable(m, Pg)
@variable(m, Ps)
@variable(m, Pm)
@variable(m, g)
@variable(m, Ast)
@variable(m, a)
@variable(m,expr1)
@variable(m,Vg)
@variable(m,Vs)
@variable(m,Vm)
@variable(m, bg)
@variable(m, bs)
@variable(m, Jx)

# Constraint Definition
@NLconstraint(m, dPgdt == (epsilong/Vg0)*(J2))
@NLconstraint(m, dPgdt == 0)
@NLconstraint(m, dPsdt == (epsilons/Vs0)*(-J2+J3))
@NLconstraint(m, dPsdt == 0)
@NLconstraint(m, dPmdt == (epsilonm/Vm0)*(J4))
@NLconstraint(m, dPmdt == 0)
@NLconstraint(m, dpsitdt == (psir-psit)/(Rr*Ct)-(psit-psicw)/(Rt*Ct))
@NLconstraint(m, dpsitdt == 0)
@NLconstraint(m, J2 == 0.5*L*A2*(psis-psig))
@NLconstraint(m, J3 == L*A3*(psicw-psis))
@NLconstraint(m, J4 == L*A4*(psis-psim))
@NLconstraint(m, psig == Pg - pig0*(1-Pg/epsilong))
@NLconstraint(m, psis == Ps - pis0*(1-Ps/epsilons))
@NLconstraint(m, psim == Pm - pim0*(1-Pm/epsilonm))
@NLconstraint(m, E == g*(cm-cta))
@NLconstraint(m, g == 1/((1/D)*(delta+(d+r)/(n*Ast)+qsi*(l/v)^0.5)))
@NLconstraint(m, expr1 == b0+bg*Pg+bs*Ps)

f(expr1, expr2) = max(expr1, expr2)
JuMP.register(m, :f, 2, f, autodiff=true)

@NLconstraint(m,a ==f(expr1, expr2))
@NLconstraint(m, Ast ==3.14*b*a)
@NLconstraint(m, Vg == Vg0*(1+Pg/epsilong))
@NLconstraint(m, Vs == Vs0*(1+Ps/epsilons))
@NLconstraint(m, Vm == Vm0*(1+Pm/epsilonm))
@NLconstraint(m, Vg <= 2*Vg0)
@NLconstraint(m, Vs <= 2*Vs0)
@NLconstraint(m, Vm <= 2*Vm0)
@NLconstraint(m, Jx == E + J3/A5 + J4/A5)
@NLconstraint(m, psicw == psir - Jx*Rp)

@NLconstraint(m, Pg >= 0)
@NLconstraint(m, Ps >= 0)
@NLconstraint(m, Pm >= 0)
@NLconstraint(m, Vg >= 0)
@NLconstraint(m, Vs >= 0)
@NLconstraint(m, Vm >= 0)
@NLconstraint(m, psit <= psir)
@NLconstraint(m, psicw <= psit)

@NLconstraint(m, a >= 0)
@NLconstraint(m, a <= b)
@NLconstraint(m, Ast >= 0)
@NLconstraint(m, g >= 0)
@NLconstraint(m, E >= 0)
@NLconstraint(m, Jx >= 0)
@NLconstraint(m, J2 >= 0)
@NLconstraint(m, J3 >= 0)
@NLconstraint(m, bg >= 0)
@NLconstraint(m, bg <= 10^-7)
@NLconstraint(m, bs >= -10^-7)
@NLconstraint(m, bs <= 0)

@NLobjective(m, Min, E)
optimize!(m)
# Display Results
println("E (mm/s)= ", JuMP.value(E))
println("")
println("Pg (bars) = ", JuMP.value(Pg/100000))
println("Pm (bars) = ", JuMP.value(Pm/100000))
println("Ps (bars) = ", JuMP.value(Ps/100000))
println("psit (bars) = ", JuMP.value(psit/100000))
println("psicw (bars) = ", JuMP.value(psicw/100000))
println("psir (bars) = ", (psir/100000))
println("")
println("a (mm) = ", JuMP.value(a))
println("Ast (mm^2) = ", JuMP.value(Ast))
println("g (mm/s) = ", JuMP.value(g))
println("")
println("bg = ", JuMP.value(bg))
println("bs = ", JuMP.value(bs))
println("")
#println("expr1 = ", JuMP.value(expr1))
println("Jx = ", JuMP.value(Jx))
println("J2 = ", JuMP.value(J2))
println("J3 = ", JuMP.value(J3))
