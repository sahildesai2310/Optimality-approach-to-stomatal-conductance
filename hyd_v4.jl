using Ipopt, JuMP
m = Model(with_optimizer(Ipopt.Optimizer))
# Definition of the parameters from Delwiche model
A2=6.3*0.0001;
A3=6.3*0.001;
A4=3.1*0.01;
A5=2.8*0.001;
Vg0=4.2*0.000001;
Vs0=4.2*0.00001;
Vm0=1*0.0001;
pim0=17*(10^5);
epsilonm=50*(10^5);
L=1*0.00000000001;
l=100;
qsi=4.0;
n=70;
D=26;
delta=0.5;
d=2*0.01;
r=0.5*0.01;
b=1*0.01;
cm=23*0.000001;
pim=0;
v=1*(10^3);

epsilong=23*(10^5);
epsilons=10*(10^5);
pig0=20*(10^5);
pis0=20*(10^5);
bg=1.6*0.00000001;
bs=-2.0*0.00000001;
b0=-4*0.001;

psir=-3*10^5;
cta=4.6*0.000001;
Rp=0.4*(10^10);
pig=5*(10^5 );
Rr=1.5*(10^9);

k=1/(1+Rp*L/A5*(A3+A4));

a1=-epsilong*L*A2/(2*Vg0)*(1+pig0/epsilong);

a2=epsilong*L*A2/(2*Vg0)*(1+pis0/epsilons);

a3=epsilong*L*A2/(2*Vg0)*pig;

a4=epsilong*L*A2/(2*Vg0)*(pig0-pis0);

a5=epsilons*L*A2/(2*Vs0)*(1+pig0/epsilong);

a6=-epsilons*L*A2/(2*Vs0)*(1+pis0/epsilons)+epsilons*L^2*A3^2*k*Rp/(Vs0*A5)*(1+pis0/epsilons)-epsilons*L*A3/Vs0*(1+pis0/epsilons);

a7=epsilons*L^2*A3*A4*k*Rp/(Vs0*A5)*(1+pim0/epsilonm);

a8=-epsilons*L*A2/(2*Vs0)*pig-epsilons*L^2*A3*A4*k*Rp/(Vs0*A5)*pim;

a9=epsilons*L*A2/(2*Vs0)*(pis0-pig0)+epsilons*L*A3/Vs0*(k*psir-k*Rp*L*A3/A5*pis0-k*Rp*L*A4/A5*pim0+pis0);

a10=epsilonm*L^2*A3*A4*k*Rp/(Vm0*A5)*(1+pis0/epsilons);

a11=epsilonm*L^2*A4^2*k*Rp/(Vm0*A5)*(1+pim0/epsilonm)-epsilonm*L*A4/(Vm0)*(1+pim0/epsilonm);

a12=(-epsilonm*L^2*A4^2*k*Rp/(Vm0*A5)+epsilonm*L*A4/Vm0)*pim;

a13=epsilonm*L*A4/Vm0*(k*psir-k*Rp*L*A3/A5*pis0-k*Rp*L*A4/A5*pim0+pim0);

a14=(1/Rp)*(1+pim0/epsilonm);

a15=-1/Rr - 1/Rp;

a16=pim0/Rp-psir/Rr;

expr2=0;

T=25;
P_atm = 101.325*10^3; #Pa
R = 8.314; # Pa.m^3/mol.degC
conc = P_atm/(R*(T+273.15));

# Definition of the variables
@variable(m, a>=0)
@variable(m, amax>=0)
@variable(m, g>=0) # in mm/s from Delwiche model
@variable(m, gs>=0) # in mol/m^2.s by multiplying by concentration (P/RT)
@variable(m, E>=0) # mol/m^2.s % Transpirationl flux
@variable(m, Ast >= 0)
@variable(m, f1)
@variable(m, f2)
@variable(m, psit <= 0)
@variable(m, Pg >=0 )
@variable(m, Ps >=0)
@variable(m, Pm >=0)
@variable(m,expr1)

#@variable(m,bg)
#@variable(m,bs)
#@variable(m,b0)

@NLobjective(m, Min, 0)

@NLconstraint(m, E==(cm-cta)*gs)
@NLconstraint(m, gs == g*conc*10^-3 ) # g is in mm/s, 10^-3 m/mm and c = mol/m^3
@NLconstraint(m, g== 1/((1/D)*(delta+(d+r)/(n*Ast)+qsi*(l/v)^0.5)))

@NLconstraint(m, expr1== b0+bg*Pg+bs*Ps)

f(expr1, expr2) = max(expr1, expr2)
JuMP.register(m, :f, 2, f, autodiff=true)

@NLconstraint(m,a==f(expr1, expr2))
@NLconstraint(m, Ast==3.14*b*a)
@NLconstraint(m, psit <= psir)
@NLconstraint(m, f1==-g*epsilons*L*A3*k*Rp/(Vs0)*(cm-cta))
@NLconstraint(m, f2 == -g*epsilonm*L*A4*k*Rp/(Vm0)*(cm-cta))
@NLconstraint(m, a1*Pg+a2*Ps+a3+a4 == 0)
@NLconstraint(m, a5*Pg + a6*Ps +a7*Pm + a8 + a9 +f1 == 0)
@NLconstraint(m, a10*Ps+a11*Pm+a12+a13+f2 == 0)
@NLconstraint(m, a14*Pm+a15*psit == a16)

JuMP.optimize!(m)
# Display Results
println("E (mol/m^2.s)= ", JuMP.value(E))
println("Pg (bars)= ", JuMP.value(Pg/100000))
println("Pm (bars)= ", JuMP.value(Pm/100000))
println("Ps (bars)= ", JuMP.value(Ps/100000))
println("psit (bars)= ", JuMP.value(psit/100000))
println("Ast (mm^2) = ", JuMP.value(Ast))
println("gs (mol/m^2.s) = ", JuMP.value(g))


#println("b0 = ", JuMP.value(b0))
#println("bg = ", JuMP.value(bg))
#println("bs = ", JuMP.value(bs))
