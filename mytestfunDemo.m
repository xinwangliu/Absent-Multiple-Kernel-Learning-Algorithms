clear;
clc;
path = 'F:\work2015\myTPAMI2018testDemo\';
n = 2000;
x0 = rand(n,1); 
v = rand;
q = randn(n,1);
alpha = rand(n,1);

LB = zeros(n,1);
UB = ones(n,1);
options = optimoptions('fmincon','Algorithm','interior-point','MaxFunctionEvaluations',3000); % run interior-point algorithm
[x,obj]= fmincon(@(x)mytestfun(x,v,q,alpha),x0,[],[],[],[],LB,UB,[],options);
