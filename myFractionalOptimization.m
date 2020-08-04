function [Obj,Grad] = myFractionalOptimization(Sigma,a,S)
%% a: m*1;
%% S: n*m: missing matrix
%% Sigma: n*1;
Obj = sum(a./(S'*Sigma));
Grad = -S*(a./((S'*Sigma).^2));