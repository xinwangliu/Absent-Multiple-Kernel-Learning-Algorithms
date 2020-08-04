function [w,d,pos,obj]= mySVMclass(y,c,K)

if min(y) ~= -1
    error(' y must coded: 1 for class one and -1 for class two')
end
%----------------------------------------------------------------------
%      monqp(H,b,c) solves the quadratic programming problem:
% 
%    min 0.5*x'Hx - d'x   subject to:  A'x = b  and  0 <= x <= c 
%     x    
%----------------------------------------------------------------------
YY = sparse(y*y');
H = K.*YY;
e = ones(size(y));
A = y;
b = 0;                                                      
[alpha,lambda0, pos] = mymonqp(H,e,A,b,c);            
obj= -0.5*alpha'*H(pos,pos)*alpha + sum(alpha);
ysup = y(pos);
w = (alpha.*ysup);
d = lambda0;