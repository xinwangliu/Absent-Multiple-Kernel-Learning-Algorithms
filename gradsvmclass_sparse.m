function [grad] = gradsvmclass_sparse(K,indsup,Alpsup)

d=size(K,3);
grad= zeros(d,1);
% pos = find(gamma>threshold);
% npos = length(pos);
for p = 1:d
    grad(p) = -0.5*Alpsup'*K(indsup,indsup,p)*Alpsup;
%     grad(pos(p)) = - 0.5*Alpsup'*K(indsup,indsup,pos(p))*Alpsup  ;
end


