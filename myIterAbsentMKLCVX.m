function [gamma,alpha,b,xi,tau,obj] = myIterAbsentMKLCVX(KH,Y,II,C,lambda)

numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M  = zeros(num,num,numker);
for p = 1 : numker
    KH(:,:,p) = (KH(:,:,p) +KH(:,:,p)')/2+1e-12*eye(num);
    [Up,Vp] = eig(KH(:,:,p));
    M(:,:,p) = Up*sqrt(Vp);
end
tau = ones(num,1);
% tau = mean(II,2);
%%
flag = 1;
iter = 0;
while flag
    iter = iter +1;
    %%--optimize alpha,gamma and xi with CVX---%%
    %%--Calculate eta--%
    tauold = tau;
    eta = 1 + (lambda/num)*(numker*sum(tauold.*tauold)+ sum(sum(II)) - 2*sum(tauold.*sum(II,2)));
    %% CVX
    cvx_begin
    variable alpha(num) 
    variable gamma(numker) 
    variable xi(num)
    variable b
    obj0 = 0;
    for p =1:numker
        obj0 = obj0 + quad_over_lin(M(:,:,p)*alpha,gamma(p));
    end
    minimize ((eta/2)*obj0 + C*sum(xi));
    subject to
    Y.*(reshape(sum(reshape(repmat(II,num,1),num*num,numker).*reshape(KH,num*num,numker),2),num,num)*alpha+b)...
        >=tauold.*(1-xi);
    xi>=0;
    sum(gamma)==1;
    gamma>=0;
    cvx_end
    %%--Calculate Obj
    obj(iter) = ((eta/2)*obj0+C*sum(xi));
    if iter>2 && (obj(iter-1)-obj(iter))/obj(iter)<1e-4
        flag =0;
    else
        %%--optimize tau with QP---%%
        v = 1-xi;
        u = Y.*(reshape(sum(reshape(repmat(II,num,1),num*num,numker).*reshape(KH,num*num,numker),2),num,num)*alpha+b);
        o = mean(II,2);
        tau  =zeros(num,1);
        for i =1:num
            Hi = 1;
            fi = -o(i);
            Ai = v(i);
            bi = u(i);
            xi = quadprog(Hi,fi,Ai,bi);
            tau(i) = xi;
        end
    end
end