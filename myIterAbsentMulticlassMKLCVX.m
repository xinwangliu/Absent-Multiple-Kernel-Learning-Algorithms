function [gamma,alpha,b,xi,tau,obj] = myIterAbsentMulticlassMKLCVX(KH,Y,II,C,lambda)

numker = size(KH,3);
num = size(KH,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M  = zeros(num,num,numker);
for p = 1 : numker
    KH(:,:,p) = (KH(:,:,p) +KH(:,:,p)')/2+1e-12*eye(num);
    [Up,Vp] = eig(KH(:,:,p));
    M(:,:,p) = Up*sqrt(Vp);
end
class_indx = unique(Y);
nc = length(class_indx);
tau = ones(num,nc);
% tau = mean(II,2);
Yone = zeros(num,nc);
for ic = 1 : nc
    Yone(:,ic) = (Y==class_indx(ic)) + (Y~=class_indx(ic))*(-1);
end
%%
flag = 1;
iter = 0;
while flag
    iter = iter +1;
    %%--optimize alpha,gamma and xi with CVX---%%
    %%--Calculate eta--%
    tauold = tau;
    eta = 1;
    for ic =1:nc
        eta = eta + (lambda/(nc*num))*(numker*sum(tauold(:,ic).*tauold(:,ic))+ sum(sum(II)) - 2*sum(tauold(:,ic).*sum(II,2)));
    end
    %% CVX
    cvx_begin
    variable alpha(num,nc) 
    variable gamma(numker) 
    variable xi(num,nc)
    variable b(nc,1)
    obj0 = 0;
    for ic = 1:nc
        for p =1:m
            obj0 = obj0 + quad_over_lin(M(:,:,p)*alpha(:,ic),gamma(p));
        end
    end
    minimize ((eta/2)*obj0 + C*sum(sum(xi)));
    subject to
    Yone.* (reshape(sum(reshape((alpha'*reshape(KH,num,num*mumker)).* repmat(reshape(II,1,num*numker),nc,1),num*nc,numker)...
        ,2),nc,num)' + repmat(b',num,1) ) >= ones(num,nc) - xi1;
%     Yone.*(reshape(sum(reshape(repmat(II,num,1),num*num,numker).*reshape(KH,num*num,numker),2),num,num)*alpha+b)...
%         >=tauold.*(1-xi);
    xi>=0;
    sum(gamma)==1;
    gamma>=0;
    cvx_end
    %%--Calculate Obj
    obj(iter) = ((eta/2)*obj0+C*sum(sum(xi)));
    if iter>2 && (obj(iter-1)-obj(iter))/obj(iter)<1e-4
        flag =0;
    else
        %%--optimize tau with QP---%%
        v = ones(num,nc) - xi1;
        u = Yone.* (reshape(sum(reshape((alpha'*reshape(KH,num,num*mumker)).* repmat(reshape(II,1,num*numker),nc,1),num*nc,numker)...
        ,2),nc,num)' + repmat(b',num,1) );
        o = mean(II,2);
        tau  =zeros(num,1);
        for i =1:num
            for ic = 1:nc
                Hic = 1;
                fic = -o(i);
                Aic = v(i,ic);
                bic = u(i,ic);
                xic = quadprog(Hic,fic,Aic,bic);
                tau(i,ic) = xic;
            end
        end
    end
end