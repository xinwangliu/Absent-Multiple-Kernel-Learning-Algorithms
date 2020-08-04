function [Sigma,Alpsup,w0,pos,Beta,obj] = myAbsentMKLFenchel_Binary(K,y,C,qnorm,II,verbose)

%% II: n*m
num = length(y);
nbkernel=size(K,3);
Sigma = ones(nbkernel,1)/nbkernel;
Beta = ones(num,1)/num;
%------------------------------------------------------------------------------%
% Initialize
%------------------------------------------------------------------------------%
nloop = 0;
loop = 1;
maxIter = 100;
%-----------------------------------------
% Initializing SVM
%------------------------------------------
for p =1:nbkernel
    K(:,:,p) = K(:,:,p).*(II(:,p)*II(:,p)');
end
%------------------------------------------------------------------------------%
% Update Main loop
%------------------------------------------------------------------------------%
while loop
    nloop = nloop+1;
    Betaold = Beta;
    %-----------------------------------------
    % Update SVM parameters
    %----------------------------------------   
    kmatrix=sumKbeta(K,Sigma./(II'*Betaold));
    [Alpsup,w0,pos,obj(nloop)]=mySVMclass(y,C,kmatrix);
    %-----------------------------------------
    % Update Beta
    %----------------------------------------  
    [grad] = gradsvmclass_sparse(K,pos,Alpsup);
    norm_kerS = (-2*grad).*Sigma;
    [Beta] = myFractionalSigmaUpdate(Betaold,norm_kerS,II);
    %-----------------------------------------
    % Update weigths Sigma
    %----------------------------------------
    grad1 = grad./(II'*Beta);
    norm_ker2 = -2*grad1;
    norm_ker = sqrt(norm_ker2);    
    norm_ker = Sigma.* norm_ker;
    % calculate Sigma
    Sigmaold = Sigma;
    % note!!!
    if qnorm == 1
        Sigma = norm_ker/sum(norm_ker);
    else
        q = qnorm;
        norm_ker_q = norm_ker.^(2/(1+q));
        Sigma = norm_ker_q/sum(norm_ker_q.^q).^(1/q);
    end

    if verbose
        if nloop == 1 || rem(nloop,10)==0
            fprintf('--------------------------------------------------\n');
            fprintf('Iter | Obj.    | DiffSigmaSs  | DiffBetaSs |\n');
            fprintf('--------------------------------------------------\n');
        end;
        fprintf('%d   | %8.4f | %6.4f  | %6.4f \n\n',[nloop obj(nloop)   max(abs(Sigma-Sigmaold)) max(abs(Beta-Betaold))]);
    end
    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
    if   max(abs(Sigma - Sigmaold))<1e-4 || nloop>=maxIter
        loop = 0;
        fprintf(1,'variation convergence criteria reached \n');
    end
end