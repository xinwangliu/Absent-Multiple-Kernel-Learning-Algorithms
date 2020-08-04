function [Sigma,w,bsvm,posw,obj]= myMKLSILP(K,yapp,C)

nbkernel=size(K,3);
Sigma=ones(nbkernel,1)/nbkernel;
verbose=0;
sumSigma=sum(Sigma);
theta=-inf;
%---------------------------------------------------------------------
% Setting the linear prog  parameters
% nbvar = nbkernel+1;
%
% var = [theta Sigma1, Sigma_2,   ..., SigmaK];
%---------------------------------------------------------------------
f=[-1;zeros(nbkernel,1)];
Aeq=[0 ones(1,nbkernel)]; % 1 seule egalit�;
beq=sumSigma;
LB=[-inf;zeros(nbkernel,1)];
UB=[inf*ones(nbkernel,1)];
A=[];
b=[];
optimopt = optimset('MaxIter',10000,'Display','off', 'TolCon',1e-3,'TolFun',1e-5);
nbitermax = 100;
nbverbose=1;

iter=0;
Sigmaold=Sigma;
Sigmaold(1)=Sigmaold(1)-1;
loop=1;
exitflag=0;

while loop
    iter = iter +1;
    Kmatrix = sumKbeta(K,Sigma);
    [w,bsvm,posw,obj0] = mySVMclass(yapp,C,Kmatrix);
    Saux = zeros(nbkernel,1);
    for p=1:nbkernel
        Saux(p) = 0.5*w'*K(posw,posw,p)*w;% - sum(alphaw);
    end
    S=Saux-sum(abs(w));
    constraintviol = S'*Sigma;

    sumfk2divdk= Saux'*Sigma;
    primalobj= sumfk2divdk  +C*sum(max( 1-yapp.*(Kmatrix(:,posw)*w + bsvm),0));
    dualobj= -max(Saux) + sum(abs(w));
    dualitygap=(primalobj-dualobj)/primalobj;
    %------------------------------------------------------
    % verbosity
    %----------------------------------------------------
    if verbose ~= 0

        if nbverbose == 1
            disp('------------------------------------------------');
            disp('iter     Theta      ConstViol     DeltaSigma');
            disp('------------------------------------------------');
        end
        if nbverbose == 20
            nbverbose=1;
        end

        if exitflag==0
            fprintf('%d   | %8.4f | %8.4f | %6.4f |%6.4f \n',[iter theta constraintviol  max(abs(Sigma-Sigmaold))], dualitygap);
        else
            fprintf('%d   | %8.4f | %8.4f | %6.4f | lp cvg pb \n',[iter theta constraintviol  max(abs(Sigma-Sigmaold))]);
        end;
        nbverbose = nbverbose+1;
    end
    %----------------------------------------------------
    % check variation of Sigma conditions
    %----------------------------------------------------
    if  max(abs(Sigma - Sigmaold))< 1e-4 || ( iter>2 && abs((obj(iter-1)-obj(iter-2))/obj(iter-2))<1e-4 )
        loop=0;
        fprintf(1,'variation convergence criteria reached \n');
    end
    %----------------------------------------------------
    % check nbiteration conditions
    %----------------------------------------------------
    if iter>=nbitermax
        loop = 0;
        fprintf(1,'Maximal number of iterations reached \n');
    end
    %----------------------------------------------------
    %  Optimize the weigths Sigma using a LP
    %----------------------------------------------------
    Sigmaold = Sigma;
    A = [A, [1 -S']'];
    aux = 0;
    b = [b;aux];
   [x,fval,exitflag] =linprog(f,A',b,Aeq,beq,LB,UB,[theta;Sigma],optimopt);
   obj(iter) = -x(1);
   Sigma = x(2:end);
end