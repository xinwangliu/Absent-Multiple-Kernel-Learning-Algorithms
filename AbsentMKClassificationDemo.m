clear
clc
warning off;

dataName = 'heart'; %%% flower17; flower102; CCV; caltech101_numofbasekernel_10
%% %% washington; wisconsin; texas; cornell
%% ionosphere; heart
load(['./',dataName,'_Kmatrix'],'KH','Y');
% load([path,'datasets\',dataName,'_Kmatrix'],'KH','Y');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numclass = length(unique(Y));
numker = size(KH,3);
num = size(KH,1);

verbose=1;
%------------------------------------------------------------------------
C = 2^0;
% lamtbda = 2^3;
KH = kcenter(KH);
KH = knorm(KH);


epsionset = [0.1:0.1:0.9];
for ie = 1:length(epsionset)
     for iter = 1:10
        load(['.\generateAbsentMatrix\',dataName,'_missingRatio_',num2str(epsionset(ie)),...
            '_missingIndex_iter_',num2str(iter),'.mat'],'S');
        II = zeros(num,numker);
        for p = 1 : numker
            %% missing index: S{p}.indx
            mis_indx = S{p}.indx';
            obs_indx = setdiff(1:num,mis_indx);
            II(obs_indx,p) = 1;
        end

       qnorm = 1;
       verbose = 1;
       [Sigma,Alpsup,w0,pos,Beta,obj] = myAbsentMKLFenchel_Binary(KH,Y,C,qnorm,II,verbose);
        ypred = zeros(num,1);
        for i =1:num
            ypred(i) = (Alpsup'*sum(repmat(II(i,:),length(pos),1).*reshape(KH(pos,i,:),length(pos),numker),2)+w0);
        end
        acc(iter) = mean(sign(ypred)==Y);
     end
end

