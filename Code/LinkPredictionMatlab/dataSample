function[featTrNorm,label_tr,featTeNorm,label_te]=dataSample(train,test)


%%%%%%%%%��Ҫ���ڽӾ���Խ�Ԫ�ش���,ȡ������ξ�����Լӿ��ٶ�

train=full(train);
test=full(test);
non=1-train-test-eye(max(size(train,1),size(train,2)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%��ͬ�ھ�����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnFeat=train*train;

cnFeat_tr=train.*cnFeat;
cnFeat_tr=cnFeat_tr(:);

cnFeat_te=test.*cnFeat;
cnFeat_te=cnFeat_te(:);

cnFeat_non=non.*cnFeat;
cnFeat_non=cnFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ڵ���������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

degreeFeat=sum(train,2);
degreeFeat=repmat(degreeFeat',size(train,1),1)+repmat(degreeFeat,1,size(train,1));
degreeFeat=degreeFeat-cnFeat;

degreeFeat_tr=train.*degreeFeat;
degreeFeat_tr=degreeFeat_tr(:);

degreeFeat_te=test.*degreeFeat;
degreeFeat_te=degreeFeat_te(:);


degreeFeat_non=non.*degreeFeat;
degreeFeat_non=degreeFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PA����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

paFeat=sum(train,2);
paFeat=paFeat*paFeat';

paFeat_tr=train.*paFeat;
paFeat_tr=paFeat_tr(:);

paFeat_te=test.*paFeat;
paFeat_te=paFeat_te(:);

paFeat_non=non.*paFeat;
paFeat_non=paFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%katz����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

katzFeat=train+train*train;

katzFeat_tr=train.*katzFeat;
katzFeat_tr=katzFeat_tr(:);

katzFeat_te=test.*katzFeat;
katzFeat_te=katzFeat_te(:);

katzFeat_non=non.*katzFeat;
katzFeat_non=katzFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%AA����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

aaFeat=train./repmat(log(sum(train,2)),[1,size(train,1)]);
aaFeat(isnan(aaFeat)) = 0; 
aaFeat(isinf(aaFeat)) = 0;
aaFeat=train*aaFeat;

aaFeat_tr=train.*aaFeat;
aaFeat_tr=aaFeat_tr(:);

aaFeat_te=test.*aaFeat;
aaFeat_te=aaFeat_te(:);

aaFeat_non=non.*aaFeat;
aaFeat_non=aaFeat_non(:);
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%simIdx����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nodeNum=size(train,1);
theta=1;
simScoreFeat=zeros(size(train,1),size(train,2));
for i=2:3
    simMat=train^i;
    norm=1;
    for j=2:i
        norm=norm*(nodeNum-j);
    end
%     simScoreFeat=simScoreFeat+simMat./((i-1)*norm);
    simScoreFeat=simScoreFeat+simMat./(exp(-i^2/(2*theta^2))*norm);
end

simScoreFeat(isnan(simScoreFeat)) = 0; 
simScoreFeat(isinf(simScoreFeat)) = 0;

simScoreFeat_tr=train.*simScoreFeat;
simScoreFeat_tr=simScoreFeat_tr(:);

simScoreFeat_te=test.*simScoreFeat;
simScoreFeat_te=simScoreFeat_te(:); 

simScoreFeat_non=non.*simScoreFeat;
simScoreFeat_non=simScoreFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ڵ��������������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
triFeat=train*train*train;
triFeat=diag(triFeat)./2;
triFeat=triFeat*triFeat';

triFeat_tr=train.*triFeat;
triFeat_tr=triFeat_tr(:);

triFeat_te=test.*triFeat;
triFeat_te=triFeat_te(:);

triFeat_non=non.*triFeat;
triFeat_non=triFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ڵ��ı�����������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

recFeat=train*train*train*train;
recFeat=diag(recFeat);
degreeFeat=train*train;
degreeFeat=diag(degreeFeat);

for i=1:size(train)
    recFeat(i)=0.5*(recFeat(i)-degreeFeat(i));
end

recFeat=recFeat'*recFeat;

recFeat_tr=train.*recFeat;
recFeat_tr=recFeat_tr(:);

recFeat_te=test.*recFeat;
recFeat_te=recFeat_te(:);

recFeat_non=non.*recFeat;
recFeat_non=recFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%%%%%%%%%%%%%%%%%%%%%ѵ����/����������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non simScoreFeat_non triFeat_non recFeat_non];
% 
% featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr simScoreFeat_tr triFeat_tr recFeat_tr];
% 
% featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te simScoreFeat_te triFeat_te recFeat_te];

feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non   triFeat_non];

featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr triFeat_tr ];

featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te  triFeat_te];

% feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non   ];
% 
% featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr  ];
% 
% featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te  ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label_non=non;
label_non=label_non(:);
indexNeg=find(label_non==1);
indexNeg=indexNeg(randperm(length(indexNeg)));
% ratioTrain=0.9;
% 
% indexNeg_tr=indexNeg(1:ceil(ratioTrain*length(indexNeg)));
% indexNeg_te=indexNeg(ceil(ratioTrain*length(indexNeg)+1:end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����ѵ����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labelTr=train;
% labelTr=diagDel(labelTr);
labelTr=labelTr(:);

                        %%%%%%%����%%%%%%%
                        
indexPosTr = find(labelTr==1);

labelTr_pos=ones(length(indexPosTr),1);
featTr_pos=zeros(length(indexPosTr),size(featTr,2));

for i=1:length(indexPosTr)
   featTr_pos(i,:)=featTr(indexPosTr(i),:);
end

                       %%%%%%%����%%%%%%%

% indexNeg = find(labelTr==0);
% indexNeg=indexNeg(randperm(length(indexNeg)));

indexNeg_tr=indexNeg(1:length(indexPosTr));




labelTr_neg=zeros(length(indexNeg_tr),1);
featTr_neg=zeros(length(indexNeg_tr),size(featTr,2));

for i=1:length(indexNeg_tr)
   featTr_neg(i,:)=feat_non(indexNeg_tr(i),:);
end

                      %%%%%%%����ѵ����%%%%%%%
label_tr=[labelTr_pos;labelTr_neg];
feat_tr=[featTr_pos;featTr_neg];

                     %%%%%%%ѵ�����һ��%%%%%%%
[featTrNorm,mu,sigma] = featureNormalize(feat_tr);
% featTrNorm=(feat_tr-repmat(min(feat_tr),size(feat_tr,1),1))/(max(feat_tr)-min(feat_tr));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labelTe=test;
% labelTe=diagDel(labelTe);
labelTe=labelTe(:);

                       %%%%%%%����%%%%%%%
                       
indexPosTe = find(labelTe==1);

labelTe_pos=ones(length(indexPosTe),1);
featTe_pos=zeros(length(indexPosTe),size(featTe,2));

for i=1:length(indexPosTe)
   featTe_pos(i,:)=featTe(indexPosTe(i),:);
end
 
                      %%%%%%%����%%%%%%%
                      
% indexNeg = find(labelTe==0);
% indexNeg=indexNeg(randperm(length(indexNeg)));

indexNeg_te=indexNeg(length(indexPosTr)+1:(length(indexPosTr)+1+length(indexPosTe)));

labelTe_neg=zeros(length(indexNeg_te),1);
featTe_neg=zeros(length(indexNeg_te),size(featTe,2));

for i=1:length(indexNeg_te)
   featTe_neg(i,:)=feat_non(indexNeg_te(i),:);
end


                    %%%%%%%���ղ�����%%%%%%%
                    
label_te=[labelTe_pos;labelTe_neg];
feat_te=[featTe_pos;featTe_neg];

                    %%%%%%%�������һ��%%%%%%%
                    
[featTeNorm,mu,sigma] = featureNormalize(feat_te);
% featTeNorm=(feat_te-repmat(min(feat_te),size(feat_te,1),1))/(max(feat_te)-min(feat_te));

% featTeNorm = zeros(size(feat_te,1),size(feat_te,2));
% m=size(feat_te,1);
% for i=1:m
% 	featTeNorm(i,:) = feat_te(i,:)-mu;
% end
% 
% for i=1:m
% 	featTe(i,:)=feat_te(i,:)./sigma;
% end

 





