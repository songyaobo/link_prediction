function[predicted_label,thisauc,decision_values,svmStruct,precision,recall,F1]=ml(train,test)

%%%%%%%%%需要对邻接矩阵对角元素处理,取上三角形矩阵可以加快速度

train=full(train);
test=full(test);
non=1-train-test-eye(max(size(train,1),size(train,2)));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%共同邻居特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cnFeat=train*train;

cnFeat_tr=train.*cnFeat;
cnFeat_tr=cnFeat_tr(:);

cnFeat_te=test.*cnFeat;
cnFeat_te=cnFeat_te(:);

cnFeat_non=non.*cnFeat;
cnFeat_non=cnFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%节点总数特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PA特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

paFeat=sum(train,2);
paFeat=paFeat*paFeat';

paFeat_tr=train.*paFeat;
paFeat_tr=paFeat_tr(:);

paFeat_te=test.*paFeat;
paFeat_te=paFeat_te(:);

paFeat_non=non.*paFeat;
paFeat_non=paFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%katz特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

katzFeat=train+train*train;

katzFeat_tr=train.*katzFeat;
katzFeat_tr=katzFeat_tr(:);

katzFeat_te=test.*katzFeat;
katzFeat_te=katzFeat_te(:);

katzFeat_non=non.*katzFeat;
katzFeat_non=katzFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%AA特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%simIdx特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%节点三角形总数特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%节点四边形总数特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
 
%%%%%%%%%%%%%%%%%%%%%%%%%%训练样本/测试样本特征%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non simScoreFeat_non triFeat_non recFeat_non];
% 
% featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr simScoreFeat_tr triFeat_tr recFeat_tr];
% 
% featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te simScoreFeat_te triFeat_te recFeat_te];

feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non   triFeat_non];

featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr  triFeat_tr];

featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te  triFeat_te];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label_non=non;
label_non=label_non(:);
indexNeg=find(label_non==1);
indexNeg=indexNeg(randperm(length(indexNeg)));
ratioTrain=0.9;

indexNeg_tr=indexNeg(1:ceil(ratioTrain*length(indexNeg)));
indexNeg_te=indexNeg(ceil(ratioTrain*length(indexNeg)+1:end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%构造训练样本%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labelTr=train;
% labelTr=diagDel(labelTr);
labelTr=labelTr(:);

                        %%%%%%%正样本%%%%%%%
                        
indexPos = find(labelTr==1);

labelTr_pos=ones(length(indexPos),1);
featTr_pos=zeros(length(indexPos),size(featTr,2));

for i=1:length(indexPos)
   featTr_pos(i,:)=featTr(indexPos(i),:);
end

                       %%%%%%%负样本%%%%%%%

% indexNeg = find(labelTr==0);
% indexNeg=indexNeg(randperm(length(indexNeg)));

labelTr_neg=zeros(length(indexNeg_tr),1);
featTr_neg=zeros(length(indexNeg_tr),size(featTr,2));

for i=1:length(indexNeg_tr)
   featTr_neg(i,:)=feat_non(indexNeg_tr(i),:);
end

                      %%%%%%%最终训练样本%%%%%%%
label_tr=[labelTr_pos;labelTr_neg];
feat_tr=[featTr_pos;featTr_neg];

                     %%%%%%%训练样本归一化%%%%%%%
[featTrNorm,mu,sigma] = featureNormalize(feat_tr);
% featTrNorm=(feat_tr-repmat(min(feat_tr),size(feat_tr,1),1))/(max(feat_tr)-min(feat_tr));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%构造测试样本%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labelTe=test;
% labelTe=diagDel(labelTe);
labelTe=labelTe(:);

                       %%%%%%%正样本%%%%%%%
                       
indexPos = find(labelTe==1);

labelTe_pos=ones(length(indexPos),1);
featTe_pos=zeros(length(indexPos),size(featTe,2));

for i=1:length(indexPos)
   featTe_pos(i,:)=featTe(indexPos(i),:);
end
 
                      %%%%%%%负样本%%%%%%%
                      
% indexNeg = find(labelTe==0);
% indexNeg=indexNeg(randperm(length(indexNeg)));

labelTe_neg=zeros(length(indexNeg_te),1);
featTe_neg=zeros(length(indexNeg_te),size(featTe,2));

for i=1:length(indexNeg_te)
   featTe_neg(i,:)=feat_non(indexNeg_te(i),:);
end


                    %%%%%%%最终测试样本%%%%%%%
                    
label_te=[labelTe_pos;labelTe_neg];
feat_te=[featTe_pos;featTe_neg];

                    %%%%%%%测试样本归一化%%%%%%%
                    
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

 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%svm分类%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 svmStruct = svmtrain(label_tr,featTrNorm,'-h 0' );
 [predicted_label, ~, decision_values]=svmpredict(label_te,featTeNorm,svmStruct);
 
 
%  predicted_label(predicted_label==0)=-1;
 [X ,Y]=perfcurve(label_te,decision_values,1);
 plot(X,Y);
 
 thisauc=0;
 for i=1:length(X)-1
     row=X(i+1)-X(i);
     col=Y(i+1)+Y(i);
     thisauc=thisauc+0.5*row*col;
 end
 
  %%%%%%%%%%%%%%%%%%%%%%%%precision/recall/F1%%%%%%%%%%%%%%%%%%%%%%%%%
 tp=0;fn=0;fp=0;tn=0;
 
 for i=1:length(labelTe_pos)
     if predicted_label(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predicted_label(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));


