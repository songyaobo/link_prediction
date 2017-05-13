function[acc,thisauc,precision,recall,F1]=classifier(train,test)

%%%%%%%%%��Ҫ���ڽӾ���Խ�Ԫ�ش���,ȡ�������ξ�����Լӿ��ٶ�

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�ڵ���������������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% recFeat=train*train*train*train;
% recFeat=diag(recFeat);
% degreeFeat=train*train;
% degreeFeat=diag(degreeFeat);
% 
% for i=1:size(train)
%     recFeat(i)=0.5*(recFeat(i)-degreeFeat(i));
% end
% 
% recFeat(isnan(recFeat)) = 0; 
% recFeat(isinf(recFeat)) = 0;
% 
% recFeat=recFeat'*recFeat;
% 
% recFeat_tr=train.*recFeat;
% recFeat_tr=recFeat_tr(:);
% 
% recFeat_te=test.*recFeat;
% recFeat_te=recFeat_te(:);
% 
% recFeat_non=non.*recFeat;
% recFeat_non=recFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����ϵ������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% a3=train*train*train;
% a3=diag(a3)./2;
% a2=train*train;
% a2=diag(a2);
% cluster=2*a3./(a2.*(a2-1));
% 
% cluster(isnan(cluster)) = 0; 
% cluster(isinf(cluster)) = 0;
% 
% clusterFeat=cluster*cluster';

% a3=train*train*train;
% a2=train*train;
% clusterFeat=a3./(a2.*(a2-1));
% clusterFeat(isnan(clusterFeat)) = 0; 
% clusterFeat(isinf(clusterFeat)) = 0;
% 
% clusterFeat_tr=train.*clusterFeat;
% clusterFeat_tr=clusterFeat_tr(:);
% 
% clusterFeat_te=test.*clusterFeat;
% clusterFeat_te=clusterFeat_te(:);
% 
% clusterFeat_non=non.*clusterFeat;
% clusterFeat_non=clusterFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���·������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [distanceFeat,~]=Aver_Path_Length(train);
% 
% distanceFeat_tr=train.*distanceFeat;
% distanceFeat_tr=distanceFeat_tr(:);
% 
% distanceFeat_te=test.*distanceFeat;
% distanceFeat_te=distanceFeat_te(:);
% 
% distanceFeat_non=non.*distanceFeat;
% distanceFeat_non=distanceFeat_non(:);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%%%%%%%%%%%%%%%%%%%%%ѵ������/������������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non simScoreFeat_non triFeat_non recFeat_non];
% 
% featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr simScoreFeat_tr triFeat_tr recFeat_tr];
% 
% featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te simScoreFeat_te triFeat_te recFeat_te];

% feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non   triFeat_non ];
% 
% featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr triFeat_tr ];
% 
% featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te   triFeat_te];

feat_non=[cnFeat_non degreeFeat_non paFeat_non aaFeat_non katzFeat_non   ];

featTr=[cnFeat_tr degreeFeat_tr paFeat_tr aaFeat_tr katzFeat_tr  ];

featTe=[cnFeat_te degreeFeat_te paFeat_te aaFeat_te katzFeat_te  ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

label_non=non;
label_non=label_non(:);
indexNeg=find(label_non==1);
indexNeg=indexNeg(randperm(length(indexNeg)));
% ratioTrain=0.9;
% 
% indexNeg_tr=indexNeg(1:ceil(ratioTrain*length(indexNeg)));
% indexNeg_te=indexNeg(ceil(ratioTrain*length(indexNeg)+1:end));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%����ѵ������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labelTr=train;
% labelTr=diagDel(labelTr);
labelTr=labelTr(:);

                        %%%%%%%������%%%%%%%
                        
indexPosTr = find(labelTr==1);

labelTr_pos=ones(length(indexPosTr),1);
featTr_pos=zeros(length(indexPosTr),size(featTr,2));

for i=1:length(indexPosTr)
   featTr_pos(i,:)=featTr(indexPosTr(i),:);
end

                       %%%%%%%������%%%%%%%

% indexNeg = find(labelTr==0);
% indexNeg=indexNeg(randperm(length(indexNeg)));

indexNeg_tr=indexNeg(1:length(indexPosTr));




labelTr_neg=zeros(length(indexNeg_tr),1);
featTr_neg=zeros(length(indexNeg_tr),size(featTr,2));

for i=1:length(indexNeg_tr)
   featTr_neg(i,:)=feat_non(indexNeg_tr(i),:);
end

                      %%%%%%%����ѵ������%%%%%%%
label_tr=[labelTr_pos;labelTr_neg];
feat_tr=[featTr_pos;featTr_neg];

                     %%%%%%%ѵ��������һ��%%%%%%%
[featTrNorm,mu,sigma] = featureNormalize(feat_tr);
% featTrNorm=(feat_tr-repmat(min(feat_tr),size(feat_tr,1),1))/(max(feat_tr)-min(feat_tr));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%�����������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labelTe=test;
% labelTe=diagDel(labelTe);
labelTe=labelTe(:);

                       %%%%%%%������%%%%%%%
                       
indexPosTe = find(labelTe==1);

labelTe_pos=ones(length(indexPosTe),1);
featTe_pos=zeros(length(indexPosTe),size(featTe,2));

for i=1:length(indexPosTe)
   featTe_pos(i,:)=featTe(indexPosTe(i),:);
end
 
                      %%%%%%%������%%%%%%%
                      
% indexNeg = find(labelTe==0);
% indexNeg=indexNeg(randperm(length(indexNeg)));

indexNeg_te=indexNeg(length(indexPosTr)+1:(length(indexPosTr)+1+length(indexPosTe)));

labelTe_neg=zeros(length(indexNeg_te),1);
featTe_neg=zeros(length(indexNeg_te),size(featTe,2));

for i=1:length(indexNeg_te)
   featTe_neg(i,:)=feat_non(indexNeg_te(i),:);
end


                    %%%%%%%���ղ�������%%%%%%%
                    
label_te=[labelTe_pos;labelTe_neg];
feat_te=[featTe_pos;featTe_neg];

                    %%%%%%%����������һ��%%%%%%%
                    
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


 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%svm����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 svmStruct = svmtrain(label_tr,featTrNorm,'-h 0  ' ); 
 [predicted_label, acc, decision_values]=svmpredict(label_te,featTeNorm,svmStruct);

 %%%%%%%%%%%%%%AUC%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
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
 
 disp('svm');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);
 
 
 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%RF����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
b = TreeBagger(200,featTrNorm,label_tr,'oobpred','on');
plot(oobError(b));
[predict_labelRf,scores]=predict(b,featTeNorm);
predict_labelRf=str2num(char(predict_labelRf));
acc=sum(predict_labelRf==label_te)/length(label_te);
 


%%%%%%%%%%%%%%%AUC%%%%%%%%%%%%%%%%%
 [X ,Y]=perfcurve(label_te,scores(:,2),1);

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
     if predict_labelRf(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predict_labelRf(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));
 
 
 disp('RF');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);
 
 
 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%NB����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model=NaiveBayes.fit(featTrNorm,label_tr);
[scores, predict_labelNB]=posterior(model,featTeNorm);

acc=sum(predict_labelNB==label_te)/length(label_te);
 
 
 %%%%%%%%%%%%%%%AUC%%%%%%%%%%%%%%%%%
 [X ,Y]=perfcurve(label_te,scores(:,2),1);

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
     if predict_labelNB(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predict_labelNB(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));
 
 disp('NB');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);
 
 
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%boost����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 


model=fitensemble(featTrNorm,label_tr,'AdaBoostM1',100,'tree');
[predict_labelBoost,scores]=predict(model,featTeNorm);

acc=sum(predict_labelBoost==label_te)/length(label_te);

 
 %%%%%%%%%%%%%%%AUC%%%%%%%%%%%%%%%%%
 [X ,Y]=perfcurve(label_te,scores(:,2),1);

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
     if predict_labelBoost(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predict_labelBoost(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));
 
 disp('Boost');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);
 
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%bagging����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model=fitensemble(featTrNorm,label_tr,'AdaBoostM1',100,'tree','type','classification');
[predict_labelBagging,scores]=predict(model,featTeNorm);

acc=sum(predict_labelBagging==label_te)/length(label_te);
  

 
 %%%%%%%%%%%%%%%AUC%%%%%%%%%%%%%%%%%
 [X ,Y]=perfcurve(label_te,scores(:,2),1);

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
     if predict_labelBagging(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predict_labelBagging(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));
 
 
 disp('Bagging');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);
 
 
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%discriminat����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
model=ClassificationDiscriminant.fit(featTrNorm,label_tr);
[predict_labelDisc,scores]=predict(model,featTeNorm);

acc=sum(predict_labelDisc==label_te)/length(label_te);

%%%%%%%%%%%%%%%AUC%%%%%%%%%%%%%%%%%
 [X ,Y]=perfcurve(label_te,scores(:,2),1);

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
     if predict_labelDisc(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predict_labelDisc(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));
 
 disp('Disciminant');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);
 
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%knn����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%  svmStruct = svmtrain(label_tr,featTrNorm,'-h 0 ' );
%  [predicted_label, thisauc, decision_values]=svmpredict(label_te,featTeNorm,svmStruct);

model=ClassificationKNN.fit(featTrNorm,label_tr,'NumNeighbors',5);
[predict_label,prob,cost]=predict(model,featTeNorm);

acc=sum(predict_label==label_te)/length(label_te);
  
 [X ,Y]=perfcurve(label_te,prob(:,2),1);
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
     if predict_label(i)==1
         tp=tp+1;
     else
         fn=fn+1;
     end
 end
 
 for i=length(labelTe_pos)+1:length(label_te)
     if predict_label(i)==0
         tn=tn+1;
     else
         fp=fp+1;
     end
 end

 
 precision=tp/(tp+fp);
 recall=tp/(tp+fn);
 F1=2*tp/(tp-tn+length(label_te));
 
 disp('knn');disp('acc');disp(acc);disp('auc');disp(thisauc);disp('precision');
 disp(precision);disp('recall');disp(recall);disp('F1');disp(F1);


