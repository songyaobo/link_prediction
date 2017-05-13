function [ thisauc ] = simIdx( train,test,length)

nodeNum=size(train,1);
theta=1;
simScore=zeros(size(train,1),size(train,2));
for i=2:length
    simMat=train^i;
    norm=1;
    for j=2:i
        norm=norm*(nodeNum-j);
    end
    simScore=simScore+simMat./((i-1)*norm);
    %%%%线性局部加权回归权重。
%     simScore=simMat./(exp(-i^2/(2*theta^2))*norm);
end

% selectNode=round(nodeNum*0.8);
% [result,index]=sort(simScore,2);
% temp=index>=(nodeNum-selectNode);
% simScore=simScore.*temp;

% tri=train*train*train;
% tri=diag(tri)./2;
% 
% %两个节点三角形数量相加结果，效果比相乘差。
% % tri=repmat(tri,1,size(tri,1));
% % sim=tri+tri';
% 
% %两个节点三角形数量相乘结果
% lamda=0;
% tri_mat=tri*tri';
% simScore=(1-lamda)*simScore+lamda*tri_mat;



thisauc = CalcAUC(train,test,simScore, 10000);

end
