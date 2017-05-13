function [  thisauc ] = RWR( train, test, lambda )
%根据train计算所有在train set中未连边的节点对的 RWR 相似度
    train = train + train';
    deg = sum(train,2);
    deg = repmat(deg,[1,size(train,2)]);
    train = train ./ deg; clear deg;    %求转移矩阵
      
    
    I = sparse(eye(size(train,1)));
%     % 对于每个节点，求它与其他节点之间的RWR相似度
%     for nodei = 1:size(train,1)
%        sim(nodei,:) =  (1 - lambda) * inv(I- lambda * train') * I(:, nodei);
%     end
%     sim = sim+sim';
    sim = (1 - lambda) * inv(I- lambda * train') * I;
    sim = sim+sim';

%     lastsim = zeros(size(train,1),size(train,2)); % 迭代过程
%     thissim = eye(size(train,1));
%     while (sum(abs(thissim - lastsim))>0.00000000000001)
%         lastsim = thissim;
%         thissim = (1-lambda)*I + lambda * train' * lastsim;
%     end
%     thissim = thissim + thissim';
    
    train = spones(train);   
    sim = triu(sim,1);
    sim = sim - sim.*train;
    %评测，计算AUC
    thisauc = CalcAUC(train,test,sim);
end
