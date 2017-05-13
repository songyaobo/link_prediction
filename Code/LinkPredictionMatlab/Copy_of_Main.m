%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%                    链路预测 ---基于相似性的链路预测算法                       %%%%%%%%%%
%%%%%%%%%%           (1)基于局部结构 (2)基于路径 (3)基于随机游走的算法 (4)其他方法        %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%    对每个数据集：      step-1 划分训练集和测试集                              %%%%%%%%%%
%%%%%%%%%%                       step-2 基于此划分评估所有算法的精读（AUC）              %%%%%%%%%%
%%%%%%%%%%                       重复前面两个过程100次 并取平均值和方差                  %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
%% 参数设定—— 设定训练集的比例 和 独立实验的次数
ratioTrain = 0.9;       %训练集比例
numOfExperiment = 100;  %独立实验的次数
%% 用到的数据集名称
dataname = strvcat('USAir','NS','PB','Yeast','Celegans','FWFB','Power','Router'); 
datapath = 'D:\data\';      %数据集所在的路径
%% 链路预测过程
for ith_data = 1:length(dataname)                                   %遍历每一个数据
    thisdatapath = strcat(datapath,dataname(ith_data,:),'.txt');    %第ith个data的路径
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     %%%%%%% test data 
%     thisdatapath = 'nettest.txt';
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    linklist = load(thisdatapath);              % 导入数据，边的list
    net = FormNet(linklist); clear linklist;    % 根据边的list构成邻接矩阵
    net = triu(net,1);                          % 由于无向网络的对称性，取上三角矩阵以节省空间
    %%% step-1 划分训练集和测试集，每个数据做 count 次独立实验，并将每次的结果存入数组中，用以计算均值和方差
    aucOfallPredictor = []; PredictorsName = [];
    for ith_experiment = 1:numOfExperiment
        %---划分训练集和测试集
        [train, test] = DivideNet(net,ratioTrain);  %返回的train和test也都是上三角矩阵
        ithAUCvector = []; Predictors = [];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%% begin test*********************************************
        train = zeros(size(train)); test = train; train(1,2)=1; train(2,1)=1; train(4,1)=1; train(1,4)=1; train(2,3)=1; train(3,2)=1;
        train(2,5)=1; train(5,2)=1; train(4,5)=1; train(5,4)=1; test(3,5)=1; test(5,3)=1; train(1,5)=1; train(5,1)=1;
        train =sparse(triu(train)); test = sparse(triu(test));
        %%%%%%%%%% end test **********************************************
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %---根据train set计算test set和nonexistent set中所有节点对连边的可能性
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 首先是基于CN的相似性算法
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempauc = CN(train, test); % CN
            Predictors = [Predictors 'CN  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = Salton(train, test); %Salton
            Predictors = [Predictors 'Salton  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = Jaccard(train, test); %Jaccard
            Predictors = [Predictors 'Jaccard  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = Sorenson(train, test); %Sorenson
            Predictors = [Predictors 'Sorenson  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = HPI(train, test); %HPI
            Predictors = [Predictors 'HPI  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = HDI(train, test); %HDI
            Predictors = [Predictors 'HDI  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LHN(train, test); %LHN
            Predictors = [Predictors 'LHN  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = AA(train, test); %AA
            Predictors = [Predictors 'AA  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = RA(train, test); %RA
            Predictors = [Predictors 'RA  ']; ithAUCvector = [ithAUCvector tempauc];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 偏好连接相似性算法
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempauc = PA(train, test); %PA
            Predictors = [Predictors 'PA  ']; ithAUCvector = [ithAUCvector tempauc];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 局部朴素贝叶斯模型
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempauc = LNBCN(train, test); 
            Predictors = [Predictors 'LNBCN  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LNBAA(train, test); 
            Predictors = [Predictors 'LNBAA  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LNBRA(train, test); 
            Predictors = [Predictors 'LNBRA  ']; ithAUCvector = [ithAUCvector tempauc];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 基于路径的相似性指标
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempauc = LocalPath(train, test, 0.0001);
            Predictors = [Predictors 'LocalPath  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = Katz(train, test, 0.01);
            Predictors = [Predictors 'Katz_0.01  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = Katz(train, test, 0.001);
            Predictors = [Predictors 'Katz_0.001  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LHNII(train, test, 0.9);
            Predictors = [Predictors 'LHNII_0.9  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LHNII(train, test, 0.95);
            Predictors = [Predictors 'LHNII_0.95  ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LHNII(train, test, 0.99);
            Predictors = [Predictors 'LHNII_0.99  ']; ithAUCvector = [ithAUCvector tempauc];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 基于随机游走的相似性指标
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempauc = ACT(train, test); %average commute time
            Predictors = [Predictors 'ACT ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = CosPlus(train, test); %cos+ based on Laplacian matrix
            Predictors = [Predictors 'CosPlus ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = RWR(train, test, 0.85); %Random walk with restart
            Predictors = [Predictors 'RWR_0.85 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = RWR(train, test, 0.95); %Random walk with restart
            Predictors = [Predictors 'RWR_0.95 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = SimRank(train, test); %simRank! not completed
            Predictors = [Predictors 'SimRank ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LRW(train, test, 3); %Local random walk
            Predictors = [Predictors 'LRW_3 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LRW(train, test, 4); %Local random walk
            Predictors = [Predictors 'LRW_4 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = LRW(train, test, 5); %Local random walk
            Predictors = [Predictors 'LRW_5 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = SRW(train, test, 3); %Superposed random walk
            Predictors = [Predictors 'SRW_3 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = SRW(train, test, 4); %Superposed random walk
            Predictors = [Predictors 'SRW_4 ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = SRW(train, test, 5); %Superposed random walk
            Predictors = [Predictors 'SRW_5 ']; ithAUCvector = [ithAUCvector tempauc];
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 其他相似性指标
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        tempauc = MFI(train, test); %Matrix forest Index
            Predictors = [Predictors 'MFI ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = TSCN(train, test, 1); %Transfer similarity - Common Neighbor
            Predictors = [Predictors 'TSCN ']; ithAUCvector = [ithAUCvector tempauc];
        tempauc = TSRWR(train, test, 1); %Transfer similarity - Random walk with restart
            Predictors = [Predictors 'TSRWR ']; ithAUCvector = [ithAUCvector tempauc];
            
        aucOfallPredictor = [aucOfallPredictor; ithAUCvector]; PredictorsName = Predictors;
    end
    %% write the results for this data (dataname(ith_data,:))
    avg_auc = mean(aucOfallPredictor,1); var_auc = var(aucOfallPredictor,1);
    respath = strcat(datapath,'result\',dataname(ith_data,:),'_res.txt'); 
    dlmwrite(respath,[PredictorsName; aucOfallPredictor; avg_auc; var_auc], ' ');
end
