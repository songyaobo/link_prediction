%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%                     链路预测 ---基于相似性的链路预测算法                       %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    (1)基于局部结构-共同邻居 (2)基于路径 (3)基于随机游走的算法 (4)其他方法        %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%    对每个数据集：      step-1 划分训练集和测试集                               %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%                       step-2 基于此划分评估所有算法的精度（AUC）               %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%                       重复前面两个过程100次 并取平均值和方差                   %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
%% 参数设定—— 设定训练集的比例 和 独立实验的次数
ratioTrain = 0.9;             %训练集比例
numOfExperiment = 2;        %独立实验的次数
%% 用到的数据集名称
dataname = strvcat('USAir','NS','PB','Yeast','Celegans','FWFB','Power','Router'); 
datapath = 'D:\research\completed\Link prediction\writting book\data\';       %数据集所在的路径
% 结果的格式
% 第一行：算法名
% 倒数第一行：每种算法在多次实验中的方差；
% 倒数第二行：每种算法在多次实验中的平均值；
% 中间数据：每种算法在多次实验中的具体值。

%% 链路预测过程
for ith_data = 1:2                              
    % 遍历每一个数据
    tempcont = strcat('正在处理第 ', int2str(ith_data), '个数据...', dataname(ith_data,:));
    disp(tempcont);
    tic;
    thisdatapath = strcat(datapath,dataname(ith_data,:),'.txt');    % 第ith个数据的路径
    linklist = load(thisdatapath);                                  % 导入数据（边的list）
    net = FormNet(linklist); clear linklist;                       % 根据边的list构成邻接矩阵
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%  每个数据做 numOfExperiment（100）次独立实验， 并将所有结果存到（实验次数×预测器个数）阶的矩阵中用以计算均值和方差
    aucOfallPredictor = [];                                         % 用于存储100次实验的结果，第j行表示第j次实验
    PredictorsName = [];                                            % 记录预测器的顺序
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %----- 开始100次实验的循环
    for ith_experiment = 1:numOfExperiment
        if mod(ith_experiment,10) == 0
            tempcont = strcat(int2str(ith_experiment),'%... ');
            disp(tempcont);
        end
        %%% step-1 划分训练集和测试集，保证训练集的连通性
        [train, test] = DivideNet(net,ratioTrain);                  % 划分训练集和测试集，返回上三角矩阵
        train = sparse(train); test = sparse(test);
        train = spones(train + train'); test = spones(test+test');
        ithAUCvector = []; Predictors = [];                         % 用于存储当前实验中所有预测器的精度
        %%% step-2 根据train set计算test set和nonexistent set中所有节点对产生（或存在）连边的可能性，并得出AUC
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 首先是基于Common Neighbor（共同邻居）的相似性算法
        disp('CN...');
        tempauc = CN(train, test);                  % Common Neighbor
            Predictors = [Predictors '%CN	'];      ithAUCvector = [ithAUCvector tempauc];
        
        disp('Salton...');
        tempauc = Salton(train, test);              % Salton Index
             Predictors = [Predictors 'Salton	'];  ithAUCvector = [ithAUCvector tempauc];
        
        disp('Jaccard...');
        tempauc = Jaccard(train, test);             % Jaccard Index
             Predictors = [Predictors 'Jaccard	'];  ithAUCvector = [ithAUCvector tempauc];  

        disp('Sorenson...');
        tempauc = Sorenson(train, test);            % Sorenson Index
             Predictors = [Predictors 'Sorens	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('HPI...');
        tempauc = HPI(train, test);                 % Hub Promoted Index
             Predictors = [Predictors 'HPI	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('HDI...');
        tempauc = HDI(train, test);                 % Hub Depressed Index
             Predictors = [Predictors 'HDI	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHN...');
        tempauc = LHN(train, test);                 % Leicht-Holme-Newman
             Predictors = [Predictors 'LHN	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('AA...');
        tempauc = AA(train, test);                  % Adar-Adamic Index
             Predictors = [Predictors 'AA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('RA...');
        tempauc = RA(train, test);                  % Resourse Allocation
             Predictors = [Predictors 'RA	'];       ithAUCvector = [ithAUCvector tempauc];  
       
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 偏好连接相似性算法
        disp('PA...');
        tempauc = PA(train, test);                  % Preferential Attachment
             Predictors = [Predictors 'PA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 局部朴素贝叶斯模型
        disp('LNBCN...');
        tempauc = LNBCN(train, test);               % Local naive bayes method - Common Neighbor
             Predictors = [Predictors 'LNBCN	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LNBAA...');
        tempauc = LNBAA(train, test);               % Local naive bayes method - Adar-Adamic Index
             Predictors = [Predictors 'LNBAA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LNBRA...');
        tempauc = LNBRA(train, test);               % Local naive bayes method - Resource Allocation
             Predictors = [Predictors 'LNBRA	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 基于路径的相似性指标
        disp('LocalPath...');
        tempauc = LocalPath(train, test, 0.0001);   % Local Path Index
             Predictors = [Predictors 'LocalP	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('Katz 0.01...');
        tempauc = Katz(train, test, 0.01);          % Katz Index 参数取0.01
             Predictors = [Predictors 'Katz.01	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('Katz 0.001...');
        tempauc = Katz(train, test, 0.001);         % Katz Index 参数取0.001
             Predictors = [Predictors '~.001	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHNII 0.9...');
        tempauc = LHNII(train, test, 0.9);          % Leicht-Holme-Newman II 参数取0.9
             Predictors = [Predictors 'LHNII.9	'];    ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHNII 0.95...');
        tempauc = LHNII(train, test, 0.95);         % Leicht-Holme-Newman II 参数取0.95
             Predictors = [Predictors '~.95	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LHNII 0.99...');
        tempauc = LHNII(train, test, 0.99);         % Leicht-Holme-Newman II 参数取0.99
             Predictors = [Predictors '~.99	'];       ithAUCvector = [ithAUCvector tempauc];  
             
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 基于随机游走的相似性指标
        disp('ACT...');
        tempauc = ACT(train, test);                 % Average commute time
             Predictors = [Predictors 'ACT	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('CosPlus...');
        tempauc = CosPlus(train, test);             % Cos+ based on Laplacian matrix
             Predictors = [Predictors 'CosPlus	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('RWR 0.85...');
        tempauc = RWR(train, test, 0.85);           % Random walk with restart 参数取0.85
             Predictors = [Predictors 'RWR.85	'];   ithAUCvector = [ithAUCvector tempauc];  
        
        disp('RWR 0.95...');
        tempauc = RWR(train, test, 0.95);           % Random walk with restart 参数取0.95
             Predictors = [Predictors '~.95	'];      ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SimRank 0.8...');
        tempauc = SimRank(train, test, 0.8);        % SimRank
             Predictors = [Predictors 'SimR	'];      ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LRW 3...');
        tempauc = LRW(train, test, 3, 0.85);        % Local random walk 步数取到3
             Predictors = [Predictors 'LRW_3	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LRW 4...');
        tempauc = LRW(train, test, 4, 0.85);        % Local random walk 步数取到4
             Predictors = [Predictors '~_4	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('LRW 5...');
        tempauc = LRW(train, test, 5, 0.85);        % Local random walk 步数取到5
             Predictors = [Predictors '~_5	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SRW 3...');
        tempauc = SRW(train, test, 3, 0.85);        % Superposed random walk 步数取到3
             Predictors = [Predictors 'SRW_3	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SRW 4...');
        tempauc = SRW(train, test, 4, 0.85);        % Superposed random walk 步数取到4
             Predictors = [Predictors '~_4	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('SRW 5...');
        tempauc = SRW(train, test, 5, 0.85);        % Superposed random walk 步数取到5
             Predictors = [Predictors '~_5	'];       ithAUCvector = [ithAUCvector tempauc];  
             
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 其他相似性指标
        disp('MFI...');
        tempauc = MFI(train, test);                 % Matrix forest Index
             Predictors = [Predictors 'MFI	'];       ithAUCvector = [ithAUCvector tempauc];  
        
        disp('TS...');
        tempauc = TSCN(train, test, 0.01);          % Transfer similarity - Common Neighbor
             Predictors = [Predictors 'TSCN	'];       ithAUCvector = [ithAUCvector tempauc];  
        %%% 所有相似性算法计算完成
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% 将此次得到的算法精度（行向量）存到矩阵aucOfallPredictor中，用于最后计算平均值和方差
        aucOfallPredictor = [aucOfallPredictor; ithAUCvector]; PredictorsName = Predictors;
    end % 100次独立循环结束
    %% write the results for this data (dataname(ith_data,:))
    avg_auc = mean(aucOfallPredictor,1); var_auc = var(aucOfallPredictor,1);
    respath = strcat(datapath,'result\',dataname(ith_data,:),'_res.txt');         %结果输出所在路径和文件名
    dlmwrite(respath,{PredictorsName}, '');
    dlmwrite(respath,[aucOfallPredictor; avg_auc; var_auc], '-append','delimiter', '	','precision', 4);
    toc;
end % 所有数据计算结束
