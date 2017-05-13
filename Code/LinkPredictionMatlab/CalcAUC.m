function [ auc ] = CalcAUC( train, test, sim, n )
%% 计算AUC，输入计算的相似度矩阵
    sim = triu(sim - sim.*train);
    % 只保留测试集和不存在边集合中的边的相似度（自环除外）
    non = 1 - train - test - eye(max(size(train,1),size(train,2)));
    test = triu(test);
    non = triu(non);
    % 分别取测试集和不存在边集合的上三角矩阵，用以取出他们对应的相似度分值
    test_num = nnz(test);
    non_num = nnz(non);
    test_rd = ceil( test_num * rand( 1, n));  
    % ceil是取大于等于的最小整数，n为抽样比较的次数
    non_rd = ceil( non_num * rand( 1, n));
    test_pre = sim .* test;
    non_pre = sim .* non;
    test_data =  test_pre( test == 1 )';  
    % 行向量，test 集合存在的边的预测值
    non_data =  non_pre( non == 1 )';   
    % 行向量，nonexist集合存在的边的预测值
    test_rd = test_data( test_rd );
    non_rd = non_data( non_rd );
    clear test_data non_data;
    n1 = length( find(test_rd > non_rd) );  
    n2 = length( find(test_rd == non_rd));
    auc = ( n1 + 0.5*n2 ) / n;
end
