function [  thisauc ] = LP_EX( train, test )
%% 计算LP指标并返回AUC值
    sim = train*train;    
    % 二阶路径
    theta=0.01;
   
    sim = exp(-2^2/(2*theta^2))*sim + exp(-3^2/(2*theta^2)) * (train*train*train);
    
    % 二阶路径 + 参数×三节路径
    thisauc = CalcAUC(train,test,sim, 10000);  
    % 评测，计算该指标对应的AUC
end
