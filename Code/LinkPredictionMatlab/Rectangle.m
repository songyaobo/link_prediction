function [ thisauc ] = Rectangle( train, test,lamda)

rec=train*train*train*train;
rec=diag(rec);
degree=train*train;
degree=diag(degree);

for i=1:size(train)
    res(i)=0.5*(rec(i)-degree(i));
end

sim=res'*res;
thisauc = CalcAUC(train,test,sim, 10000);