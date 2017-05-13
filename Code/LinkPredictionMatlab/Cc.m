function [ thisauc ] = Cc( train, test,lamda)

tri=train*train*train;
tri=diag(tri);

degree=train*train;
degree=diag(degree);

for i=1:size(train)
    c(i)=tri(i)/(degree(i)*(degree(i)-1));
end

sim=c'*c;

thisauc = CalcAUC(train,test,sim, 10000);
