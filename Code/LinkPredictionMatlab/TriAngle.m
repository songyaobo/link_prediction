function [ thisauc ] = TriAngle( train, test,lamda)
%%%%%邻接矩阵三次幂的对角元素则表示的是从任意一个节点出发经过两条边又回到出发点
%%%%%的路径数,也就是包含该顶点的三角形的数量的两倍
tri=train*train*train;
tri=diag(tri)./2;

%两个节点三角形数量相加结果，效果比相乘差。
% tri=repmat(tri,1,size(tri,1));
% sim=tri+tri';

%两个节点三角形数量相乘结果
tri_mat=tri*tri';
sim = lamda*tri_mat;
thisauc = CalcAUC(train,test,sim, 10000);


