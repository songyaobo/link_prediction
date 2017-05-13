function mat=diagDel(matOri)

[m,n]=size(matOri);
mat=[];
mat=matOri(1,2:n);
for i=2:m
    mat=[mat;matOri(i,[1:i-1 i+1:n])];
end

end