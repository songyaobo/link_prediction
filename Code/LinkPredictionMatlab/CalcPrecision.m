function precision  = CalcPrecision( train, test, sim, n )

sim = triu(sim - sim.*train);
test = triu(test);

sim=sim(:);
test=test(:);

[sim, indexSim]=sort(sim,'descend');
indexTe=find(test==1);

sum=0;
for i=1:n
    for j=1:length(indexTe)
        if indexSim(i)==indexTe(j)
            sum=sum+1;
        end
    end
end

precision=sum/n;

