function [n] = calcN(p)
    n=zeros(1,size(p,2));
    for i=1:size(p,2)
        for j=1:size(p,1)
           n(i)=n(i)+(p(j,i)/j);
        end
    end
end

