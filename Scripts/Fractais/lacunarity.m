function [lac] = lacunaridade(p)
    lac=zeros(1,size(p,2));
    for i=1:size(p,2)
        m1=0;
        m2=0;
        for j=1:size(p,1)
            m1=m1+(j*p(j,i));
            m2=m2+((j^2)*p(j,i));
        end
        lac(1,i)=(m2-(m1^2))/(m1^2);
    end

end

