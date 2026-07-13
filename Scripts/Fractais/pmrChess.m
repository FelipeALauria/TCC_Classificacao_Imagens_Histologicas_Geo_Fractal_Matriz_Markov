function [p] = pmrChess(img,maxr)%
aux=double(img);
    r=3:2:maxr; %iterate r from 3 to maxr by 2
    p=zeros(r(1,length(r))^2,size(r,2));
    %each box size
    for k=1:size(r,2)
        ncaixas=(size(img,1)-r(k)+1)*(size(img,2)-r(k)+1);
        lim=(r(k)/2)-0.5;
        %go through center pixels
        for x=lim+1:(size(img,1)-lim)
            for y=lim+1:(size(img,2)-lim)
               m=0;
               xi=x-lim;
               xf=x+lim;
               yi=y-lim;
               yf=y+lim;          
               %go throught box
               for i=xi:xf
                   for j=yi:yf
                        dist=abs(aux(i,j,1)-aux(x,y,1));
                        if(dist<=r(k))
                            dist=abs(aux(i,j,2)-aux(x,y,2));
                            if(dist<=r(k))
                                dist=abs(aux(i,j,3)-aux(x,y,3));
                                if(dist<=r(k))
                                     m=m+1;
                                end
                            end
                        end                
                   end                   
               end
               p(m,k)=p(m,k)+1;
            end
        end
        p(:,k)=p(:,k)./ncaixas;
     r(1,k);
    end
end


