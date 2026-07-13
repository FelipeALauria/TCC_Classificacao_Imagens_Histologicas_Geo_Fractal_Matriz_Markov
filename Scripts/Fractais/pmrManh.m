function [p] = pmrManh(img,maxr)%inicia função de geração da matriz de prob. com parãmetros imagem e R Max
aux=double(img);
    r=3:2:maxr; %iterar de r de 3 a R max de 2 em 2
    p=zeros(r(1,length(r))^2,size(r,2));
    %cada tamanho de caixa
    for k=1:size(r,2)
        ncaixas=(size(img,1)-r(k)+1)*(size(img,2)-r(k)+1);
        lim=(r(k)/2)-0.5;
        %percorrer os pixeis centrais
        for x=lim+1:(size(img,1)-lim)
            for y=lim+1:(size(img,2)-lim)
               m=0;
               xi=x-lim;
               xf=x+lim;
               yi=y-lim;
               yf=y+lim;          
               %percorre a caixa
               for i=xi:xf
                   for j=yi:yf
                        dist = abs(aux(i, j, 1)-aux(x, y, 1))+abs(aux(i, j, 2)-aux(x, y, 2))+abs(aux(i, j, 3)-aux(x, y, 3));
                        if(dist<=r(k))
                            m=m+1;
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


