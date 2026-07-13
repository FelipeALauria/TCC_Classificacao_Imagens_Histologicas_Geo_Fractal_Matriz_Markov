function  [MaxClusterIndex, MaxPercIndex, MaxMaxClusterIndex, AreaRatioMaxCluster, MaxMaxCluster, SkewnessMaxCluster, AreaMaxCluster, AreaRatioCluster, AreaRatioPerc, MaxCluster, MaxPerc, SkewnessCluster, SkewnessPerc, AreaPerc, AreaCluster, p,g,h] = percManh(img,maxr)%inicia função de geração da matriz de prob. com parãmetros imagem e R Max
    aux=double(img);
    r=3:2:maxr; %iterar de r de 3 a R max de 2 em 2
    g=zeros(1,size(r,2)); %criação de vetor para armazenar os valores de percolação para cada tamanho de r
    p=zeros(1,size(r,2));%criação de vetor para armazenar os valores de n. aglomerados para cada tamanho de r
    h=zeros(1,size(r,2)); %vetor para armazenar o valor do maior aglomerado médio para cada tamanho de r
    %cada tamanho de caixa
    parfor k=1:size(r,2)
        vetBigClusters=zeros(1,r(k)^2); %vetor que armazena a ocupação do maior cluster em cada caixa
        ptemp=0;
        gtemp=0;
        ncaixas=(size(img,1)-r(k)+1)*(size(img,2)-r(k)+1);
        lim=(r(k)/2)-0.5;
        %percorrer os pixeis centrais
        caixa=1;
        for x=lim+1:(size(img,1)-lim)         
            for y=lim+1:(size(img,2)-lim)
               xi=x-lim;
               xf=x+lim;
               yi=y-lim;
               yf=y+lim;          
               %percorre a caixa
               percCount=0;
            box=zeros(xf-xi+1,yf-yi+1);
            a=0;
               for i=xi:xf
                   a=a+1;
                   b=0;
                   for j=yi:yf
                       b=b+1;
						dist = abs(aux(i, j, 1)-aux(x, y, 1))+abs(aux(i, j, 2)-aux(x, y, 2))+abs(aux(i, j, 3)-aux(x, y, 3));
                        if(dist<=r(k))
                            box(a,b)=1;
                            percCount=percCount+1;
                        else
                            box(a,b)=0;
                        end
                   end
               end   
               [L,ROTULO] = bwlabel(box,4);
               [~, F] = mode(L(L>0)); %calcula qual a frequência do maior aglomerado nesta caixa
               vetBigClusters(caixa) = F/(r(k)^2); %armazena a porcentagem que ele ocupa na caixa
                ptemp=ptemp+ROTULO;
                if(percCount/(r(k)^2)>=0.59275) %se o número total de pixels percolantes ofr maior que o limiar
                    gtemp=gtemp+1;
                end                
                caixa=caixa+1;
            end
        end 
     p(k)=ptemp/ncaixas;
     g(k)=gtemp/ncaixas;
     h(k)=mean(vetBigClusters);
    end
    AreaCluster = trapz(p);
    AreaPerc = trapz(g);
    AreaMaxCluster = trapz(h);
    SkewnessCluster = skewness(p);
    SkewnessPerc = skewness(g);
    SkewnessMaxCluster = skewness(h);
    [MaxCluster,MaxClusterIndex] = max(p);
    [MaxPerc,MaxPercIndex] = max(g);
    [MaxMaxCluster,MaxMaxClusterIndex] = max(h);
    half = ceil(length(p)/2);
    AreaRatioCluster = trapz(p(half+1:end))/trapz(p(1:half));
    AreaRatioPerc = trapz(g(half+1:end))/trapz(g(1:half));
    AreaRatioMaxCluster = trapz(h(half+1:end))/trapz(h(1:half));
end