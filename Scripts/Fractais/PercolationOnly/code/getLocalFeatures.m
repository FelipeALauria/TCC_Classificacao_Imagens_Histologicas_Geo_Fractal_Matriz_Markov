function  [C,Q,M] = getLocalFeatures(aux, L)

%----------------------------
% This function extracts local percolation features using the gliding-box
% algoithm and by calculating the Minkowski distance for each box. Then,
% each of the three percolation properties are evaluated for each scale L
% 
%
% Input:
%   aux - RGB image in double format
%   L - array of values for the box size iterations in the gliding-box
%
% Output:
%   Local features:
%   C - local values for average number of clusters per box
%   Q - local values for occurrence of percolation
%   M - local values for average size of the largest cluster
%   
%
% [1] Roberto, Guilherme F., et al. "Features based on the percolation
% theory for quantification of non-hodgkin lymphomas" Computers in bio-
% logy and medicine 91 (2017): 135-147.

C=zeros(1,size(L,2));%array for local values of the C function
Q=zeros(1,size(L,2)); %array for local values of the Q function
M=zeros(1,size(L,2)); %array for local values of the M function

%% Gliding-box
%Loop for iterating through all box sizes
%If a parallel pool is used, change this to parfor
for k=1:size(L,2)
    arrBigClusters=zeros(1,L(k)^2); %array for storing the size of the biggest cluster of each box
    Ctemp=0; %C function counter
    Qtemp=0; %Q function counter
    nbox=(size(aux,1)-L(k)+1)*(size(aux,2)-L(k)+1); %total number of boxes (see Eq. 1 in [1])
    lim=(L(k)/2)-0.5; %find center pixel position
    countBox=1;
    
    %Loop for iterating through all boxes
    for x=lim+1:(size(aux,1)-lim)
        for y=lim+1:(size(aux,2)-lim)
            xi=x-lim;
            xf=x+lim;
            yi=y-lim;
            yf=y+lim;
            percCount=0; %counter for percolation occurrence
            box=zeros(xf-xi+1,yf-yi+1); %binary matrix for Minkowski distance
            a=0;
            %% Minkowski distance
            %Loop for iterating through all pixels in a box
            for i=xi:xf
                a=a+1;
                b=0;
                for j=yi:yf
                    b=b+1;
                    %Calculate Minkowski distance for each pixel
                    dist=abs(aux(i,j,1)-aux(x,y,1)); %R channel
                    if(dist<=L(k))
                        dist=abs(aux(i,j,2)-aux(x,y,2)); %G channel
                        if(dist<=L(k))
                            dist=abs(aux(i,j,3)-aux(x,y,3)); %B channel
                            if(dist<=L(k))%Label pixel as 1 if criteria is met
                                box(a,b)=1;
                                percCount=percCount+1;
                            else
                                box(a,b)=0;
                            end
                        else
                            box(a,b)=0;
                        end
                    else
                        box(a,b)=0;
                    end
                end
            end
            %% Evaluate percolation properties
            [labeledMatrix,nClusters] = bwlabel(box,4); %Find and label clusters in a box
            [~, F] = mode(labeledMatrix(labeledMatrix>0)); %Find size of biggest cluster
            arrBigClusters(countBox) = F/(L(k)^2); %Counter for M function
            Ctemp=Ctemp+nClusters; %Add number of clusters in the current box to C function counter
            if(percCount/(L(k)^2)>=0.59275) %Checks if percolation threshold is met
                Qtemp=Qtemp+1;
            end
            countBox=countBox+1;
        end
    end
    %Obtain local features
    C(k)=Ctemp/nbox;
    Q(k)=Qtemp/nbox;
    M(k)=mean(arrBigClusters);
end