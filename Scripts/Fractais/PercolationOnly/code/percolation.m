function  [globalFeatures, C,Q,M] = percolation(img,maxL)

%----------------------------
% This function extracts local and global percolation features from a
% RGB image. Used for the first time in [1]
%
% Input:
% img - RGB image
% maxL - maximum size of the L scale. Must a be an odd number greater
% or equal than 3
%
% Output:
%   C - local values for average number of clusters per box
%   Q - local values for occurrence of percolation
%   M - local values for average size of the largest cluster
%   globalFeatures: struct containing all the 15 global percolation
%
% [1] Roberto, Guilherme F., et al. "Features based on the percolation
% theory for quantification of non-hodgkin lymphomas" Computers in bio-
% logy and medicine 91 (2017): 135-147.

aux=double(img); %converts input image to double
L=3:2:maxL; %iterates L by an increment of 2 (this can be changed, but the increment must always be an even number

[C,Q,M] = getLocalFeatures(aux,L);

%Obtain global features
globalFeatures = getGlobalFeatures(C,Q,M);

end