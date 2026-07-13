function  globalFeatures = getGlobalFeatures(C,Q,M)

%----------------------------
% This function extracts global percolation features the C, Q and M
% functions
%
% Input:
%   C - local values for average number of clusters per box
%   Q - local values for occurrence of percolation
%   M - local values for average size of the largest cluster
%
% Output:
%   globalFeatures: struct containing all the 15 global percolation
%   features:
%       aC,aQ,aM - area under the curve for the C, Q and M functions
%       skC,skQ,skM - skewness for the C, Q and M functions
%       arC,arQ,arM - area ratio for the C, Q and M functions
%       maxC,maxQ,maxM - maximum point of the C, Q and M functions
%       sigmaC,sigmaQ,sigmaM - scale L where the maximum point of the C, Q
%       and M functions occurs
%
% [1] Roberto, Guilherme F., et al. "Features based on the percolation
% theory for quantification of non-hodgkin lymphomas" Computers in bio-
% logy and medicine 91 (2017): 135-147.

%Area
globalFeatures.aC = trapz(C);
globalFeatures.aQ = trapz(Q);
globalFeatures.aM = trapz(M);

%Skewness
globalFeatures.skC = skewness(C);
globalFeatures.skQ = skewness(Q);
globalFeatures.skM = skewness(M);

%Maximum point and its scale
[maxC,sigmaC] = max(C);
globalFeatures.maxC = maxC;
globalFeatures.sigmaC = sigmaC;
[maxQ,sigmaQ] = max(Q);
globalFeatures.maxQ = maxQ;
globalFeatures.sigmaQ = sigmaQ;
[maxM,sigmaM] = max(M);
globalFeatures.maxM = maxM;
globalFeatures.sigmaM = sigmaM;
half = ceil(length(C)/2);

%Area ratio
globalFeatures.arC = trapz(C(half+1:end))/trapz(C(1:half));
globalFeatures.arQ = trapz(Q(half+1:end))/trapz(Q(1:half));
globalFeatures.arM = trapz(M(half+1:end))/trapz(M(1:half));
