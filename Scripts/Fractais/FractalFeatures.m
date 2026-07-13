%%Script for obtaining PERC, LAC and FD global and local features using Chessboard, Manhattan and Euclidian distances
%%Requirements:
%
% pmrChess.m
% pmrEucl.m
% pmrManh.m
% percChess.m
% percEucl.m
% percManh.m
% lacunarity.m

%Max L value
maxL = 7;

%Number of images in dataset (change accordingly)
Num_Img = 150;

%\UCSBB (1).png%
%%
%Classe 1
source='C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Datasets\Imagens Histológicas\LG\2'; %Path where the images are stored
destination= source; %Path where the .mat file will be saved (can be the same as source)
disp(['Salvando: ', source])
for n=1:Num_Img %número de imagens da classe
    tic
original = strcat('Class 2 - (', num2str(n), ').png'); %File name. Must be in the format: 'File_number from 1 to N_.png'. If your images are named differently, rename them or than rewrite this line accordingly
%.png format is not mandatory. If your images are in a different format, just rewrite the line above to match your format

fullname=fullfile(source,original);
PIC=imread(fullname);
disp(['Algoritmos'])
%PERC
disp("PercChess")
%[ChessMaxClusterIndex, ChessMaxPercIndex, ChessMaxMaxClusterIndex, ChessAreaRatioMaxCluster, ChessMaxMaxCluster, ChessSkewnessMaxCluster, ChessAreaMaxCluster, ChessAreaRatioCluster, ChessAreaRatioPerc, ChessMaxCluster, ChessMaxPerc, ChessSkewnessCluster, ChessSkewnessPerc, ChessAreaPerc, ChessAreaCluster, Chessp, Chessg, Chessh] = percChess(PIC,maxL);
disp("PercEucl")
%[EuclMaxClusterIndex, EuclMaxPercIndex, EuclMaxMaxClusterIndex, EuclAreaRatioMaxCluster, EuclMaxMaxCluster, EuclSkewnessMaxCluster, EuclAreaMaxCluster, EuclAreaRatioCluster, EuclAreaRatioPerc, EuclMaxCluster, EuclMaxPerc, EuclSkewnessCluster, EuclSkewnessPerc, EuclAreaPerc, EuclAreaCluster, Euclp, Euclg, Euclh] = percEucl(PIC,maxL);
disp("PercManh")
%[ManhMaxClusterIndex, ManhMaxPercIndex, ManhMaxMaxClusterIndex, ManhAreaRatioMaxCluster, ManhMaxMaxCluster, ManhSkewnessMaxCluster, ManhAreaMaxCluster, ManhAreaRatioCluster, ManhAreaRatioPerc, ManhMaxCluster, ManhMaxPerc, ManhSkewnessCluster, ManhSkewnessPerc, ManhAreaPerc, ManhAreaCluster, Manhp, Manhg, Manhh] = percManh(PIC,maxL);

disp("LAC e DF")
%Chessboard LAC FD
ProbMatrix = pmrChess(PIC, maxL);
ChessLAC = lacunarity(ProbMatrix);
r = 3:2:maxL;
ChessAreaLAC = trapz(ChessLAC);
ChessSkewnessLAC = skewness(ChessLAC);
half = ceil(length(ChessLAC)/2);
ChessAreaRatioLAC = trapz(ChessLAC(half+1:end))/trapz(ChessLAC(1:half));
[ChessMaxLAC, ChessMaxLACIndex] = max(ChessLAC);
Chessnn = calcN(ProbMatrix);
x = log(r);
y = -log(Chessnn);
X = x.';
Y = y.';
mdl = fitlm(X, Y, 'RobustOpts', 'on');
ChessFD = mdl.Coefficients.Estimate(2);

disp(['prob_matrix'])
%Euclidian LAC FD
ProbMatrix = pmrEucl(PIC, maxL);
EuclLAC = lacunarity(ProbMatrix);
r = 3:2:maxL;
EuclAreaLAC = trapz(EuclLAC);
EuclSkewnessLAC = skewness(EuclLAC);
half = ceil(length(EuclLAC)/2);
EuclAreaRatioLAC = trapz(EuclLAC(half+1:end))/trapz(EuclLAC(1:half));
[EuclMaxLAC, EuclMaxLACIndex] = max(EuclLAC);
Euclnn = calcN(ProbMatrix);
x = log(r);
y = -log(Euclnn);
X = x.';
Y = y.';
mdl = fitlm(X, Y, 'RobustOpts', 'on');
EuclFD = mdl.Coefficients.Estimate(2);

%Manhattan LAC FD
ProbMatrix = pmrManh(PIC, maxL);
ManhLAC = lacunarity(ProbMatrix);
r = 3:2:maxL;
ManhAreaLAC = trapz(ManhLAC);
ManhSkewnessLAC = skewness(ManhLAC);
half = ceil(length(ManhLAC)/2);
ManhAreaRatioLAC = trapz(ManhLAC(half+1:end))/trapz(ManhLAC(1:half));
[ManhMaxLAC, ManhMaxLACIndex] = max(ManhLAC);
Manhnn = calcN(ProbMatrix);
x = log(r);
y = -log(Manhnn);
X = x.';
Y = y.';
mdl = fitlm(X, Y, 'RobustOpts', 'on');
ManhFD = mdl.Coefficients.Estimate(2);

clear PIC;     
final=strcat(num2str(n),'.mat');
filename=fullfile(destination,final);
disp(['Salvando: ', filename])
save(filename);
disp('Salvou!!')
toc
end