%%Converte os atributos locais salvos no arquivo CSV com uso do script
%%SaveCSVPercLACDF3Distances em uma imagem de atributos fractais com u-
%%so da técnica recurrence plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%input: arquivo CSV com a primeira linha removida, número de amostras
%%requer o arquivo cerecurr_y.m
%%Guilherme Freire Roberto

function reshapeRecPlot(file, samples)

readFile = file;
features = readFile(369:368+samples,1:300);
features = cell2mat(features);
featuresSplit = zeros(samples,100,3);
featuresSplit(:,:,1) = features(:, 1:100);
featuresSplit(:,:,2) = features(:, 101:200);
featuresSplit(:,:,3) = features(:, 201:300);
newFeatures = zeros(samples,100,3);
newFeatures(:, 1:20, :) = mat2gray(featuresSplit(:, 1:20, :));
newFeatures(:, 21:40, :) = mat2gray(featuresSplit(:, 21:40, :));
newFeatures(:, 41:60, :) = mat2gray(featuresSplit(:, 41:60, :));
newFeatures(:, 61:80, :) = mat2gray(featuresSplit(:, 61:80, :));
newFeatures(:, 81:100, :) = mat2gray(featuresSplit(:, 81:100, :));
for n = 1:samples
signalR =  newFeatures(n, :, 1);
signalG =  newFeatures(n, :, 2);
signalB =  newFeatures(n, :, 3);

signalR = signalR';
signalG = signalG';
signalB = signalB';

r_channel = cerecurr_y(signalR);
g_channel = cerecurr_y(signalG);
b_channel = cerecurr_y(signalB);

IMG = zeros(size(r_channel, 1), size(r_channel, 2), 3);
IMG(:, :, 1) = r_channel;
IMG(:, :, 2) = g_channel;
IMG(:, :, 3) = b_channel;
imwrite(IMG, strcat('F-RecPlot ', num2str(n), '.png'));
end


end