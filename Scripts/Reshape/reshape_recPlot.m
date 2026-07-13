% ================================================
% Gera Recurrence Plot e salva como .mat
% ================================================

source      = 'C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Datasets\Imagens Histológicas\UCSB\Malignant';
destination = source;

addpath('C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Scripts\Reshape');

% Conta os .mat ignorando os já gerados
matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'rp'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf'));
Num_Img  = length(matFiles);
disp(['Total de imagens encontradas: ', num2str(Num_Img)])

for n = 1:Num_Img
    tic
    matName = fullfile(source, matFiles(n).name);
    data    = load(matName);
    disp(['Carregando: ', matName]) 

    % Monta o signal
    signal  = [data.ChessLAC, data.EuclLAC, data.ManhLAC, ...
        data.ChessFD,  data.EuclFD,  data.ManhFD];
    signal  = signal(:);

    % Divide em 3 canais
    third   = floor(length(signal) / 3);
    signalR = mat2gray(signal(1:third));
    signalG = mat2gray(signal(third+1:2*third));
    signalB = mat2gray(signal(2*third+1:3*third));

    % Gera o RP para cada canal
    r_channel = mat2gray(cerecurr_y(signalR));
    g_channel = mat2gray(cerecurr_y(signalG));
    b_channel = mat2gray(cerecurr_y(signalB));

    % Monta matriz RGB
    RP = zeros(size(r_channel, 1), size(r_channel, 2), 3);
    RP(:, :, 1) = r_channel;
    RP(:, :, 2) = g_channel;
    RP(:, :, 3) = b_channel;

    % Salva como .mat
    rpName = fullfile(destination, strcat(num2str(n), '_RecPlot.mat'));
    save(rpName, 'RP');
    disp(['Salvo: ', rpName])
    toc
end

disp('Concluído!')