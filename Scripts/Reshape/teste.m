% ================================================
% Gera Recurrence Plot e salva como .mat
% ================================================

source      = 'C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Datasets\Imagens Histológicas\CR\Benign';
destination = source;

addpath('C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Scripts\Reshape');

% Conta os .mat ignorando os já gerados
matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'RecPlot'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf'));
matFiles = matFiles(~contains({matFiles.name}, 'bits'));
Num_Img  = length(matFiles);

% Ordena numericamente
[~, idx] = sort(arrayfun(@(f) str2double(regexp(f.name, '^\d+', 'match', 'once')), matFiles));
matFiles  = matFiles(idx);

disp(['Total de imagens encontradas: ', num2str(Num_Img)])

for n = 1:Num_Img
    tic
    matName = fullfile(source, matFiles(n).name);
    data    = load(matName);
    disp(['Carregando: ', matName])

    % Monta o signal completo
    signal = [data.ChessLAC, data.EuclLAC, data.ManhLAC, ...
              data.ChessFD,  data.EuclFD,  data.ManhFD];
    signal = signal(:);

    % Gera o RP
    RP = mat2gray(cerecurr_y(signal));

    % Salva como .mat
    rpName = fullfile(destination, strcat(num2str(n), '_RecPlot.mat'));
    save(rpName, 'RP');
    disp(['Salvo: ', rpName])
    toc
end

disp('Concluído!')