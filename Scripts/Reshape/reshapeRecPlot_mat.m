function reshapeRecPlot_mat(source, destination, class_name)
% RESHAPERECPLOT_MAT - Gera Recurrence Plots a partir dos .mat com atributos fractais
%
% Entradas:
%   source      - pasta onde estão os .mat
%   destination - pasta onde serão salvos os .png
%   class_name  - nome da classe (ex: 'Benign', 'MCL')
%
% Exemplo de uso:
%   reshapeRecPlot_mat('C:\...\CR\Benign', 'C:\...\CR\Benign', 'Benign')

addpath('C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Scripts\Reshape');

% Conta os .mat automaticamente (ignora os MTF já gerados)
matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf')); % exclui os MTF_*.mat
Num_Img = length(matFiles);
disp(['Total de imagens encontradas: ', num2str(Num_Img)])

for n = 1:Num_Img
    tic
    % Carrega o .mat
    matName = fullfile(source, matFiles(n).name);
    data = load(matName);
    disp(['Carregando: ', matName])

    % Monta o signal completo
    signal = [data.ChessLAC, data.EuclLAC, data.ManhLAC, ...
        data.ChessFD,  data.EuclFD,  data.ManhFD];
    signal = signal(:); % garante vetor coluna

    % Divide o signal em 3 canais RGB
    len = length(signal);
    third = floor(len / 3);

    signalR = mat2gray(signal(1:third));
    signalG = mat2gray(signal(third+1:2*third));
    signalB = mat2gray(signal(2*third+1:3*third));

    % Gera o Recurrence Plot para cada canal
    r_channel = cerecurr_y(signalR);
    g_channel = cerecurr_y(signalG);
    b_channel = cerecurr_y(signalB);

    % Normaliza cada canal entre 0 e 1
    r_channel = mat2gray(r_channel);
    g_channel = mat2gray(g_channel);
    b_channel = mat2gray(b_channel);

    % Monta imagem RGB
    IMG = zeros(size(r_channel, 1), size(r_channel, 2), 3);
    IMG(:, :, 1) = r_channel;
    IMG(:, :, 2) = g_channel;
    IMG(:, :, 3) = b_channel;

    % Salva como PNG
    % Ex: Benign_1_rp.png, Benign_2_rp.png...
    imgName = fullfile(destination, ...
        strcat(class_name, '_', num2str(n), '_rp.png'));
    imwrite(IMG, imgName);
    disp(['Salvo: ', imgName])

    toc
end

disp('Recurrence Plots gerados com sucesso!')
end