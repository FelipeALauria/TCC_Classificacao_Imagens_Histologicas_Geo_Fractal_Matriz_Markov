function reshapeRecPlot_mat(source, destination, class_name)
% RESHAPERECPLOT_MAT - Gera Recurrence Plots RGB a partir dos .mat com atributos fractais
%
% Entradas:
%   source      - pasta onde estão os .mat
%   destination - pasta onde serão salvos os .png
%   class_name  - nome da classe (ex: 'Benign', 'MCL')
%
% Requer: cerecurr_y.m no path
%
% Pula imagem cujo .png já existe no destino (retomável).
%
% Exemplo de uso:
%   reshapeRecPlot_mat('C:\...\CR\Benign\maxL7', 'C:\...\CR\RecPlot\Benign\maxL7', 'Benign')

matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf'));
matFiles = matFiles(~contains({matFiles.name}, 'gasf'));
matFiles = matFiles(~contains({matFiles.name}, 'gadf'));

% Número real da imagem = número no início do nome do .mat (ex:
% '10_maxL7.mat' -> 10), não a posição na listagem do dir().
numeros = arrayfun(@(f) str2double(regexp(f.name, '^\d+', 'match', 'once')), matFiles);
[numeros, idx] = sort(numeros);
matFiles = matFiles(idx);
Num_Img = length(matFiles);
disp(['Total de imagens encontradas: ', num2str(Num_Img)])

for i = 1:Num_Img
    n = numeros(i);
    tic
    imgName = fullfile(destination, strcat(class_name, '_', num2str(n), '_rp.png'));
    if isfile(imgName)
        toc
        continue
    end

    matName = fullfile(source, matFiles(i).name);
    data = load(matName);
    disp(['Carregando: ', matName])

    % Normaliza cada atributo (p, g, h, LAC, nn) usando a MESMA escala entre
    % as 3 distâncias (Chess/Eucl/Manh), igual ao newFeatures do reshapeRecPlot.m
    [normChessp,   normEuclp,   normManhp]   = normalizeAcrossChannels(data.Chessp,   data.Euclp,   data.Manhp);
    [normChessg,   normEuclg,   normManhg]   = normalizeAcrossChannels(data.Chessg,   data.Euclg,   data.Manhg);
    [normChessh,   normEuclh,   normManhh]   = normalizeAcrossChannels(data.Chessh,   data.Euclh,   data.Manhh);
    [normChessLAC, normEuclLAC, normManhLAC] = normalizeAcrossChannels(data.ChessLAC, data.EuclLAC, data.ManhLAC);
    [normChessnn,  normEuclnn,  normManhnn]  = normalizeAcrossChannels(data.Chessnn,  data.Euclnn,  data.Manhnn);

    % Mesmo agrupamento de atributos usado no MTF/GASF/GADF, um canal por
    % distância: R = Chessboard, G = Euclidiana, B = Manhattan
    signalR = [normChessp(:); normChessg(:); normChessh(:); normChessLAC(:); normChessnn(:)];
    signalG = [normEuclp(:);  normEuclg(:);  normEuclh(:);  normEuclLAC(:);  normEuclnn(:)];
    signalB = [normManhp(:);  normManhg(:);  normManhh(:);  normManhLAC(:);  normManhnn(:)];

    % Gera um canal para cada distância e empilha como RGB
    r_channel = mat2gray(cerecurr_y(signalR));
    g_channel = mat2gray(cerecurr_y(signalG));
    b_channel = mat2gray(cerecurr_y(signalB));

    IMG        = zeros(size(r_channel,1), size(r_channel,2), 3);
    IMG(:,:,1) = r_channel;
    IMG(:,:,2) = g_channel;
    IMG(:,:,3) = b_channel;

    imwrite(IMG, imgName);
    disp(['Salvo: ', imgName])

    toc
end

disp('Recurrence Plots gerados com sucesso!')
end

function [normA, normB, normC] = normalizeAcrossChannels(a, b, c)
% Normaliza os três vetores (um por distância, mesmo atributo) para [0,1]
% usando o min/max dos três combinados - igual ao mat2gray aplicado ao
% bloco de atributo nas 3 dimensões (amostras x atributo x canal) do
% newFeatures em reshapeRecPlot.m, só que aqui por imagem.
    a = a(:); b = b(:); c = c(:);
    combined = [a; b; c];
    mn = min(combined);
    mx = max(combined);

    if mx == mn
        normA = zeros(size(a));
        normB = zeros(size(b));
        normC = zeros(size(c));
    else
        normA = (a - mn) / (mx - mn);
        normB = (b - mn) / (mx - mn);
        normC = (c - mn) / (mx - mn);
    end
end
