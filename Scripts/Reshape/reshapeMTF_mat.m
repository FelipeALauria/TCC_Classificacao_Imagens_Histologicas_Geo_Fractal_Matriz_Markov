function reshapeMTF_mat(source, destination, class_name, maxL, Q_list)
% RESHAPEMTF_MAT - Gera imagens MTF RGB (uma por Q) a partir dos .mat com atributos fractais
%
% Entradas:
%   source      - pasta onde estão os .mat (<n>_maxL<N>.mat)
%   destination - pasta onde serão salvos os .png
%   class_name  - nome da classe (mantido por padrão com os outros reshapes, não usado no nome do arquivo)
%   maxL        - maxL usado na extração fractal (define o N do nome do arquivo, sem precisar carregar o .mat)
%   Q_list      - (opcional) níveis de quantização, default [8 16 32 64]
%
% Requer: build_mtf.m no path
%
% Pula (classe, imagem, Q) cujo .png já existe no destino, sem nem carregar
% o .mat correspondente (retomável).
%
% Exemplo de uso:
%   reshapeMTF_mat('C:\...\CR\Benign\maxL7', 'C:\...\CR\MTF\Benign\maxL7', 'Benign', 7)

    if nargin < 5
        Q_list = [8, 16, 32, 64];
    end

    % Tamanho do sinal concatenado (p+g+h+LAC+nn), fixo pro maxL (não depende da imagem)
    m = 5 * length(3:2:maxL);

    matFiles = dir(fullfile(source, '*.mat'));

    isValid  = ~cellfun(@isempty, regexp({matFiles.name}, '^\d+_maxL\d+\.mat$'));
    matFiles = matFiles(isValid);

    % Número real da imagem = número no início do nome do .mat (ex:
    % '10_maxL7.mat' -> 10), não a posição na listagem do dir().
    numeros  = arrayfun(@(f) str2double(regexp(f.name, '^\d+', 'match', 'once')), matFiles);
    [numeros, idx] = sort(numeros);
    matFiles  = matFiles(idx);
    Num_Img  = length(matFiles);
    disp(['Total de arquivos encontrados: ', num2str(Num_Img)]);

    for i = 1:Num_Img
        n = numeros(i);
        tic

        nomeImagem = @(q) fullfile(destination, strcat(num2str(n), '_MTF_Q', num2str(q), '_N', num2str(m), '.png'));
        qsFaltando = Q_list(~arrayfun(@(q) isfile(nomeImagem(q)), Q_list));
        if isempty(qsFaltando)
            toc
            continue
        end

        matName = fullfile(source, matFiles(i).name);
        data    = load(matName);
        disp(['Carregando: ', matFiles(i).name]);

        signalR = [data.Chessp(:); data.Chessg(:); data.Chessh(:); data.ChessLAC(:); data.Chessnn(:)];
        signalG = [data.Euclp(:);  data.Euclg(:);  data.Euclh(:);  data.EuclLAC(:);  data.Euclnn(:)];
        signalB = [data.Manhp(:);  data.Manhg(:);  data.Manhh(:);  data.ManhLAC(:);  data.Manhnn(:)];

        % Defensivo: se algum dia p/g/h/LAC/nn tiverem tamanhos diferentes
        % entre distâncias, completa com zero até o maior (na prática, com
        % maxL fixo os 3 já saem do mesmo tamanho -- daí o `m` acima bater).
        maxLen  = max([length(signalR), length(signalG), length(signalB)]);
        signalR(end+1:maxLen) = 0;
        signalG(end+1:maxLen) = 0;
        signalB(end+1:maxLen) = 0;

        signalR = mat2gray(signalR(:));
        signalG = mat2gray(signalG(:));
        signalB = mat2gray(signalB(:));

        for q = qsFaltando
            r_channel = build_mtf(signalR, q);
            g_channel = build_mtf(signalG, q);
            b_channel = build_mtf(signalB, q);

            IMG        = zeros(size(r_channel,1), size(r_channel,2), 3);
            IMG(:,:,1) = r_channel;
            IMG(:,:,2) = g_channel;
            IMG(:,:,3) = b_channel;

            IMG_resized = imresize(im2uint8(IMG), [224 224], 'nearest');
            imwrite(IMG_resized, nomeImagem(q));
            disp(['Salvo: ', nomeImagem(q)]);
        end

        toc
    end

    disp('MTFs gerados com sucesso!');
end
