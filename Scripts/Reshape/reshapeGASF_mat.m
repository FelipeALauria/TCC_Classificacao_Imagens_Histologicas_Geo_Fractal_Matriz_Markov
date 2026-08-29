function reshapeGASF_mat(source, destination, class_name)
% RESHAPEGASF_MAT - Gera imagens GASF RGB a partir dos .mat com atributos fractais
%
% Entradas:
%   source      - pasta onde estão os .mat
%   destination - pasta onde serão salvos os .png
%   class_name  - nome da classe (ex: 'Benign', 'MCL')
%
% Requer: build_gasf.m no path
%
% Pula imagem cujo .png já existe no destino (retomável).
%
% Exemplo de uso:
%   reshapeGASF_mat('C:\...\CR\Benign', 'C:\...\CR\Benign', 'Benign')

matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf'));
matFiles = matFiles(~contains({matFiles.name}, 'RecPlot'));
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
    imgName = fullfile(destination, strcat(class_name, '_', num2str(n), '_gasf.png'));
    if isfile(imgName)
        toc
        continue
    end

    matName = fullfile(source, matFiles(i).name);
    data = load(matName);
    disp(['Carregando: ', matName])

    chessSignal = [data.Chessp(:); data.Chessg(:); data.Chessh(:); data.ChessLAC(:); data.Chessnn(:)];
    euclSignal  = [data.Euclp(:);  data.Euclg(:);  data.Euclh(:);  data.EuclLAC(:);  data.Euclnn(:)];
    manhSignal  = [data.Manhp(:);  data.Manhg(:);  data.Manhh(:);  data.ManhLAC(:);  data.Manhnn(:)];

    % Gera um canal para cada distância e empilha como RGB
    chessChannel = (build_gasf(chessSignal) + 1) / 2;
    euclChannel  = (build_gasf(euclSignal)  + 1) / 2;
    manhChannel  = (build_gasf(manhSignal)  + 1) / 2;

    IMG = cat(3, chessChannel, euclChannel, manhChannel);
    IMG = imresize(IMG, [224 224], 'nearest');

    imwrite(IMG, imgName);
    disp(['Salvo: ', imgName])

    toc
end

disp('GASF gerado com sucesso!')
end
