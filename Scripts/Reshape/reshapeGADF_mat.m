function reshapeGADF_mat(source, destination, class_name)
% RESHAPEGADF_MAT - Gera imagens GADF RGB a partir dos .mat com atributos fractais
%
% Entradas:
%   source      - pasta onde estão os .mat
%   destination - pasta onde serão salvos os .png
%   class_name  - nome da classe (ex: 'Benign', 'MCL')
%
% Requer: build_gadf.m no path
%
% Exemplo de uso:
%   reshapeGADF_mat('C:\...\CR\Benign', 'C:\...\CR\Benign', 'Benign')

matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf'));
matFiles = matFiles(~contains({matFiles.name}, 'RecPlot'));
matFiles = matFiles(~contains({matFiles.name}, 'gasf'));
matFiles = matFiles(~contains({matFiles.name}, 'gadf'));
Num_Img = length(matFiles);
disp(['Total de imagens encontradas: ', num2str(Num_Img)])

for n = 1:Num_Img
    tic
    matName = fullfile(source, matFiles(n).name);
    data = load(matName);
    disp(['Carregando: ', matName])

    chessSignal = [data.Chessp(:); data.Chessg(:); data.Chessh(:); data.ChessLAC(:); data.Chessnn(:)];
    euclSignal  = [data.Euclp(:);  data.Euclg(:);  data.Euclh(:);  data.EuclLAC(:);  data.Euclnn(:)];
    manhSignal  = [data.Manhp(:);  data.Manhg(:);  data.Manhh(:);  data.ManhLAC(:);  data.Manhnn(:)];

    % Gera um canal para cada distância e empilha como RGB
    chessChannel = (build_gadf(chessSignal) + 1) / 2;
    euclChannel  = (build_gadf(euclSignal)  + 1) / 2;
    manhChannel  = (build_gadf(manhSignal)  + 1) / 2;

    IMG = cat(3, chessChannel, euclChannel, manhChannel);
    IMG = imresize(IMG, [224 224], 'nearest');

    imgName = fullfile(destination, ...
        strcat(class_name, '_', num2str(n), '_gadf.png'));
    imwrite(IMG, imgName);
    disp(['Salvo: ', imgName])

    toc
end

disp('GADF gerado com sucesso!')
end
