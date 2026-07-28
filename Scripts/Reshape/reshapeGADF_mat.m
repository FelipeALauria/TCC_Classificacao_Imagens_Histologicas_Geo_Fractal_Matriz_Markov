function reshapeGADF_mat(source, destination, class_name)
% RESHAPEGADF_MAT - Gera imagens GADF (mono-canal) a partir dos .mat com atributos fractais
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

    % Monta o signal completo (mesmo padrão do MTF)
    signal = [data.ChessLAC, data.EuclLAC, data.ManhLAC, ...
        data.ChessFD,  data.EuclFD,  data.ManhFD];
    signal = signal(:);

    % Gera o GADF
    F = build_gadf(signal);

    % GADF varia em [-1, 1]; converte para [0, 1] antes de salvar
    IMG = (F + 1) / 2;

    imgName = fullfile(destination, ...
        strcat(class_name, '_', num2str(n), '_gadf.png'));
    imwrite(IMG, imgName);
    disp(['Salvo: ', imgName])

    toc
end

disp('GADF gerado com sucesso!')
end
