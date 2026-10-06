% ORQUESTRADORFRACTAIS - Gera os atributos fractais (5 maxL) e os 4 reshapes
% (GASF, GADF, MTF, RecPlot) pra todos os datasets, no mesmo padrão de
% pastas já usado no CR:
%   <Dataset>/<Classe>/maxL<N>/<n>_maxL<N>.mat              (fractais)
%   <Dataset>/<Canal>/<Classe>/maxL<N>/...                  (reshapes)
%
% Pula dataset/classe/maxL/imagem que já existe no destino -- roda de novo
% quantas vezes precisar sem duplicar trabalho nem reprocessar o que já
% está pronto (hoje só o CR).
%
% Como executar: ajustar BASE_DIR se necessário, garantir Scripts/Fractais
% e Scripts/Reshape no path (addpath), e rodar (F5).

BASE_DIR = 'C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas';
DATASETS = {'CR-SameRes', 'DIS','LA', 'LG', 'NHL', 'UCSB'};
MAXL_LIST = [7, 11, 15, 25, 41, 65];
MTF_Q_LIST = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];
RESHAPES = {'GASF', 'GADF', 'MTF', 'RecPlot'};

for d = 1:length(DATASETS)
    datasetName = DATASETS{d};
    datasetDir = fullfile(BASE_DIR, datasetName);
    fprintf('\n===== Dataset: %s =====\n', datasetName);

    try
        classes = listarClasses(datasetDir, RESHAPES);

        for c = 1:length(classes)
            className = classes{c};
            classSource = fullfile(datasetDir, className);
            fprintf('--- Classe: %s / %s ---\n', datasetName, className);

            % Pastas de destino dos atributos fractais, uma por maxL
            destinations = struct();
            for m = 1:length(MAXL_LIST)
                maxLTag = sprintf('maxL%d', MAXL_LIST(m));
                destN = fullfile(classSource, maxLTag);
                if ~isfolder(destN)
                    mkdir(destN);
                end
                destinations.(maxLTag) = destN;
            end

            try
                FractalFeatures(classSource, destinations, MAXL_LIST);
            catch fractalErr
                fprintf(2, 'ERRO nos fractais de %s/%s: %s\n', datasetName, className, fractalErr.message);
                continue % sem fractais não dá pra rodar os reshapes dessa classe
            end

            % Reshapes, um maxL por vez
            for m = 1:length(MAXL_LIST)
                maxL = MAXL_LIST(m);
                maxLTag = sprintf('maxL%d', maxL);
                fractalSource = destinations.(maxLTag);

                gasfDest = fullfile(datasetDir, 'GASF', className, maxLTag);
                gadfDest = fullfile(datasetDir, 'GADF', className, maxLTag);
                mtfDest  = fullfile(datasetDir, 'MTF',  className, maxLTag);
                rpDest   = fullfile(datasetDir, 'RecPlot', className, maxLTag);
                pastasDestino = {gasfDest, gadfDest, mtfDest, rpDest};
                for p = 1:length(pastasDestino)
                    if ~isfolder(pastasDestino{p})
                        mkdir(pastasDestino{p});
                    end
                end

                try
                    reshapeGASF_mat(fractalSource, gasfDest, className);
                catch e
                    fprintf(2, 'ERRO GASF em %s/%s/%s: %s\n', datasetName, className, maxLTag, e.message);
                end

                try
                    reshapeGADF_mat(fractalSource, gadfDest, className);
                catch e
                    fprintf(2, 'ERRO GADF em %s/%s/%s: %s\n', datasetName, className, maxLTag, e.message);
                end

                try
                    reshapeMTF_mat(fractalSource, mtfDest, className, maxL, MTF_Q_LIST);
                catch e
                    fprintf(2, 'ERRO MTF em %s/%s/%s: %s\n', datasetName, className, maxLTag, e.message);
                end

                try
                    reshapeRecPlot_mat(fractalSource, rpDest, className);
                catch e
                    fprintf(2, 'ERRO RecPlot em %s/%s/%s: %s\n', datasetName, className, maxLTag, e.message);
                end
            end
        end
    catch datasetErr
        fprintf(2, 'ERRO no dataset %s: %s\n', datasetName, datasetErr.message);
        continue
    end
end

disp(' ')
disp('Orquestrador concluído!')

function classes = listarClasses(datasetDir, reshapeNames)
% LISTARCLASSES - Subpastas de um dataset que são classes (exclui as
% pastas de reshape -- GASF/GADF/MTF/RecPlot -- que ficam no mesmo nível).
    d = dir(datasetDir);
    d = d([d.isdir]);
    names = {d.name};
    names = names(~startsWith(names, '.'));
    classes = names(~ismember(names, reshapeNames));
end
