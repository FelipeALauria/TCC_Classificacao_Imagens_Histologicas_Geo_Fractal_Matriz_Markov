function FractalFeatures(source, destinations, maxLList)
% FRACTALFEATURES - Extrai atributos fractais (percolação, lacunaridade, FD)
% nas 3 distâncias (Chessboard/Euclidiana/Manhattan) pra cada imagem de `source`.
%
% Calcula a parte cara (percolação + matriz de probabilidade) UMA VEZ por
% imagem, no maior maxL da lista, e fatia o resultado pros maxL menores --
% cada coluna de percChess/percEucl/percManh/pmrChess/pmrEucl/pmrManh já é
% independente do maxL usado na chamada, só os agregados finais (trapz/
% skewness/max/FD) dependem de até onde o r vai, e esses são recalculados
% por fatia (barato). Ver percStatsFromCurve.m/lacStatsFromCurve.m/
% fractalDimensionFromProbMatrix.m.
%
% Entradas:
%   source       - pasta com as imagens originais da classe (*.png)
%   destinations - struct com um campo por maxL (ex: destinations.maxL7 = '...')
%                  apontando pra pasta onde salvar os .mat desse maxL
%   maxLList     - vetor com os maxL desejados, ex: [7 11 15 25 41]
%
% Requer no path: percChess, percEucl, percManh, pmrChess, pmrEucl, pmrManh,
% lacunarity, calcN, percStatsFromCurve, lacStatsFromCurve,
% fractalDimensionFromProbMatrix
%
% Pula imagem/maxL cujo .mat já existe no destino (retomável).
%
% Exemplo de uso:
%   destinations.maxL7  = 'C:\...\CR\Benign\maxL7';
%   destinations.maxL11 = 'C:\...\CR\Benign\maxL11';
%   FractalFeatures('C:\...\CR\Benign', destinations, [7 11]);

    imgFiles = dir(fullfile(source, '*.png'));

    % As imagens têm que ser processadas em ordem NUMÉRICA crescente
    % (1,2,3,4,...), não na ordem alfabética que dir()+sort(nome) dá --
    % "(10)" vem antes de "(2)" alfabeticamente, o que fazia a imagem 10
    % virar '2_maxL7.mat' (2ª posição na lista alfabética). O número vem do
    % nome do arquivo (ex: 'Benign (10).png' -> 10) e é usado tanto pra
    % ordenar quanto pro nome do .mat de saída (classe já vem embutida em
    % `destinations`/pasta de origem, o número identifica a imagem dentro
    % da classe).
    nomes = {imgFiles.name};
    tokens = regexp(nomes, '\((\d+)\)', 'tokens', 'once');
    semNumero = cellfun(@isempty, tokens);
    if any(semNumero)
        warning('FractalFeatures:semNumero', 'Ignorando %d arquivo(s) sem número entre parênteses em %s: %s', ...
            sum(semNumero), source, strjoin(nomes(semNumero), ', '));
        imgFiles = imgFiles(~semNumero);
        tokens = tokens(~semNumero);
    end
    numeros = cellfun(@(t) str2double(t{1}), tokens);
    [numeros, idx] = sort(numeros);
    imgFiles = imgFiles(idx);
    Num_Img = length(imgFiles);
    disp(['Total de imagens encontradas: ', num2str(Num_Img)])

    maxLFull = max(maxLList);
    rFull = 3:2:maxLFull;

    for i = 1:Num_Img
        n = numeros(i); % número real da imagem (usado no nome do .mat)
        outNames = arrayfun(@(maxL) fullfile(destinations.(sprintf('maxL%d', maxL)), ...
            strcat(num2str(n), '_maxL', num2str(maxL), '.mat')), maxLList, 'UniformOutput', false);
        if all(cellfun(@isfile, outNames))
            continue
        end

        tic
        fullname = fullfile(source, imgFiles(i).name);
        PIC = imread(fullname);
        disp(['Processando: ', imgFiles(i).name])

        % Parte cara (contagem de caixas), rodada uma única vez no maxL cheio
        [ChessMaxClusterIndex_, ChessMaxPercIndex_, ChessMaxMaxClusterIndex_, ChessAreaRatioMaxCluster_, ChessMaxMaxCluster_, ChessSkewnessMaxCluster_, ChessAreaMaxCluster_, ChessAreaRatioCluster_, ChessAreaRatioPerc_, ChessMaxCluster_, ChessMaxPerc_, ChessSkewnessCluster_, ChessSkewnessPerc_, ChessAreaPerc_, ChessAreaCluster_, ChessP, ChessG, ChessH] = percChess(PIC, maxLFull); %#ok<ASGLU>
        [EuclMaxClusterIndex_,  EuclMaxPercIndex_,  EuclMaxMaxClusterIndex_,  EuclAreaRatioMaxCluster_,  EuclMaxMaxCluster_,  EuclSkewnessMaxCluster_,  EuclAreaMaxCluster_,  EuclAreaRatioCluster_,  EuclAreaRatioPerc_,  EuclMaxCluster_,  EuclMaxPerc_,  EuclSkewnessCluster_,  EuclSkewnessPerc_,  EuclAreaPerc_,  EuclAreaCluster_,  EuclP,  EuclG,  EuclH ] = percEucl(PIC, maxLFull); %#ok<ASGLU>
        [ManhMaxClusterIndex_,  ManhMaxPercIndex_,  ManhMaxMaxClusterIndex_,  ManhAreaRatioMaxCluster_,  ManhMaxMaxCluster_,  ManhSkewnessMaxCluster_,  ManhAreaMaxCluster_,  ManhAreaRatioCluster_,  ManhAreaRatioPerc_,  ManhMaxCluster_,  ManhMaxPerc_,  ManhSkewnessCluster_,  ManhSkewnessPerc_,  ManhAreaPerc_,  ManhAreaCluster_,  ManhP,  ManhG,  ManhH ] = percManh(PIC, maxLFull); %#ok<ASGLU>

        ProbMatrixChess = pmrChess(PIC, maxLFull);
        ProbMatrixEucl  = pmrEucl(PIC, maxLFull);
        ProbMatrixManh  = pmrManh(PIC, maxLFull);

        for m = 1:length(maxLList)
            maxL = maxLList(m);
            outName = outNames{m};
            if isfile(outName)
                continue
            end

            k = length(3:2:maxL);
            r = rFull(1:k);

            Chessp = ChessP(1:k); Chessg = ChessG(1:k); Chessh = ChessH(1:k);
            Euclp  = EuclP(1:k);  Euclg  = EuclG(1:k);  Euclh  = EuclH(1:k);
            Manhp  = ManhP(1:k);  Manhg  = ManhG(1:k);  Manhh  = ManhH(1:k);

            [ChessMaxClusterIndex, ChessMaxPercIndex, ChessMaxMaxClusterIndex, ChessAreaRatioMaxCluster, ChessMaxMaxCluster, ChessSkewnessMaxCluster, ChessAreaMaxCluster, ChessAreaRatioCluster, ChessAreaRatioPerc, ChessMaxCluster, ChessMaxPerc, ChessSkewnessCluster, ChessSkewnessPerc, ChessAreaPerc, ChessAreaCluster] = percStatsFromCurve(Chessp, Chessg, Chessh);
            [EuclMaxClusterIndex,  EuclMaxPercIndex,  EuclMaxMaxClusterIndex,  EuclAreaRatioMaxCluster,  EuclMaxMaxCluster,  EuclSkewnessMaxCluster,  EuclAreaMaxCluster,  EuclAreaRatioCluster,  EuclAreaRatioPerc,  EuclMaxCluster,  EuclMaxPerc,  EuclSkewnessCluster,  EuclSkewnessPerc,  EuclAreaPerc,  EuclAreaCluster ] = percStatsFromCurve(Euclp, Euclg, Euclh);
            [ManhMaxClusterIndex,  ManhMaxPercIndex,  ManhMaxMaxClusterIndex,  ManhAreaRatioMaxCluster,  ManhMaxMaxCluster,  ManhSkewnessMaxCluster,  ManhAreaMaxCluster,  ManhAreaRatioCluster,  ManhAreaRatioPerc,  ManhMaxCluster,  ManhMaxPerc,  ManhSkewnessCluster,  ManhSkewnessPerc,  ManhAreaPerc,  ManhAreaCluster ] = percStatsFromCurve(Manhp, Manhg, Manhh);

            ChessLAC = lacunarity(ProbMatrixChess(:, 1:k));
            EuclLAC  = lacunarity(ProbMatrixEucl(:, 1:k));
            ManhLAC  = lacunarity(ProbMatrixManh(:, 1:k));

            [ChessAreaLAC, ChessSkewnessLAC, ChessAreaRatioLAC, ChessMaxLAC, ChessMaxLACIndex] = lacStatsFromCurve(ChessLAC);
            [EuclAreaLAC,  EuclSkewnessLAC,  EuclAreaRatioLAC,  EuclMaxLAC,  EuclMaxLACIndex ] = lacStatsFromCurve(EuclLAC);
            [ManhAreaLAC,  ManhSkewnessLAC,  ManhAreaRatioLAC,  ManhMaxLAC,  ManhMaxLACIndex ] = lacStatsFromCurve(ManhLAC);

            [ChessFD, Chessnn] = fractalDimensionFromProbMatrix(ProbMatrixChess(:, 1:k), r);
            [EuclFD,  Euclnn ] = fractalDimensionFromProbMatrix(ProbMatrixEucl(:, 1:k),  r);
            [ManhFD,  Manhnn ] = fractalDimensionFromProbMatrix(ProbMatrixManh(:, 1:k),  r);

            save(outName, ...
                'Chessp', 'Chessg', 'Chessh', 'ChessLAC', 'Chessnn', 'ChessFD', ...
                'ChessMaxClusterIndex', 'ChessMaxPercIndex', 'ChessMaxMaxClusterIndex', 'ChessAreaRatioMaxCluster', 'ChessMaxMaxCluster', 'ChessSkewnessMaxCluster', 'ChessAreaMaxCluster', 'ChessAreaRatioCluster', 'ChessAreaRatioPerc', 'ChessMaxCluster', 'ChessMaxPerc', 'ChessSkewnessCluster', 'ChessSkewnessPerc', 'ChessAreaPerc', 'ChessAreaCluster', ...
                'ChessAreaLAC', 'ChessSkewnessLAC', 'ChessAreaRatioLAC', 'ChessMaxLAC', 'ChessMaxLACIndex', ...
                'Euclp', 'Euclg', 'Euclh', 'EuclLAC', 'Euclnn', 'EuclFD', ...
                'EuclMaxClusterIndex', 'EuclMaxPercIndex', 'EuclMaxMaxClusterIndex', 'EuclAreaRatioMaxCluster', 'EuclMaxMaxCluster', 'EuclSkewnessMaxCluster', 'EuclAreaMaxCluster', 'EuclAreaRatioCluster', 'EuclAreaRatioPerc', 'EuclMaxCluster', 'EuclMaxPerc', 'EuclSkewnessCluster', 'EuclSkewnessPerc', 'EuclAreaPerc', 'EuclAreaCluster', ...
                'EuclAreaLAC', 'EuclSkewnessLAC', 'EuclAreaRatioLAC', 'EuclMaxLAC', 'EuclMaxLACIndex', ...
                'Manhp', 'Manhg', 'Manhh', 'ManhLAC', 'Manhnn', 'ManhFD', ...
                'ManhMaxClusterIndex', 'ManhMaxPercIndex', 'ManhMaxMaxClusterIndex', 'ManhAreaRatioMaxCluster', 'ManhMaxMaxCluster', 'ManhSkewnessMaxCluster', 'ManhAreaMaxCluster', 'ManhAreaRatioCluster', 'ManhAreaRatioPerc', 'ManhMaxCluster', 'ManhMaxPerc', 'ManhSkewnessCluster', 'ManhSkewnessPerc', 'ManhAreaPerc', 'ManhAreaCluster', ...
                'ManhAreaLAC', 'ManhSkewnessLAC', 'ManhAreaRatioLAC', 'ManhMaxLAC', 'ManhMaxLACIndex');
            disp(['Salvo: ', outName])
        end
        toc
    end

    disp('Atributos fractais gerados com sucesso!')
end
