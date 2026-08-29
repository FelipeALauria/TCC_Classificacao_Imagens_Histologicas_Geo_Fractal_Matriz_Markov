% VALIDARFRACTAIS - Teste de regressão: compara a extração fractal nova
% (percChess/percEucl/percManh + pmr* uma vez em maxL=41, fatiados pros
% demais) contra os .mat que já estão salvos no CR (gerados pelo
% FractalFeatures.m antigo, um maxL por vez, um por um).
%
% Roda em poucas imagens só (rápido) e mostra a maior diferença absoluta
% por campo -- deve dar ~0 (diferença de ponto flutuante) se a lógica de
% fatiamento estiver certa. Rodar isso antes de soltar o orquestrador em
% lote nos outros datasets.
%
% Como executar: ajustar BASE_DIR/DATASET/CLASSE/IMAGENS_TESTE se
% necessário (tem que ser um dataset/classe que já tem os .mat antigos
% pros MAXL_LIST inteiros, hoje só o CR), e rodar (F5). Precisa de
% Scripts/Fractais no path.

BASE_DIR = 'C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas';
DATASET = 'CR';
CLASSE = 'Benign';
IMAGENS_TESTE = [1, 2]; % números reais das imagens a comparar (ex: 'Benign (10).png' -> 10)
MAXL_LIST = [7, 11, 15, 25, 41];

classSource = fullfile(BASE_DIR, DATASET, CLASSE);

tempDest = fullfile(tempdir, 'validacao_fractais_in');
if isfolder(tempDest)
    rmdir(tempDest, 's');
end
mkdir(tempDest);

% Copia só as imagens de teste (identificadas pelo número real no nome,
% ex: 'Benign (10).png' -> 10, não pela posição em dir()+sort()) pra uma
% pasta temporária.
imgFiles = dir(fullfile(classSource, '*.png'));
nomes = {imgFiles.name};
tokens = regexp(nomes, '\((\d+)\)', 'tokens', 'once');
semNumero = cellfun(@isempty, tokens);
imgFiles = imgFiles(~semNumero);
tokens = tokens(~semNumero);
numerosImg = cellfun(@(t) str2double(t{1}), tokens);

for n = IMAGENS_TESTE
    match = find(numerosImg == n, 1);
    if isempty(match)
        error('validarFractais:imagemNaoEncontrada', 'Imagem número %d não encontrada em %s', n, classSource);
    end
    copyfile(fullfile(classSource, imgFiles(match).name), tempDest);
end

destinations = struct();
for m = 1:length(MAXL_LIST)
    maxLTag = sprintf('maxL%d', MAXL_LIST(m));
    destN = fullfile(tempdir, 'validacao_fractais_out', maxLTag);
    if isfolder(destN)
        rmdir(destN, 's');
    end
    mkdir(destN);
    destinations.(maxLTag) = destN;
end

FractalFeatures(tempDest, destinations, MAXL_LIST);

camposComparar = {'Chessp','Chessg','Chessh','ChessLAC','Chessnn','ChessFD', ...
    'Euclp','Euclg','Euclh','EuclLAC','Euclnn','EuclFD', ...
    'Manhp','Manhg','Manhh','ManhLAC','Manhnn','ManhFD'};

for i = 1:length(IMAGENS_TESTE)
    n = IMAGENS_TESTE(i);
    fprintf('\n--- Imagem %d ---\n', n);
    for m = 1:length(MAXL_LIST)
        maxL = MAXL_LIST(m);
        maxLTag = sprintf('maxL%d', maxL);

        antigo = load(fullfile(classSource, maxLTag, strcat(num2str(n), '_maxL', num2str(maxL), '.mat')));
        novo   = load(fullfile(destinations.(maxLTag), strcat(num2str(n), '_maxL', num2str(maxL), '.mat')));

        fprintf('  maxL=%d:\n', maxL);
        for f = 1:length(camposComparar)
            campo = camposComparar{f};
            if ~isfield(antigo, campo) || ~isfield(novo, campo)
                fprintf('    %s: campo ausente em um dos dois .mat\n', campo);
                continue
            end
            diferenca = max(abs(antigo.(campo)(:) - novo.(campo)(:)));
            fprintf('    %s: diferença máxima = %g\n', campo, diferenca);
        end
    end
end

disp(' ')
disp('Validação concluída -- diferenças devem ser ~0 (ponto flutuante).')
