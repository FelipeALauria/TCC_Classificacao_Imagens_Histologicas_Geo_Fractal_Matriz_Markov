source = 'C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Datasets\Imagens Histológicas\\LG\2';
destination = source;

% Conta os .mat automaticamente
matFiles = dir(fullfile(source, '*.mat'));
Num_Img = length(matFiles);
disp(['Total de arquivos encontrados: ', num2str(Num_Img)])

Q_list = [8, 16, 32, 64];

for n = 1:Num_Img
    tic
    % Carrega o .mat da imagem n
    matName = fullfile(source, matFiles(n).name);
    data = load(matName);
    disp(['Carregando: ', matName])

    % Monta o sinal com os atributos fractais
    signal = [data.ChessLAC, data.EuclLAC, data.ManhLAC, ...
              data.ChessFD,  data.EuclFD,  data.ManhFD];

    for q = Q_list
        % Gera a MTF pro Q atual
        F = build_mtf(signal, q);

        % Ex: benign_1_mtf_8bits.mat, benign_1_mtf_16bits.mat...
        mtfName = fullfile(destination, ...
            strcat(num2str(n), '_mtf_', num2str(q), 'bits.mat'));

        save(mtfName, 'F');
        disp(['Salvo: ', mtfName])
    end

    toc
end