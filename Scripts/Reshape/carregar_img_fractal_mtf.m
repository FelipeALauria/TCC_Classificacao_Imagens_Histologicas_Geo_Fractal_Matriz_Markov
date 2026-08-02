% ================================================
% Figura para apresentação ao orientador — todas as imagens
% ================================================
source = 'C:\Users\felip\Documents\MATLAB\Scripts e Datasets\Datasets\Imagens Histológicas\CR\Benign';
Q_list = [8, 16, 32, 64];

% Conta e ordena os .mat
matFiles = dir(fullfile(source, '*.mat'));
matFiles = matFiles(~contains({matFiles.name}, 'RecPlot'));
matFiles = matFiles(~contains({matFiles.name}, 'mtf'));
matFiles = matFiles(~contains({matFiles.name}, 'bits'));
[~, idx] = sort(arrayfun(@(f) str2double(regexp(f.name, '^\d+', 'match', 'once')), matFiles));
matFiles = matFiles(idx);
Num_Img  = length(matFiles);
disp(['Total de imagens: ', num2str(Num_Img)])

for n = 1:Num_Img
    tic
    figure('Position', [100, 100, 1800, 400], 'Visible', 'off');

    % --- Coluna 1: Imagem histológica original ---
    subplot(1, 7, 1);
    img = imread(fullfile(source, strcat('Benign (', num2str(n), ').png')));
    imshow(img);
    title('Imagem Original', 'FontSize', 11);
    axis square;

    % --- Coluna 2: Atributos fractais como sinal 1D ---
    subplot(1, 7, 2);
    data = load(fullfile(source, strcat(num2str(n), '.mat')));
    signal = [data.ChessLAC, data.EuclLAC, data.ManhLAC, ...
              data.ChessFD,  data.EuclFD,  data.ManhFD];
    plot(signal, 'b-o', 'MarkerSize', 3);
    title('Atributos Fractais (1D)', 'FontSize', 11);
    xlabel('Índice');
    ylabel('Valor');
    grid on;

    % --- Coluna 3: Recurrence Plot ---
    subplot(1, 7, 3);
    rp = load(fullfile(source, strcat(num2str(n), '_RecPlot.mat')));
    imagesc(rp.RP);
    colormap(jet);
    colorbar;
    title('Recurrence Plot', 'FontSize', 11);
    axis square;

    % --- Colunas 4 a 7: MTF em cada resolução ---
    for i = 1:length(Q_list)
        q       = Q_list(i);
        mtfName = fullfile(source, strcat(num2str(n), '_mtf_', num2str(q), 'bits.mat'));
        mtf     = load(mtfName);

        subplot(1, 7, i + 3);
        imagesc(mtf.F);
        colormap(jet);
        colorbar;
        title(['MTF ', num2str(q), ' bins'], 'FontSize', 11);
        axis square;
    end

    sgtitle(['Pipeline de Reshape - Benign ', num2str(n)], 'FontSize', 13, 'FontWeight', 'bold');

    % Salva como n_amostragem.png
    saveas(gcf, fullfile(source, strcat(num2str(n), '_amostragem.png')));
    close(gcf);
    disp(['Salvo: ', num2str(n), '_amostragem.png'])
    toc
end

disp('Concluído!')