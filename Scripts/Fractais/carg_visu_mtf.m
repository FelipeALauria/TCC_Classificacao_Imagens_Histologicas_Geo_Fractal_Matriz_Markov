% Carrega e plota as 4 variações da imagem 1
figure;

Q_list = [8, 16, 32, 64];

for i = 1:length(Q_list)
    q = Q_list(i);

    % Carrega o arquivo correspondente
    mtfName = fullfile(source, ...
        strcat('benign_1_mtf_', num2str(q), 'bits.mat'));
    data = load(mtfName);

    % Plota lado a lado
    subplot(1, 4, i);
    imagesc(data.F);
    colormap(jet);
    colorbar;
    title([num2str(q), ' bins']);
    axis square;
end

sgtitle('MTF - Benign 1 - Comparação de Resoluções');