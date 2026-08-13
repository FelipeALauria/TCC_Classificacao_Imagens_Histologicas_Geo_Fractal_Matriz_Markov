function reshapeMTF_mat()

source      = 'C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas\CR\Malignant\maxL7';
destination = 'C:\Users\felip\Desktop\Facul\TCC\Scripts_Datasets\Datasets\Imagens Histológicas\CR\MTF\Malignant\maxL7';
class_name  = 'Malignan';
Q_list      = [8, 16, 32, 64];

matFiles = dir(fullfile(source, '*.mat'));

isValid  = ~cellfun(@isempty, regexp({matFiles.name}, '^\d+_maxL\d+\.mat$'));
matFiles = matFiles(isValid);

[~, idx] = sort(arrayfun(@(f) str2double(regexp(f.name, '^\d+', 'match', 'once')), matFiles));
matFiles  = matFiles(idx);
Num_Img  = length(matFiles);
disp(['Total de arquivos encontrados: ', num2str(Num_Img)]);

for n = 1:Num_Img
    tic
    matName = fullfile(source, matFiles(n).name);
    data    = load(matName);
    disp(['Carregando: ', matFiles(n).name]);

    signalR = [data.Chessp(:); data.Chessg(:); data.Chessh(:); data.ChessLAC(:); data.Chessnn(:)];
    signalG = [data.Euclp(:);  data.Euclg(:);  data.Euclh(:);  data.EuclLAC(:);  data.Euclnn(:)];
    signalB = [data.Manhp(:);  data.Manhg(:);  data.Manhh(:);  data.ManhLAC(:);  data.Manhnn(:)];

    maxLen  = max([length(signalR), length(signalG), length(signalB)]);
    signalR(end+1:maxLen) = 0;
    signalG(end+1:maxLen) = 0;
    signalB(end+1:maxLen) = 0;

    signalR = mat2gray(signalR(:));
    signalG = mat2gray(signalG(:));
    signalB = mat2gray(signalB(:));

    for q = Q_list
        r_channel = build_mtf(signalR, q);
        g_channel = build_mtf(signalG, q);
        b_channel = build_mtf(signalB, q);

        IMG        = zeros(size(r_channel,1), size(r_channel,2), 3);
        IMG(:,:,1) = r_channel;
        IMG(:,:,2) = g_channel;
        IMG(:,:,3) = b_channel;

        % m = tamanho do signal (NxN), q = bins da MTF (QxQ)
        m           = length(signalR);
        IMG_resized = imresize(im2uint8(IMG), [224 224], 'nearest');
        imgName     = fullfile(destination, ...
            strcat(num2str(n), '_MTF_Q', num2str(q), '_N', num2str(m), '.png'));
        imwrite(IMG_resized, imgName);
        disp(['Salvo: ', imgName]);
    end

    toc
end

disp('MTFs gerados com sucesso!');
end