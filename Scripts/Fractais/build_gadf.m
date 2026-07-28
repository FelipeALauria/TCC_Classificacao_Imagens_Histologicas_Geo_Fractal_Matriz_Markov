function F = build_gadf(x)
%BUILD_GADF Gera o Gramian Angular Difference Field de um sinal 1D.
%   F = BUILD_GADF(x) recebe um vetor numérico x e retorna a matriz
%   GADF (NxN), onde N = length(x). Mono-canal, mesmo padrão do
%   build_mtf.m: o vetor inteiro é processado de uma vez.

    validateattributes(x, {'numeric'}, {'vector', 'nonempty', 'finite'}, 'build_gadf', 'x');

    x = x(:);

    x_norm = normalize_series(x);
    phi = acos(x_norm);

    F = sin(phi - phi');

end

function x_norm = normalize_series(x)
    x_min = min(x);
    x_max = max(x);

    if x_max == x_min
        x_norm = zeros(size(x));
    else
        x_norm = (2 * x - x_max - x_min) / (x_max - x_min);
    end

    x_norm = min(max(x_norm, -1), 1);
end
