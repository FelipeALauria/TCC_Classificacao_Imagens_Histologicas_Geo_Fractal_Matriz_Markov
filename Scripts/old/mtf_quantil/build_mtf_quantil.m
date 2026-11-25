function F = build_mtf_quantil(x, Q)
% BUILD_MTF_QUANTIL - (versão arquivada, ver README.md) Markov Transition Field de um sinal 1D com Q estados.
% Quantização por quantil (bins de mesma frequência); Q é limitado ao
% tamanho do sinal, já que não há como ter mais estados que pontos.

    validateattributes(x, {'numeric'}, {'vector', 'nonempty', 'finite'}, 'build_mtf_quantil', 'x');
    validateattributes(Q, {'numeric'}, {'scalar', 'integer', 'positive'}, 'build_mtf_quantil', 'Q');

    x = x(:);
    Q = min(Q, numel(x));

    q = quantize_signal(x, Q);
    M = markov_matrix(q, Q);
    F = M(q, q);

end

function q = quantize_signal(x, Q)
% Posto (rank) de cada ponto; valores iguais recebem o posto médio pra
% caírem no mesmo estado.
    n = numel(x);

    [~, ord] = sort(x);
    r = zeros(n, 1);
    r(ord) = 1:n;

    [~, ~, ic] = unique(x);
    r = accumarray(ic, r, [], @mean);
    r = r(ic);

    q = ceil(r * Q / n);
end

function M = markov_matrix(q, Q)
    M = accumarray([q(1:end-1) q(2:end)], 1, [Q Q]);

    row_sums = sum(M, 2);

    zero_rows = (row_sums == 0);
    M(zero_rows, :) = 1 / Q;
    row_sums(zero_rows) = 1;

    M = M ./ row_sums;
end
