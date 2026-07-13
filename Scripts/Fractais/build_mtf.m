function F = build_mtf(x, Q)
    
    validateattributes(x, {'numeric'}, {'vector', 'nonempty', 'finite'}, 'build_mtf', 'x');
    validateattributes(Q, {'numeric'}, {'scalar', 'integer', 'positive'}, 'build_mtf', 'Q');

    x = x(:);

    q = quantize_signal(x, Q);
    M = markov_matrix(q, Q);
    F = markov_field(q, M);

end

function q = quantize_signal(x, Q)
    
    edges = linspace(min(x), max(x), Q + 1);
    edges(end) = edges(end) + eps;

    [~, ~, q] = histcounts(x, edges);

    q(q == 0) = 1;
end


function M = markov_matrix(q, Q)
    M = zeros(Q, Q);

    for i = 1:length(q) - 1
        M(q(i), q(i+1)) = M(q(i), q(i+1)) + 1;
    end

    row_sums = sum(M, 2);

    zero_rows = (row_sums == 0);
    M(zero_rows, :) = 1 / Q;
    row_sums(zero_rows) = 1;

    M = M ./ row_sums;

end


function F = markov_field(q, M)

    n = length(q);
    F = zeros(n, n);

    for i = 1:n
        for j = 1:n
            F(i, j) = M(q(i), q(j));
        end
    end

end