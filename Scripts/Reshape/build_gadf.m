function F = build_gadf(x)
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
