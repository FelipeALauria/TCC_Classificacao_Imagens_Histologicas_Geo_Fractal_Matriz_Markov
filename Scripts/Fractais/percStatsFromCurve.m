function [MaxClusterIndex, MaxPercIndex, MaxMaxClusterIndex, AreaRatioMaxCluster, MaxMaxCluster, SkewnessMaxCluster, AreaMaxCluster, AreaRatioCluster, AreaRatioPerc, MaxCluster, MaxPerc, SkewnessCluster, SkewnessPerc, AreaPerc, AreaCluster] = percStatsFromCurve(p, g, h)
% PERCSTATSFROMCURVE - Agregados de percolação (Area/Skewness/Max/Index) a
% partir das curvas p,g,h já calculadas (ou de uma fatia delas pra um maxL
% menor). Mesmas fórmulas do trecho final de percChess.m/percEucl.m/percManh.m,
% extraídas pra função separada pra poder ser recalculada em cima de uma
% fatia sem repetir o cálculo caro (contagem de caixas) desses scripts.
%
% Exemplo de uso:
%   [MaxClusterIndex, MaxPercIndex, MaxMaxClusterIndex, AreaRatioMaxCluster, MaxMaxCluster, SkewnessMaxCluster, AreaMaxCluster, AreaRatioCluster, AreaRatioPerc, MaxCluster, MaxPerc, SkewnessCluster, SkewnessPerc, AreaPerc, AreaCluster] = percStatsFromCurve(p, g, h)

    AreaCluster = trapz(p);
    AreaPerc = trapz(g);
    AreaMaxCluster = trapz(h);
    SkewnessCluster = skewness(p);
    SkewnessPerc = skewness(g);
    SkewnessMaxCluster = skewness(h);
    [MaxCluster,MaxClusterIndex] = max(p);
    [MaxPerc,MaxPercIndex] = max(g);
    [MaxMaxCluster,MaxMaxClusterIndex] = max(h);
    half = ceil(length(p)/2);
    AreaRatioCluster = trapz(p(half+1:end))/trapz(p(1:half));
    AreaRatioPerc = trapz(g(half+1:end))/trapz(g(1:half));
    AreaRatioMaxCluster = trapz(h(half+1:end))/trapz(h(1:half));
end
