function [AreaLAC, SkewnessLAC, AreaRatioLAC, MaxLAC, MaxLACIndex] = lacStatsFromCurve(LAC)
% LACSTATSFROMCURVE - Agregados da lacunaridade (Area/Skewness/AreaRatio/Max/
% Index) a partir da curva LAC já calculada (ou de uma fatia dela pra um
% maxL menor). Mesmas fórmulas que já estavam inline em FractalFeatures.m,
% extraídas pra função separada pra reaproveitar por fatia.
%
% Exemplo de uso:
%   [AreaLAC, SkewnessLAC, AreaRatioLAC, MaxLAC, MaxLACIndex] = lacStatsFromCurve(ChessLAC)

    AreaLAC = trapz(LAC);
    SkewnessLAC = skewness(LAC);
    half = ceil(length(LAC)/2);
    AreaRatioLAC = trapz(LAC(half+1:end))/trapz(LAC(1:half));
    [MaxLAC, MaxLACIndex] = max(LAC);
end
