function [FD, nn] = fractalDimensionFromProbMatrix(ProbMatrix, r)
% FRACTALDIMENSIONFROMPROBMATRIX - Dimensão fractal (regressão robusta de
% -log(nn) por log(r)) a partir da matriz de probabilidade já calculada
% (ou de uma fatia de colunas dela pra um maxL menor). Mesma lógica que já
% estava inline em FractalFeatures.m, extraída pra função separada pra
% reaproveitar por fatia sem precisar rodar pmrChess/pmrEucl/pmrManh de novo.
%
% Exemplo de uso:
%   [ChessFD, Chessnn] = fractalDimensionFromProbMatrix(ProbMatrixChess, r)

    nn = calcN(ProbMatrix);
    x = log(r);
    y = -log(nn);
    mdl = fitlm(x.', y.', 'RobustOpts', 'on');
    FD = mdl.Coefficients.Estimate(2);
end
