img = imread('../data/sample.png');
maxL = 65;

[globalFeatures, C,Q,M] = percolation(img,maxL)