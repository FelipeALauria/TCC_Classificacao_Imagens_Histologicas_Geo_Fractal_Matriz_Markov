%%Salva os atributos obtidos com ScriptPercLACDF3Distances
%%em um arquivo CSV
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Guilherme Freire Roberto


%%%%%%%%%%%%%%%%%%
%%
%%Número de amostras da Classe 1
class1 = 1424;
diretorioOrigem2='1st class path'; %change accordingly
features=zeros(class1,364);
for i = 1:class1
    filename=strcat(num2str(i),'.mat');
    orgName=fullfile(diretorioOrigem2,filename);
    load(orgName);
    features(i, 1:20) = Minkp;
    features(i, 21:40) = Minkg;
    features(i, 41:60) = Minkh;
    features(i, 61:80) = MinkLAC;
    features(i, 81:100) = Minknn;
    features(i, 101:120) = Euclp;
    features(i, 121:140) = Euclg;
    features(i, 141:160) = Euclh;
    features(i, 161:180) = EuclLAC;
    features(i, 181:200) = Euclnn;
    features(i, 201:220) = Manhp;
    features(i, 221:240) = Manhg;
    features(i, 241:260) = Manhh;
    features(i, 261:280) = ManhLAC;
    features(i, 281:300) = Manhnn;
    features(i, 301) = MinkAreaCluster;
    features(i, 302) = MinkSkewnessCluster;
    features(i, 303) = MinkAreaRatioCluster;
    features(i, 304) = MinkMaxCluster;
    features(i, 305) = MinkMaxClusterIndex;
    features(i, 306) = MinkAreaPerc;
    features(i, 307) = MinkSkewnessPerc;
    features(i, 308) = MinkAreaRatioPerc;
    features(i, 309) = MinkMaxPerc;
    features(i, 310) = MinkMaxPercIndex;
    features(i, 311) = MinkAreaMaxCluster;
    features(i, 312) = MinkSkewnessMaxCluster;
    features(i, 313) = MinkAreaRatioMaxCluster;
    features(i, 314) = MinkMaxMaxCluster;
    features(i, 315) = MinkMaxMaxClusterIndex;
    features(i, 316) = EuclAreaCluster;
    features(i, 317) = EuclSkewnessCluster;
    features(i, 318) = EuclAreaRatioCluster;
    features(i, 309) = EuclMaxCluster;
    features(i, 320) = EuclMaxClusterIndex;
    features(i, 321) = EuclAreaPerc;
    features(i, 322) = EuclSkewnessPerc;
    features(i, 323) = EuclAreaRatioPerc;
    features(i, 324) = EuclMaxPerc;
    features(i, 325) = EuclMaxPercIndex;
    features(i, 326) = EuclAreaMaxCluster;
    features(i, 327) = EuclSkewnessMaxCluster;
    features(i, 328) = EuclAreaRatioMaxCluster;
    features(i, 329) = EuclMaxMaxCluster;
    features(i, 330) = EuclMaxMaxClusterIndex;
    features(i, 331) = ManhAreaCluster;
    features(i, 332) = ManhSkewnessCluster;
    features(i, 333) = ManhAreaRatioCluster;
    features(i, 334) = ManhMaxCluster;
    features(i, 335) = ManhMaxClusterIndex;
    features(i, 336) = ManhAreaPerc;
    features(i, 337) = ManhSkewnessPerc;
    features(i, 338) = ManhAreaRatioPerc;
    features(i, 339) = ManhMaxPerc;
    features(i, 340) = ManhMaxPercIndex;
    features(i, 341) = ManhAreaMaxCluster;
    features(i, 342) = ManhSkewnessMaxCluster;
    features(i, 343) = ManhAreaRatioMaxCluster;
    features(i, 344) = ManhMaxMaxCluster;
    features(i, 345) = ManhMaxMaxClusterIndex;
    features(i, 346) = MinkAreaLAC;
    features(i, 347) = MinkSkewnessLAC;
    features(i, 348) = MinkAreaRatioLAC;
    features(i, 349) = MinkMaxLAC;
    features(i, 350) = MinkMaxLACIndex;
    features(i, 351) = EuclAreaLAC;
    features(i, 352) = EuclSkewnessLAC;
    features(i, 353) = EuclAreaRatioLAC;
    features(i, 354) = EuclMaxLAC;
    features(i, 355) = EuclMaxLACIndex;
    features(i, 356) = ManhAreaLAC;
    features(i, 357) = ManhSkewnessLAC;
    features(i, 358) = ManhAreaRatioLAC;
    features(i, 359) = ManhMaxLAC;
    features(i, 360) = ManhMaxLACIndex;
    features(i, 361) = MinkDF;
    features(i, 362) = EuclDF;
    features(i, 363) = ManhDF;
    features(i, 364) = 0;
end

%%
%%Número de amostras da Classe 2
class2 = 706;
%%%%%%%%%%%%%%%%%%%%%%%%%%
diretorioOrigem2='2nd class path'; %change accordingly
%%%%%%%%%%%%%%%%%%
features2=zeros(class2,364);
for i = 1:class2
    filename=strcat(num2str(i),'.mat');
    orgName=fullfile(diretorioOrigem2,filename);
    load(orgName);
    features2(i, 1:20) = Minkp;
    features2(i, 21:40) = Minkg;
    features2(i, 41:60) = Minkh;
    features2(i, 61:80) = MinkLAC;
    features2(i, 81:100) = Minknn;
    features2(i, 101:120) = Euclp;
    features2(i, 121:140) = Euclg;
    features2(i, 141:160) = Euclh;
    features2(i, 161:180) = EuclLAC;
    features2(i, 181:200) = Euclnn;
    features2(i, 201:220) = Manhp;
    features2(i, 221:240) = Manhg;
    features2(i, 241:260) = Manhh;
    features2(i, 261:280) = ManhLAC;
    features2(i, 281:300) = Manhnn;
    features2(i, 301) = MinkAreaCluster;
    features2(i, 302) = MinkSkewnessCluster;
    features2(i, 303) = MinkAreaRatioCluster;
    features2(i, 304) = MinkMaxCluster;
    features2(i, 305) = MinkMaxClusterIndex;
    features2(i, 306) = MinkAreaPerc;
    features2(i, 307) = MinkSkewnessPerc;
    features2(i, 308) = MinkAreaRatioPerc;
    features2(i, 309) = MinkMaxPerc;
    features2(i, 310) = MinkMaxPercIndex;
    features2(i, 311) = MinkAreaMaxCluster;
    features2(i, 312) = MinkSkewnessMaxCluster;
    features2(i, 313) = MinkAreaRatioMaxCluster;
    features2(i, 314) = MinkMaxMaxCluster;
    features2(i, 315) = MinkMaxMaxClusterIndex;
    features2(i, 316) = EuclAreaCluster;
    features2(i, 317) = EuclSkewnessCluster;
    features2(i, 318) = EuclAreaRatioCluster;
    features2(i, 309) = EuclMaxCluster;
    features2(i, 320) = EuclMaxClusterIndex;
    features2(i, 321) = EuclAreaPerc;
    features2(i, 322) = EuclSkewnessPerc;
    features2(i, 323) = EuclAreaRatioPerc;
    features2(i, 324) = EuclMaxPerc;
    features2(i, 325) = EuclMaxPercIndex;
    features2(i, 326) = EuclAreaMaxCluster;
    features2(i, 327) = EuclSkewnessMaxCluster;
    features2(i, 328) = EuclAreaRatioMaxCluster;
    features2(i, 329) = EuclMaxMaxCluster;
    features2(i, 330) = EuclMaxMaxClusterIndex;
    features2(i, 331) = ManhAreaCluster;
    features2(i, 332) = ManhSkewnessCluster;
    features2(i, 333) = ManhAreaRatioCluster;
    features2(i, 334) = ManhMaxCluster;
    features2(i, 335) = ManhMaxClusterIndex;
    features2(i, 336) = ManhAreaPerc;
    features2(i, 337) = ManhSkewnessPerc;
    features2(i, 338) = ManhAreaRatioPerc;
    features2(i, 339) = ManhMaxPerc;
    features2(i, 340) = ManhMaxPercIndex;
    features2(i, 341) = ManhAreaMaxCluster;
    features2(i, 342) = ManhSkewnessMaxCluster;
    features2(i, 343) = ManhAreaRatioMaxCluster;
    features2(i, 344) = ManhMaxMaxCluster;
    features2(i, 345) = ManhMaxMaxClusterIndex;
    features2(i, 346) = MinkAreaLAC;
    features2(i, 347) = MinkSkewnessLAC;
    features2(i, 348) = MinkAreaRatioLAC;
    features2(i, 349) = MinkMaxLAC;
    features2(i, 350) = MinkMaxLACIndex;
    features2(i, 351) = EuclAreaLAC;
    features2(i, 352) = EuclSkewnessLAC;
    features2(i, 353) = EuclAreaRatioLAC;
    features2(i, 354) = EuclMaxLAC;
    features2(i, 355) = EuclMaxLACIndex;
    features2(i, 356) = ManhAreaLAC;
    features2(i, 357) = ManhSkewnessLAC;
    features2(i, 358) = ManhAreaRatioLAC;
    features2(i, 359) = ManhMaxLAC;
    features2(i, 360) = ManhMaxLACIndex;
    features2(i, 361) = MinkDF;
    features2(i, 362) = EuclDF;
    features2(i, 363) = ManhDF;
    features2(i, 364) = 1;
end

%%
%%Número de amostras da Classe 3
class3 = 930;
%%%%%%%%%%%%%%%%%%%%%%%%%%
diretorioOrigem2='3rd class path'; %change accordingly
%%%%%%%%%%%%%%%%%%
features3=zeros(class3,364);
for i = 1:class3
    filename=strcat(num2str(i),'.mat');
    orgName=fullfile(diretorioOrigem2,filename);
    load(orgName);
    features3(i, 1:20) = Minkp;
    features3(i, 21:40) = Minkg;
    features3(i, 41:60) = Minkh;
    features3(i, 61:80) = MinkLAC;
    features3(i, 81:100) = Minknn;
    features3(i, 101:120) = Euclp;
    features3(i, 121:140) = Euclg;
    features3(i, 141:160) = Euclh;
    features3(i, 161:180) = EuclLAC;
    features3(i, 181:200) = Euclnn;
    features3(i, 201:220) = Manhp;
    features3(i, 221:240) = Manhg;
    features3(i, 241:260) = Manhh;
    features3(i, 261:280) = ManhLAC;
    features3(i, 281:300) = Manhnn;
    features3(i, 301) = MinkAreaCluster;
    features3(i, 302) = MinkSkewnessCluster;
    features3(i, 303) = MinkAreaRatioCluster;
    features3(i, 304) = MinkMaxCluster;
    features3(i, 305) = MinkMaxClusterIndex;
    features3(i, 306) = MinkAreaPerc;
    features3(i, 307) = MinkSkewnessPerc;
    features3(i, 308) = MinkAreaRatioPerc;
    features3(i, 309) = MinkMaxPerc;
    features3(i, 310) = MinkMaxPercIndex;
    features3(i, 311) = MinkAreaMaxCluster;
    features3(i, 312) = MinkSkewnessMaxCluster;
    features3(i, 313) = MinkAreaRatioMaxCluster;
    features3(i, 314) = MinkMaxMaxCluster;
    features3(i, 315) = MinkMaxMaxClusterIndex;
    features3(i, 316) = EuclAreaCluster;
    features3(i, 317) = EuclSkewnessCluster;
    features3(i, 318) = EuclAreaRatioCluster;
    features3(i, 309) = EuclMaxCluster;
    features3(i, 320) = EuclMaxClusterIndex;
    features3(i, 321) = EuclAreaPerc;
    features3(i, 322) = EuclSkewnessPerc;
    features3(i, 323) = EuclAreaRatioPerc;
    features3(i, 324) = EuclMaxPerc;
    features3(i, 325) = EuclMaxPercIndex;
    features3(i, 326) = EuclAreaMaxCluster;
    features3(i, 327) = EuclSkewnessMaxCluster;
    features3(i, 328) = EuclAreaRatioMaxCluster;
    features3(i, 329) = EuclMaxMaxCluster;
    features3(i, 330) = EuclMaxMaxClusterIndex;
    features3(i, 331) = ManhAreaCluster;
    features3(i, 332) = ManhSkewnessCluster;
    features3(i, 333) = ManhAreaRatioCluster;
    features3(i, 334) = ManhMaxCluster;
    features3(i, 335) = ManhMaxClusterIndex;
    features3(i, 336) = ManhAreaPerc;
    features3(i, 337) = ManhSkewnessPerc;
    features3(i, 338) = ManhAreaRatioPerc;
    features3(i, 339) = ManhMaxPerc;
    features3(i, 340) = ManhMaxPercIndex;
    features3(i, 341) = ManhAreaMaxCluster;
    features3(i, 342) = ManhSkewnessMaxCluster;
    features3(i, 343) = ManhAreaRatioMaxCluster;
    features3(i, 344) = ManhMaxMaxCluster;
    features3(i, 345) = ManhMaxMaxClusterIndex;
    features3(i, 346) = MinkAreaLAC;
    features3(i, 347) = MinkSkewnessLAC;
    features3(i, 348) = MinkAreaRatioLAC;
    features3(i, 349) = MinkMaxLAC;
    features3(i, 350) = MinkMaxLACIndex;
    features3(i, 351) = EuclAreaLAC;
    features3(i, 352) = EuclSkewnessLAC;
    features3(i, 353) = EuclAreaRatioLAC;
    features3(i, 354) = EuclMaxLAC;
    features3(i, 355) = EuclMaxLACIndex;
    features3(i, 356) = ManhAreaLAC;
    features3(i, 357) = ManhSkewnessLAC;
    features3(i, 358) = ManhAreaRatioLAC;
    features3(i, 359) = ManhMaxLAC;
    features3(i, 360) = ManhMaxLACIndex;
    features3(i, 361) = MinkDF;
    features3(i, 362) = EuclDF;
    features3(i, 363) = ManhDF;
    features3(i, 364) = 2;
end

%%
%%Cabeçalho do arquivo (pronto para ser convertido para ARFF)
header = cell(368,1);
header{1, 1} = '@RELATION BrainSegPERCLACDF3Dist41GB2';
for i = 3:22
        header{i, 1} = strcat('@ATTRIBUTE Minkp', num2str(i-2), ' REAL');
end
for i = 23:42
        header{i, 1} = strcat('@ATTRIBUTE Minkg', num2str(i-22), ' REAL');
end
for i = 43:62
        header{i, 1} = strcat('@ATTRIBUTE Minkh', num2str(i-42), ' REAL');
end
for i = 63:82
        header{i, 1} = strcat('@ATTRIBUTE MinkLAC', num2str(i-62), ' REAL');
end
for i = 83:102
        header{i, 1} = strcat('@ATTRIBUTE MinkDF', num2str(i-82), ' REAL');
end

for i = 103:122
        header{i, 1} = strcat('@ATTRIBUTE Euclp', num2str(i-102), ' REAL');
end
for i = 123:142
        header{i, 1} = strcat('@ATTRIBUTE Euclg', num2str(i-122), ' REAL');
end
for i = 143:162
        header{i, 1} = strcat('@ATTRIBUTE Euclh', num2str(i-142), ' REAL');
end
for i = 163:182
        header{i, 1} = strcat('@ATTRIBUTE EuclLAC', num2str(i-162), ' REAL');
end
for i = 183:202
        header{i, 1} = strcat('@ATTRIBUTE EuclDF', num2str(i-182), ' REAL');
end

for i = 203:222
        header{i, 1} = strcat('@ATTRIBUTE Manhp', num2str(i-202), ' REAL');
end
for i = 223:242
        header{i, 1} = strcat('@ATTRIBUTE Manhg', num2str(i-222), ' REAL');
end
for i = 243:262
        header{i, 1} = strcat('@ATTRIBUTE Manhh', num2str(i-242), ' REAL');
end
for i = 263:282
        header{i, 1} = strcat('@ATTRIBUTE ManhLAC', num2str(i-262), ' REAL');
end
for i = 283:302
        header{i, 1} = strcat('@ATTRIBUTE ManhDF', num2str(i-282), ' REAL');
end

    header{303, 1} = '@ATTRIBUTE MinkAreaCluster REAL';
    header{304, 1} = '@ATTRIBUTE MinkSkewnessCluster REAL';
    header{305, 1} = '@ATTRIBUTE MinkAreaRatioCluster REAL';
    header{306, 1} = '@ATTRIBUTE MinkMaxCluster REAL';
    header{307, 1} = '@ATTRIBUTE MinkMaxClusterIndex REAL';
    header{308, 1} = '@ATTRIBUTE MinkAreaPerc REAL';
    header{309, 1} = '@ATTRIBUTE MinkSkewnessPerc REAL';
    header{310, 1} = '@ATTRIBUTE MinkAreaRatioPerc REAL';
    header{311, 1} = '@ATTRIBUTE MinkMaxPerc REAL';
    header{312, 1} = '@ATTRIBUTE MinkMaxPercIndex REAL';
    header{313, 1} = '@ATTRIBUTE MinkAreaMaxCluster REAL';
    header{314, 1} = '@ATTRIBUTE MinkSkewnessMaxCluster REAL';
    header{315, 1} = '@ATTRIBUTE MinkAreaRatioMaxCluster REAL';
    header{316, 1} = '@ATTRIBUTE MinkMaxMaxCluster REAL';
    header{317, 1} = '@ATTRIBUTE MinkMaxMaxClusterIndex REAL';
    header{318, 1} = '@ATTRIBUTE EuclAreaCluster REAL';
    header{319, 1} = '@ATTRIBUTE EuclSkewnessCluster REAL';
    header{320, 1} = '@ATTRIBUTE EuclAreaRatioCluster REAL';
    header{321, 1} = '@ATTRIBUTE EuclMaxCluster REAL';
    header{322, 1} = '@ATTRIBUTE EuclMaxClusterIndex REAL';
    header{323, 1} = '@ATTRIBUTE EuclAreaPerc REAL';
    header{324, 1} = '@ATTRIBUTE EuclSkewnessPerc REAL';
    header{325, 1} = '@ATTRIBUTE EuclAreaRatioPerc REAL';
    header{326, 1} = '@ATTRIBUTE EuclMaxPerc REAL';
    header{327, 1} = '@ATTRIBUTE EuclMaxPercIndex REAL';
    header{328, 1} = '@ATTRIBUTE EuclAreaMaxCluster REAL';
    header{329, 1} = '@ATTRIBUTE EuclSkewnessMaxCluster REAL';
    header{330, 1} = '@ATTRIBUTE EuclAreaRatioMaxCluster REAL';
    header{331, 1} = '@ATTRIBUTE EuclMaxMaxCluster REAL';
    header{332, 1} = '@ATTRIBUTE EuclMaxMaxClusterIndex REAL';
    header{333, 1} = '@ATTRIBUTE ManhAreaCluster REAL';
    header{334, 1} = '@ATTRIBUTE ManhSkewnessCluster REAL';
    header{335, 1} = '@ATTRIBUTE ManhAreaRatioCluster REAL';
    header{336, 1} = '@ATTRIBUTE ManhMaxCluster REAL';
    header{337, 1} = '@ATTRIBUTE ManhMaxClusterIndex REAL';
    header{338, 1} = '@ATTRIBUTE ManhAreaPerc REAL';
    header{339, 1} = '@ATTRIBUTE ManhSkewnessPerc REAL';
    header{340, 1} = '@ATTRIBUTE ManhAreaRatioPerc REAL';
    header{341, 1} = '@ATTRIBUTE ManhMaxPerc REAL';
    header{342, 1} = '@ATTRIBUTE ManhMaxPercIndex REAL';
    header{343, 1} = '@ATTRIBUTE ManhAreaMaxCluster REAL';
    header{344, 1} = '@ATTRIBUTE ManhSkewnessMaxCluster REAL';
    header{345, 1} = '@ATTRIBUTE ManhAreaRatioMaxCluster REAL';
    header{346, 1} = '@ATTRIBUTE ManhMaxMaxCluster REAL';
    header{347, 1} = '@ATTRIBUTE ManhMaxMaxClusterIndex REAL';
    header{348, 1} = '@ATTRIBUTE MinkAreaLAC REAL';
    header{349, 1} = '@ATTRIBUTE MinkSkewnessLAC REAL';
    header{350, 1} = '@ATTRIBUTE MinkAreaRatioLAC REAL';
    header{351, 1} = '@ATTRIBUTE MinkMaxLAC REAL';
    header{352, 1} = '@ATTRIBUTE MinkMaxLACIndex REAL';
    header{353, 1} = '@ATTRIBUTE EuclAreaLAC REAL';
    header{354, 1} = '@ATTRIBUTE EuclSkewnessLAC REAL';
    header{355, 1} = '@ATTRIBUTE EuclAreaRatioLAC REAL';
    header{356, 1} = '@ATTRIBUTE EuclMaxLAC REAL';
    header{357, 1} = '@ATTRIBUTE EuclMaxLACIndex REAL';
    header{358, 1} = '@ATTRIBUTE ManhAreaLAC REAL';
    header{359, 1} = '@ATTRIBUTE ManhSkewnessLAC REAL';
    header{360, 1} = '@ATTRIBUTE ManhAreaRatioLAC REAL';
    header{361, 1} = '@ATTRIBUTE ManhMaxLAC REAL';
    header{362, 1} = '@ATTRIBUTE ManhMaxLACIndex REAL';
    header{363, 1} = '@ATTRIBUTE MinkDF REAL';
    header{364, 1} = '@ATTRIBUTE EuclDF REAL';
    header{365, 1} = '@ATTRIBUTE ManhDF REAL';
    header{366, 1} = '@ATTRIBUTE class {0, 1}';
    header{368, 1} = '@DATA';
    
features = num2cell(features);
features2 = num2cell(features2);
features3 = num2cell(features3);
%features4 = num2cell(features4);
finalcsv = cell(368+class1+class2,364);
finalcsv(1:368, 1) = header;
finalcsv(369:369+class1-1, 1:364) = features;
finalcsv(369+class1:369+class1+class2-1, 1:364) = features2;
finalcsv(369+class1+class2:369+class1+class2+class3-1, 1:364) = features3;
%finalcsv(369+class1+class2+class3:369+class1+class2+class3+class4-1, 1:364) = features4;
T = cell2table(finalcsv);
writetable(T,'Filename.csv');    