clear;
load('flickr.mat');
num_samples = 1000; % 根据需要调整样本量
indices = randperm(size(XTrain,1), num_samples);
XTrain = XTrain(indices, :);
YTrain = YTrain(indices, :);
LTrain = LTrain(indices, :);
%% parameter set
param.alpha = 1e-4;
param.beta = 1e-3; 
param.lamb  = 1e4;
param.eta  = 1e-4;
param.iter  = 5;
nbitset     = [8,16,32];
eva_info    = cell(1,length(nbitset));
%% centralization
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));
%% kernelization
[XKTrain,XKTest] = Kernelize(XTrain, XTest, 150); [YKTrain,YKTest]=Kernelize(YTrain,YTest, 200);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
%% ALECH
for kk= 1:length(nbitset)
    
param.nbits = nbitset(kk);

[B1, B2, B3] = ALECH_new(XKTrain, YKTrain, LTrain, param, XKTest, YKTest);

DHamm = hammingDist(B2, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
 
DHamm = hammingDist(B3, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);

eva_info{1,kk} = eva_info_;
Image_to_Text_MAP = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP = eva_info_.Text_to_Image_MAP;  

fprintf('ALECH %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP,Text_to_Image_MAP);
%ALECH_o
[B1, B2, B3] = ALECH_o(XKTrain, YKTrain, LTrain, param, XKTest, YKTest);

DHamm = hammingDist(B2, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
 
DHamm = hammingDist(B3, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);

eva_info{1,kk} = eva_info_;
Image_to_Text_MAP = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP = eva_info_.Text_to_Image_MAP;  

fprintf('ALECH %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP,Text_to_Image_MAP);

end