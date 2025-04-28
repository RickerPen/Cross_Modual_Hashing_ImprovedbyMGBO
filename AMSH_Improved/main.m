clear all
dataset = 'mirflickr25k.mat';
addpath('BMPG\')
%% database selection
[LTrain, LTest,XTrain,XTest,YTrain,YTest] = loaddata(dataset);
% 控制样本量参数设置
train_num = 2000; % 训练样本量
test_num = 500;   % 测试样本量

% 随机抽取训练样本
train_idx = randperm(size(XTrain,1), train_num);
XTrain = XTrain(train_idx, :);
YTrain = YTrain(train_idx, :);
LTrain = LTrain(train_idx, :);

% 随机抽取测试样本
test_idx = randperm(size(XTest,1), test_num);
XTest = XTest(test_idx, :);
YTest = YTest(test_idx, :);
LTest = LTest(test_idx, :);
%% parameter set
param.beta = 1e-3;
param.lambda = 1e-3;
param.eta = 1e0;
param.iter  = 15;
nbitset     = [8 16];
eva_info    = cell(1,length(nbitset));
%% centralization
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));
%% kernelization
[XKTrain,XKTest] = Kernelize(XTrain, XTest, 150); [YKTrain,YKTest]=Kernelize(YTrain,YTest, 100);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
%% data preprocessing
index1 = randperm(size(XKTrain,1));
index2 = randperm(size(YKTrain,1));%shuffle data
X = XKTrain(index1,:)';
Y = YKTrain(index2,:)';
LX = LTrain(index1,:)';
LY = LTrain(index2,:)';

%% AMSH
Image_to_Text_MAP = zeros(1,length(nbitset));
Text_to_Image_MAP = zeros(1,length(nbitset));
for kk= 1:length(nbitset)
    
param.nbits = nbitset(kk);
[B1,B2,B1_test,B2_test] = AMSH(X, Y, LX, LY, param, XKTest, YKTest);

DHamm = pdist2(B1_test, B2,'hamming');
% DHamm = hammingDist(B1_test, B2);
[~, orderH] = sort(DHamm, 2);
eva_info_.Image_to_Text_MAP = mAP(orderH', LY', LTest);

DHamm = pdist2(B2_test, B1,'hamming');
% DHamm = hammingDist(B2_test, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_to_Image_MAP = mAP(orderH', LX', LTest);

eva_info{1,kk} = eva_info_;
Image_to_Text_MAP(kk) = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP(kk) = eva_info_.Text_to_Image_MAP;  

fprintf('AMSH %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP(kk),Text_to_Image_MAP(kk));
end
