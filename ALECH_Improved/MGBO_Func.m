function H = MGBO_Func(Zinit,nbits, n, L, W, P, S)
% 参数说明：
% 输入：
%   nbits  - 哈希码位数
%   n      - 样本量
%   L      - 标签矩阵
%   W, P, S - 算法依赖的预定义矩阵
% 输出：
%   H      - 优化后的哈希码矩阵

% --- 1. BPGM算法参数配置 ---
opts_BPGM = struct();
opts_BPGM.gamma = 0.2;         % 梯度下降步长参数 
opts_BPGM.maxItr = 100;          % 外层最大迭代次数
opts_BPGM.record = 0;          % 是否记录过程
opts_BPGM.mxitr = 500;         % 内层最大迭代次数
opts_BPGM.sub_mxitr = 5;       % 子问题迭代次数
opts_BPGM.ftol = 1.0e-5;       % 函数值收敛阈值
opts_BPGM.gtol = 1.0e-10;      % 梯度收敛阈值
opts_BPGM.r = nbits;           % 哈希码位数
opts_BPGM.beta = 1.0;          % 惩罚系数
opts_BPGM.mu = 1e2;              % 选择固定mu值（原代码中mu_values的默认第一个值）
opts_BPGM.n = n;               % 样本量参数传递

% --- 2. 生成随机初始点 ---
seed = 13 + 100; %73
randn('state',seed)                % 固定随机种子与原始代码一致
% Zinit = randn(n, nbits);        % 初始点矩阵
Bk = orth(Zinit - repmat(mean(Zinit), n, 1)); % 中心化正交处理

% --- 3. 调用优化算法求解H ---
H = OptStiefelGBB(Bk, @obj_func_ALECH, @moreau_hc, opts_BPGM, L, W, P, S, n);
feaKer = pen_hc(H');          % 核空间可行性
% --- 4. 后处理与约束强制 ---
H = sign(H);                   % 二值化处理（与原始代码逻辑一致）
H = H';

% --- 5. 计算可行性指标（可选）---
 feaSt = norm(H*H' - n*eye(nbits), 'fro'); % 正交性误差
fprintf('\nOrth: %2.2f, Ker: %2.2f \n', feaSt, feaKer)
end