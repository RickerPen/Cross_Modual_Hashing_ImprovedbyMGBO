function [fval, grad] = obj_func_ALECH(Z, L, W, P, S, n, varargin)
% 目标函数与梯度计算（ALECH算法）
% 输入:
%   Z  - 待优化的哈希码矩阵 (n x nbits)
%   L  - 标签矩阵 (c x n)
%   W  - 投影矩阵 （c*nbits）
%   P  - 权重矩阵 (c x c)
%   S  - 标签相关性矩阵 （n*n）
%   n  - 样本量
% 输出:
%   fval - 目标函数值
%   grad - 梯度矩阵 (n x nbits)

% --- 参数初始化 ---
Alpha = 1e-5;   % 正则化系数（修正语法错误）
r = 0.95;          % 缩放因子
% --- 中间变量计算 ---
X = (1 / sqrt(n)) .* Z;                % 缩放后的哈希码 (n x nbits)
X = X';                                 % X维度调整为（nbits x n)
A = sqrt(n) .* (W * X) - P * L;      % 重构误差项 (c x n)
G = r .* S;                            % 缩放后的标签相关性矩阵 (n x n)

% --- 目标函数计算 ---
f_term1 = trace(A' * A);                % ||A||_F^2
B = n .* (X' * X) - G;                 % 正交性约束项 (n x n)
f_term2 = Alpha * trace(B' * B);        % Alpha * ||B||_F^2
fval = f_term1 + f_term2;               % 总目标函数值

% --- 梯度计算（仅在需要时计算）---
if nargout > 1
    grad_term1 = 2 * sqrt(n).* W' * (sqrt(n) .* (W * X) - P * L);
    grad_term2 = 4 * n^2 * X' * (X * X');
    grad_term3 = -2 * n * (G' + G) * X';
    grad = grad_term1 + grad_term2' + grad_term3';
    grad = grad';  % 确保梯度维度为 (n x nbits)
end
end