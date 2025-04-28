function [fval, grad] = obj_func_AMSH_V(Z, L, P, R, E, n, B, V_j, L_j, varargin)
% 目标函数为 max Tr（H*V'）转化成 min -Tr(H*V')
% 输入:
%   Z  - 待优化的哈希码矩阵 (r x n)
%   L  - 标签矩阵 (c x n)
%   P  - 权重矩阵 (c x r)
%   S  - 标签相关性矩阵 （n*n）
%   n  - 样本量
%   B  - 二进制约束矩阵(nbits x n)
%   E -  (c X n)
% 输出:
%   fval - 目标函数值
%   grad - 梯度矩阵 (n x nbits)

lambda = 1e-3;
eta = 1e0;
r = 1;          % 缩放因子
Beta = 1e-3;
Z = Z';
H_term1 = P'*(L+R.*E);
H_term2 = eta.*B;
H_term3 = lambda*r.*B*L'*L;
H_term4 = Beta*r.*V_j*L_j'*L;
H = H_term1+H_term2+H_term3+H_term4;
fval = -trace(H*Z');
if nargout > 1
     grad = -H';
end
end
