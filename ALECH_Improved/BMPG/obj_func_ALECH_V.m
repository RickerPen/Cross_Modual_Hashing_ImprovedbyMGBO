function [fval, grad] = obj_func_ALECH_V(Z, L, W, P, S, n, B, varargin)
% 目标函数为 max Tr（H*V'）转化成 min -Tr(H*V')
% 输入:
%   Z  - 待优化的哈希码矩阵 (n x nbits)
%   L  - 标签矩阵 (c x n)
%   W  - 投影矩阵 （c*nbits）
%   P  - 权重矩阵 (c x c)
%   S  - 标签相关性矩阵 （n*n）
%   n  - 样本量
%   B  - 二进制约束矩阵(nbits x n)
% 输出:
%   fval - 目标函数值
%   grad - 梯度矩阵 (n x nbits)
Alpha = 1e-5;   % 正则化系数（修正语法错误）
r = 1;          % 缩放因子
Beta = 10;
H_term1 = (2*Alpha*r).*B*S'*S;
H_term2 = -Alpha*r.*B*ones(n,1)*ones(1,n);
H_term3 = Beta.*B;
H_term4 = W'*P*L;
H = H_term1+H_term2+H_term3+H_term4;
fval = -trace(H*Z);
if nargout > 1
      grad = -H';
end
end
