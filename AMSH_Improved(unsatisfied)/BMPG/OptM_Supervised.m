function [G,F,B] = OptM_Supervised(X,Y,B,nbits,opts_pgm)
% X= feaTrain;
% y= traingnd;
%
% optimize the supervised hashing problem by BMPG.
% B : learned binary codes.
% F : learned Hash function.
% G : learned classification matrix.
global sqn;

% if nargin < 6 || isempty(mani)
%     mani = 1;
% end

tol = 1e-6;
n = opts_pgm.n;
opts_pgm.r = nbits;

sqn = 1/sqrt(n);
% label matrix N x c

% Y = nbits*Y;

G = [];

% init B
Wg0 = zeros(nbits,size(Y,2));

i = 0;
while i < opts_pgm.maxItr
    i=i+1;
    % W-step
    Wg = RRC(sign(B), Y, 1);
    G.W = Wg;
    conv_w = norm(Wg-Wg0,'fro');
    fprintf('convW: %2.2f\n', conv_w);
    if conv_w < tol*norm(Wg0,'fro')
        break
    end    
    Wg0 = Wg;
    
    % B-step    
    B0 = B;
    
%     Xk = B;
%     ffun = @obj_func_BIP;
%     gfun = @moreau_hc;
%     penfun = @pen_hc;
%     varargin = {Wg, Y};

%     B = ABMO(B,Y,Wg,opts_pgm);
%     Z = B*sqn;

%     Z = orth(B );    
    Z = orth(B - repmat(sum(B),n,1)/n);
    
    B = OptStiefelGBB(Z, @obj_func_BIP, @moreau_hc, opts_pgm, Wg, Y);
    % if mani == 1
    %     B = BMPG_BB_old(Z, @obj_func_BIP, @moreau_hc, @pen_hc, opts_pgm, Wg, Y);
    % elseif mani == 2
    %     B = BPG_BB(Z, @obj_func_BIP, @moreau_hc, @pen_hc, opts_pgm, Wg, Y);
    % end
%     B = sign(Z);  % norm(Z'*Z - eye(nbits)) %  norm(Z'*ones(n,1))
    
    fprintf('\n obj val: %2.2f\n', obj_func_BIP(B,Wg,Y))

    conv_B = norm(B-B0,'fro'); % norm(B'*B - eye(nbits)*n) %  norm(B'*ones(n,1))
    fprintf('convB: %2.2f\n', conv_B);
    if conv_B < tol*norm(B0,'fro') 
        break
    end
   fprintf('\n');
end

% F-step
fprintf('obj val: %2.2f\n', obj_func_BIP(B,Wg,Y))
B = sign(B);
WF = RRC(X, B, 1e-2);
F.W = WF;
% B = sign(B);