function [f, g] = pen_moreau_hc(X, Lambda, mu, bta, sqn, varargin)
 
global XZ;

bta_mu = mu/bta; % 1/beta*mu
bta_Lmb = Lambda/bta; % 1/beta/Lambda

X_bta_Lmb = X - bta_Lmb; % X - 1/beta*Lambda

sgnX = sign(X_bta_Lmb);

% calculate Z = prox(X- 1/beta*Lambda)
Z = sign(X_bta_Lmb)*sqn;
absX = abs(X_bta_Lmb);
set1 = absX < sqn;
set2 = absX > sqn + bta_mu;

Z(set1) = X_bta_Lmb(set1);
Z(set2) = X_bta_Lmb(set2) - bta_mu*sign(X_bta_Lmb(set2));

XZ = X - Z;
XZ_bta_Lmb = XZ - bta_Lmb;

f = mu*sum(sum(max(0, -Z - sqn) + max(0, Z - sqn))) ...
    + (0.5*bta)*norm(XZ_bta_Lmb, 'fro')^2; % f = env(X- 1/beta*Lambda)

if nargout > 1
     g = bta*XZ_bta_Lmb;
end

end
% [EOF]

