function Z = prox_hc(X, sqn, mu, gmma)
 
% global sqn mu gmma;

gmma_mu = mu/gmma; % 1/beta*mu

sgnX = sign(X);

% calculate Z = prox(X)
Z = sign(X)*sqn;
absX = abs(X);
set1 = absX < sqn;
set2 = absX > sqn + gmma;

Z(set1) = X(set1);
Z(set2) = X(set2) - gmma*sign(X(set2));


% [EOF]

