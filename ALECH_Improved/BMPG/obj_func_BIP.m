function [f, g] = obj_func_BIP(Z, Wg, Y, varargin)


% global sqn;
sn = sqrt(size(Z,1));


% case 1
f = -sn*sum(sum((Z'*Y).*Wg));

% case 2 
% B = sn*Z;
% B = Z;
% BW = B*Wg;
% f = (sum(sum((.5*BW-Y).*BW)));
% f = .5*norm(BW - Y,'fro')^2;

if nargout > 1
    % case 1
    g =  -sn*Y*Wg';

    % case 2 sn.*
%     g = sn*(BW - Y)*Wg';
end
