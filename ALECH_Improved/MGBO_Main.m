clear;
close all;

startup;

n_values = [500]; % 不同的 n 值
nbits = 8; %26;%4*k*r; %4*k*r;
% params for BPGM algos
opts_BPGM.gamma = 0.2;
opts_BPGM.rho = 1e-2;

opts_BPGM.maxItr   = 1;
opts_BPGM.record   = 1;
opts_BPGM.mxitr    = 500;
opts_BPGM.sub_mxitr= 5;
opts_BPGM.ftol     = 1.0e-5;
opts_BPGM.gtol     = 1.0e-10;

opts_BPGM.r        = nbits;
opts_BPGM.beta     = 1.0;   %penalty parameter

% select different mu values
mu_values = logspace(-2,8,10);% [1, 2, 5, 10, 20]; % 可以根据需要修改 mu 值

% % prepare dataset
seed = 13 + 100; %73
randn('state',seed);
% rand('state',seed);
r = nbits;

% 存储不同 n 值下的各项数据的平均值
num_n_values = length(n_values);
all_cputime_avg = zeros(length(mu_values), num_n_values);
all_feaKer_avg = zeros(length(mu_values), num_n_values);
all_feaOne_avg = zeros(length(mu_values), num_n_values);
all_feaSt_avg = zeros(length(mu_values), num_n_values);
all_Obj_avg = zeros(length(mu_values), num_n_values);

% 颜色和标记数组，用于区分不同 n 值的曲线
colors = {'r', 'g', 'b', 'm'};
markers = {'o', 's', 'd', '^'};

for n_index = 1:num_n_values
    n = n_values(n_index);
    str_dataset = ['orthrandr',num2str(nbits)];

    X = randn(n,r);
    Lamb = (sum(X).^(-1))';
    A = eye(n) - X*(Lamb.*X');

    num_initial_points = 10; % 要测试的初始点数量
    cputime_total = zeros(length(mu_values), num_initial_points);
    feaKer_total = zeros(length(mu_values), num_initial_points);
    feaOne_total = zeros(length(mu_values), num_initial_points);
    feaSt_total = zeros(length(mu_values), num_initial_points);
    Obj_total = zeros(length(mu_values), num_initial_points);

    for initial_point = 1:num_initial_points
        Zinit = randn(n, nbits); % 不同的初始点
        opts_BPGM.n = n;

        fieldname = {'feaKer', 'feaSt','Obj', 'cputime','mu_values'};

        for m = 1:length(mu_values)
            mu = mu_values(m);
            opts_BPGM.mu = mu;

            %fprintf('======start encoding with mu = %f, initial point %d, n = %d======\n\n', mu, initial_point, n);

            % give the initial point
            Bk = orth(Zinit - repmat(sum(Zinit),n,1)/n);
            a = tic;

            %                 H = BMPG_BB(Bk, @obj_func_svd, @moreau_hc, @pen_hc, opts_BPGM, X, Lamb);
            H = OptStiefelGBB(Bk, @obj_func_ALECH, @moreau_hc, opts_BPGM, L, W, P, S,n);
            feaKer = pen_hc(H);
            H = sign(H);
            feaOne = pen_hc(H);

            cputime = toc(a);

            % calculate feasibility
            Obj = obj_func(H, A);
            feaSt = norm(H'*H - n*eye(nbits),'fro');

        end
    end
end
