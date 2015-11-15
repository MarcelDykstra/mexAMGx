clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix; num_rows = size(M, 1);

amgA = mexAMGx(M, 'W.json');
amgA.replace(M);

x = ones(num_rows, 1);
amgA.initial(x);

b = ones(num_rows, 1);
xx = amgA \ b;

tic_matlab = tic;
xm = M \ b;
disp(['[t_matlab: ' num2str(toc(tic_matlab)) ']']);

disp(['Matlab: [norm_residual: ' num2str(norm(M * xm - b)) ']']);
disp(['AMGx: [norm_residual: ' num2str(norm(M * xx - b)) ']']);
clear amgA;
