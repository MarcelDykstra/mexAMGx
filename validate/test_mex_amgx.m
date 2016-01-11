clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix; num_rows = size(M, 1);

cfg.config_version = 2;
cfg.solver.preconditioner.print_grid_stats = 1;
cfg.solver.preconditioner.print_vis_data = 0;
cfg.solver.preconditioner.solver = 'AMG';
cfg.solver.preconditioner.smoother.scope = 'jacobi';
cfg.solver.preconditioner.smoother.solver = 'BLOCK_JACOBI';
cfg.solver.preconditioner.smoother.monitor_residual = 0;
cfg.solver.preconditioner.smoother.print_solve_stats = 0;
cfg.solver.preconditioner.print_solve_stats = 0;
cfg.solver.preconditioner.presweeps = 1;
cfg.solver.preconditioner.max_iters = 1;
cfg.solver.preconditioner.monitor_residual = 0;
cfg.solver.preconditioner.store_res_history = 0;
cfg.solver.preconditioner.scope = 'amg';
cfg.solver.preconditioner.max_levels = 100;
cfg.solver.preconditioner.cycle = 'W';
cfg.solver.preconditioner.postsweeps = 1;
cfg.solver.solver = 'PCG';
cfg.solver.print_solve_stats = 1;
cfg.solver.obtain_timings = 1;
cfg.solver.max_iters = 300;
cfg.solver.monitor_residual = 1;
cfg.solver.convergence = 'ABSOLUTE';
cfg.solver.scope = 'main';
cfg.solver.tolerance = 1e-6;
cfg.solver.norm = 'L2';

amgA = mexAMGx(M, cfg);
% amgA = mexAMGx(M, 'PCG_F.json');
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
