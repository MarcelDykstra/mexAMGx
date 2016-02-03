clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

load matrix; num_rows = size(A, 1);

cfg.config_version = 2;
cfg.solver.preconditioner.print_grid_stats = 1;
cfg.solver.preconditioner.solver = 'AMG';
cfg.solver.preconditioner.smoother.scope = 'jacobi';
cfg.solver.preconditioner.smoother.solver = 'BLOCK_JACOBI';
cfg.solver.preconditioner.presweeps = 1;
cfg.solver.preconditioner.max_iters = 1;
cfg.solver.preconditioner.scope = 'amg';
cfg.solver.preconditioner.max_levels = 100;
cfg.solver.preconditioner.cycle = 'W';
cfg.solver.preconditioner.postsweeps = 1;
cfg.solver.solver = 'PCG';
cfg.solver.print_solve_stats = 1;
cfg.solver.store_res_history = 1;
cfg.solver.obtain_timings = 1;
cfg.solver.max_iters = 300;
cfg.solver.monitor_residual = 1;
cfg.solver.convergence = 'ABSOLUTE';
cfg.solver.scope = 'main';
cfg.solver.tolerance = 1e-6;
cfg.solver.norm = 'L2';

amgA = mexAMGx(A, cfg, false);
% amgA = mexAMGx(A, 'PCG_F.json');
amgA.replace(A);

x = ones(num_rows, 1);
amgA.initial(x);

b = ones(num_rows, 1);

tic_amgx = tic;
xx = amgA \ b;
disp(['[t_amgx: ' num2str(toc(tic_amgx)) ']']);

tic_matlab = tic;
xm = A \ b;
disp(['[t_matlab: ' num2str(toc(tic_matlab)) ']']);

disp(['Matlab: [norm_residual: ' num2str(norm(A * xm - b)) ']']);
disp(['AMGx: [norm_residual: ' num2str(norm(A * xx - b)) ']']);

r = amgA.residual;
if ~isempty(r)
  figure; semilogy(r);
  grid on; box on; xlabel('iteration'); ylabel('rel. residual');
end

clear amgA;
