clear all; clc;
curr_path = pwd; cd ..; addpath(pwd); cd(curr_path);

objAMGx = mexAMGx('W.json');
load matrix; num_rows = size(M, 1);

objAMGx.matrixUploadA(M);
objAMGx.solverSetup();
objAMGx.matrixReplaceCoeffA(M);

x = ones(num_rows, 1);
objAMGx.vectorUploadX(x);
xx = objAMGx.vectorDownloadX();
disp([ 'Vector transfer: [norm_residual: ' num2str(norm(xx - x)) ']']);

objAMGx.vectorSetZeroX();
xx = objAMGx.vectorDownloadX();
disp([ 'Vector set zero: [norm_residual: ' num2str(norm(xx)) ']']);

b = ones(num_rows, 1);
objAMGx.vectorUploadB(b);
objAMGx.solverSolve();

xx = objAMGx.vectorDownloadX();

tic_matlab = tic;
xm = M \ b;
disp(['[t_matlab: ' num2str(toc(tic_matlab)) ']']);

disp(['Matlab: [norm_residual: ' num2str(norm(M * xm - b)) ']']);
disp(['AMGx: [norm_residual: ' num2str(norm(M * xx - b)) ']']);
clear objAMGx;
