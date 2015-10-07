classdef mexAMGx < handle
  methods
    function self = mexAMGx(conf_str)
      warn = warning('off', 'all');
      amgx_path = fileparts(which('mexAMGx'));
      curr_path = pwd; cd(amgx_path);
      loadlibrary('mex_amgx', @mex_amgx_proto);
      warning(warn);
      calllib('mex_amgx', 'mexAMGxInitialize', ...
        fullfile(amgx_path, 'configs', conf_str));
      cd(curr_path);
    end
    function matrixUploadA(self, A)
      calllib('mex_amgx', 'mexAMGxMatrixUploadA', A');
    end
    function matrixReplaceCoeffA(self, A)
      calllib('mex_amgx', 'mexAMGxMatrixReplaceCoeffA', A');
    end
    function vectorUploadX(self, x)
      calllib('mex_amgx', 'mexAMGxVectorUploadX', x);
    end
    function vectorUploadB(self, b)
      calllib('mex_amgx', 'mexAMGxVectorUploadB', b);
    end
    function vectorSetZeroX(self)
      calllib('mex_amgx', 'mexAMGxVectorSetZeroX');
    end
    function x = vectorDownloadX(self)
      x = calllib('mex_amgx', 'mexAMGxVectorDownloadX');
    end
    function readSystem(self)
      calllib('mex_amgx', 'mexAMGxReadSystem');
    end
    function solverSetup(self)
      calllib('mex_amgx', 'mexAMGxSolverSetup');
    end
    function solverSolve(self)
      calllib('mex_amgx', 'mexAMGxSolverSolve');
    end
    function delete(self)
      disp('mexAMGx: finalize');
      calllib('mex_amgx','mexAMGxFinalize');
      unloadlibrary('mex_amgx');
    end
  end
end
