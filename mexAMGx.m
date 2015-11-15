classdef mexAMGx < handle
  properties (Access = private)
    x_initial = [];
  end
  methods
    function self = mexAMGx(A, conf_str)
      warn = warning('off', 'all');
      amgx_path = fileparts(which('mexAMGx'));
      curr_path = pwd; cd(amgx_path);
      loadlibrary('mex_amgx', @mex_amgx_proto);
      warning(warn);
      calllib('mex_amgx', 'mexAMGxInitialize', ...
        fullfile(amgx_path, 'configs', conf_str));
      cd(curr_path);
      calllib('mex_amgx', 'mexAMGxMatrixUploadA', A');
      calllib('mex_amgx', 'mexAMGxSolverSetup');
    end
    function x = mldivide(self, b)
      calllib('mex_amgx', 'mexAMGxVectorUploadB', b);
      if isempty(self.x_initial)
        calllib('mex_amgx', 'mexAMGxVectorSetZeroX');
      else
        calllib('mex_amgx', 'mexAMGxVectorUploadX', self.x_initial);
      end
      calllib('mex_amgx', 'mexAMGxSolverSolve');
      x = calllib('mex_amgx', 'mexAMGxVectorDownloadX');
    end
    function replace(self, A)
      calllib('mex_amgx', 'mexAMGxMatrixReplaceCoeffA', A');
    end
    function initial(self, x)
      self.x_initial = x;
    end
    function delete(self)
      disp('mexAMGx: finalize');
      calllib('mex_amgx','mexAMGxFinalize');
      unloadlibrary('mex_amgx');
    end
  end
end
