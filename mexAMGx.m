classdef mexAMGx < handle
  properties (Access = private)
    x_initial = [];
  end
  methods
    function self = mexAMGx(A, cfg)
      warn = warning('off', 'all');
      loadlibrary('mex_amgx', @mex_amgx_proto);
      warning(warn);
      if isstruct(cfg)
        cfg_str = self.parse_json(cfg);
        calllib('mex_amgx', 'mexAMGxInitialize', ...
          cfg_str, false);
      else
        amgx_path = fileparts(which('mexAMGx'));
        curr_path = pwd; cd(amgx_path);

        calllib('mex_amgx', 'mexAMGxInitialize', ...
          fullfile(amgx_path, 'configs', cfg), true);
        cd(curr_path);
      end
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
  methods (Access = private, Static)
    function str = parse_json(cfg, varargin)

      flds = fields(cfg);
      if nargin > 1,  str = []; else str = '{'; end

      for flds_cnt = 1:length(flds)
        if flds_cnt == 1, str = [str '"']; else str = [str ', "']; end
        str = [str flds{flds_cnt} '" : '];
        if isstruct(getfield(cfg, flds{flds_cnt}))
          str = [str '{' parse_json(getfield(cfg, flds{flds_cnt}), true) '}'];
        elseif isnumeric(getfield(cfg, flds{flds_cnt}))
          str = [str num2str(getfield(cfg, flds{flds_cnt}))];
        else
          str = [str '"' getfield(cfg,flds{flds_cnt}) '"'];
        end
      end

      if nargin == 1,  str = [str '}']; end
    end
  end
end
