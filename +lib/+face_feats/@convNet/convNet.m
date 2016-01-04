%  Copyright (c) 2015, Omkar Parkhi
%  All rights reserved.

classdef convNet < handle


    properties
        net
        useGPU
        cudnn;
    end

    methods

      function obj = convNet(netWorkPath)
      	temp = load(netWorkPath);
        %temp.net.layers = temp.net.layers(1:end-2);
      	obj.net = temp.net;
        obj.useGPU = false;
        obj.cudnn = {'NoCuDNN'} ; % If using CuDNN {'CuDNN'}
      end

      feat = simpleNN(obj,img);

    end
    methods(Static)
    	x = vl_nnconv(x,f,b,varargin);
    	x = vl_nnpool(x,pool,varargin);
    	x = vl_nnrelu(x);
        x = vl_nnsoftmax(x);
    end
end
