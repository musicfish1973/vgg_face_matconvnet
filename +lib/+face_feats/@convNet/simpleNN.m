%  Copyright (c) 2015, Omkar Parkhi
%  All rights reserved.

function feats = simpleNN(obj,img)

img = single(img);

img = cat(3,img(:,:,1)-obj.net.normalization.averageImage(1),...
    img(:,:,2)-obj.net.normalization.averageImage(2),...
    img(:,:,3)-obj.net.normalization.averageImage(3));

if obj.useGPU
    img = gpuArray(img);
end

n = numel(obj.net.layers) ;
res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;



res(1).x = img ;

for i=1:n
    l = obj.net.layers{i} ;
    res(i).time = tic ;
    switch l.type
        case 'conv'
            res(i+1).x = obj.vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                'pad', l.pad, 'stride', l.stride, ...
                obj.cudnn{:}) ;
        case 'pool'
            res(i+1).x = obj.vl_nnpool(res(i).x, l.pool, ...
                'pad', l.pad, 'stride', l.stride, ...
                'method', l.method, ...
                obj.cudnn{:}) ;
        case 'relu'
            res(i+1).x = obj.vl_nnrelu(res(i).x) ;
        case 'dropout'
            res(i+1).x = res(i).x ;
        case 'softmax'
            res(i+1).x = obj.vl_nnsoftmax(res(i).x) ;
        otherwise
            error('Unknown layer type %s', l.type) ;
    end
    % optionally forget intermediate results
    forget = true;
    forget = forget & (strcmp(l.type, 'relu')) ;
    forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
    if forget
        fprintf('forget layer %d\n',i);
        res(i).x = [] ;
    end
    res(i).time = toc(res(i).time) ;
end
%size(res(end-2).x)
feats = squeeze(res(end-2).x);
%size(feats)
end
