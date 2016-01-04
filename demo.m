%  Copyright (c) 2015, Omkar M. Parkhi
%  All rights reserved.

function sim = demo(fimg0,fimg1,convNet)
%config.paths.net_path = 'data/vgg_face.mat';
%convNet = lib.face_feats.convNet(config.paths.net_path);

%img = imread('ak.jpg');
%img = imread('ak.png');
%det = faceDet.detect(img);
%crop = lib.face_proc.faceCrop.crop(img,det(1:4,1));
%[score,class] = max(convNet.simpleNN(crop));
%[score,class] = max(convNet.simpleNN(img));

%face verification
img0 = imread(fimg0);
img1 = imread(fimg1);
feat0 = convNet.simpleNN(img0);
feat1 = convNet.simpleNN(img1);
sim = 1 - pdist([feat0';feat1'],'cosine');

end