# vgg_face_matconvnet
VGG Face descriptor source code and models (MatConvNet)

This is my starting point for learning NN & DL.I would like to share learning experiences with the great Github community.

# Step by step
1.download and install "MatConvNet: CNNs for MATLAB",see http://www.vlfeat.org/matconvnet/
  I use version 1.0-beta17.
2.download this vgg_face_matconvnet and run it in MATLAB
  I use MATLAB 2014b.
  Pretrained model vgg_face.mat (1GB) is not contained in my repository.Just use the one in "http://www.robots.ox.ac.uk/~vgg/software/vgg_face/".
  In MATLAB:
  >> convNet = lib.face_feats.convNet('data\vgg_face.mat');
  >> demo('1.jpg','2.jpg',convNet)
  The result is the cosine similarity of two faces.


# Acknowdgements
Thanks to "http://www.robots.ox.ac.uk/~vgg/software/vgg_face/",with which I have had the first intimate touch with NN & DL.