Python 2.7 -- Cuda-8.0 -- TensorFlow 1.4.0 -- Keras 2.1.5

# 3D Segmentation with Adversarial Networks

This project is inspired in the 3 papers below with implementation in Keras, used in a 3D dataset. The application is the segmentation of the cerebrovascular system in MRA phase contrastusing images.

- [SegAN: Adversarial Network with Multi-scale L1 Loss for Medical Image Segmentation](https://arxiv.org/pdf/1706.01805.pdf) by Yuan Xue, Tao Xu, Han Zhang, L. Rodney Long, Xiaolei Huang.
- [Adversarial Learning with Multi-Scale Loss for Skin Lesion Segmentation](http://www.cse.lehigh.edu/~huang/ISBI_Paper2018.pdf) by Yuan Xue, Tao Xu, Xiaolei Huang.
- [Semantic Segmentation using Adversarial Networks](https://arxiv.org/pdf/1611.08408.pdf) by Pauline Luc, Camille Couprie, Soumith Chintala, Jakob Verbeek.

In the segmentor part of the network, I am also using a dice score as a mixed loss function.
