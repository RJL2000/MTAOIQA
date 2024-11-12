# Multitask Auxiliary Network for Perceptual Quality Assessment of Non-Uniformly Distorted Omnidirectional Images

[Jiebin Yan], [Jiale Rao], [Junjie Chen], [Ziwen Tan], [Yuming Fang]

Omnidirectional image quality assessment (OIQA) has been widely investigated in the past few years and achieved much success. However, most of existing studies are dedicated to solve the uniform distortion problem in OIQA, which has a natural gap with the non-uniform distortion problem, and their ability in capturing non-uniform distortion is far from satisfactory. To narrow this gap, in this paper, we propose a multitask auxiliary network for non-uniformly distorted omnidirectional images, where the parameters are optimized by jointly training the main task and other auxiliary tasks. The proposed network mainly consists of three parts: a backbone for extracting multiscale features from the viewport sequence, a multitask feature selection module for dynamically allocating specific features to different tasks, and auxiliary sub-networks for guiding the proposed model to capture local distortion and global quality change. Extensive experiments conducted on two large-scale OIQA databases demonstrate that the proposed model outperforms other state-of-the-art OIQA metrics, and these auxiliary sub-networks contribute to improve the performance of the proposed model.
