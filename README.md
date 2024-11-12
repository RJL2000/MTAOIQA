# Multitask Auxiliary Network for Perceptual Quality Assessment of Non-Uniformly Distorted Omnidirectional Images

[Jiebin Yan], [Jiale Rao], [Junjie Chen], [Ziwen Tan], [Yuming Fang]

## Abstract
Omnidirectional image quality assessment (OIQA) has been widely investigated in the past few years and achieved much success. However, most of existing studies are dedicated to solve the uniform distortion problem in OIQA, which has a natural gap with the non-uniform distortion problem, and their ability in capturing non-uniform distortion is far from satisfactory. To narrow this gap, in this paper, we propose a multitask auxiliary network for non-uniformly distorted omnidirectional images, where the parameters are optimized by jointly training the main task and other auxiliary tasks. The proposed network mainly consists of three parts: a backbone for extracting multiscale features from the viewport sequence, a multitask feature selection module for dynamically allocating specific features to different tasks, and auxiliary sub-networks for guiding the proposed model to capture local distortion and global quality change. Extensive experiments conducted on two large-scale OIQA databases demonstrate that the proposed model outperforms other state-of-the-art OIQA metrics, and these auxiliary sub-networks contribute to improve the performance of the proposed model.

## Model Architecture

### Training MTAOIQA
Then, you can select the corresponding training and test files under the file/JUFE-10K folder.

Modify the configuration in code/MTAOIQA_config.py

- Modify "dataset_name" to choose which datasets you want to train in config
- Modify training and test dataset path

```
sh mtaoiqa_train_run.sh
```

## Citation
**If you find this code is useful for  your research, please cite**:

```latex
@article{yan2024mtaoiqa,
  title={Multitask Auxiliary Network for Perceptual Quality Assessment of Non-Uniformly Distorted Omnidirectional Images},
  author={Yan, Jiebin and Rao, Jiale and Chen, Junjie and Tan, Ziwen and Fang, Yuming},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024}
}
```

## Acknowledgement
1. <https://github.com/RJL2000/OIQAND>
2. <https://github.com/WenJuing/IQCaption360>
