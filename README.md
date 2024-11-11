# TGCA-PVT 




## SER30K dataset

The SER30K dataset  should be applied  and downloaded at https://github.com/nku-shengzheliu/SER30K.

## Prerequisites

- Python 3.6
- Pytorch 1.10.2
- Others (Pytorch-Bert, etc.) Check requirements.txt for reference.

In addition, please download the ImageNet pre-trained model weights for PVT-small from [PVT](https://github.com/whai362/PVT/tree/v2/classification) and place it in the `./weight` folder.




## Training
To train TGCA-PVT on SER30K on a single node with 2 gpus for 50 epochs run:


```shell
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6666 \
--use_env main.py \
--config configs/pvt/pvt_small.py \
--visfinetune weights/pvt_small.pth \
--output_dir checkpoints/SER \
--dataset SER \
--data-path {path to SER30K dataset} \
--alpha 8 \
--batch-size 16 \
--locals 1 1 1 0
```



## Evaluation
To evaluate TGCA-PVT model performance on SER30K with a single GPU, run the following script using command line:

```shell
python -m torch.distributed.launch --nproc_per_node=1 --master_port=6666 \
--use_env main.py \
--config configs/pvt/pvt_small.py \
--resume checkpoints/SER/checkpoint_best.pth \
--dataset SER \
--data-path {path to SER30K dataset} \
--batch-size 16 \
--alpha 8 \
--locals 1 1 1 0 \
--eval
```

## Reference
Please cite our paper as below:
@inproceedings{chen2024tgca,
  title={TGCA-PVT: Topic-Guided Context-Aware Pyramid Vision Transformer for Sticker Emotion Recognition},
  author={Chen, Jian and Wang, Wei and Hu, Yuzhu and Chen, Junxin and Liu, Han and Hu, Xiping},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={9709--9718},
  year={2024}
}
