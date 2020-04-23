# MAML-Tensorflow
Tensorflow r2.1 reimplementation of Model-Agnostic Meta-Learning from this paper: 

[Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

Reinforcement Learning part is taken from [MoritzTaylor/maml-rl-tf2](https://github.com/MoritzTaylor/maml-rl-tf2) Sincerely Thanks for his fantasic work!!!

## Project Requirements

1. python 3.x
2. Tensorflow r2.1
3. numpy 
4. matplotlib
5. ...

## MiniImagenet Dataset

I wrote a task generator which samples randomly from the whole dataset to set up a train batch during every training steps so it won't consume too much GPU memory.

For 5-way 1-shot tasks on the MiniImagenet, it takes at round 1.6 s to run one training steps on the GTX1070 and for each task update fast_weights 1 time, and it allocate 1.3GB GPU memory.

For 5-way 5-shot tasks on MiniImagenet, it takes at around 2.2 s to run one training steps.

If you set the `--update_steps > 1`, it will take more time for one training step.



1. Download the [MiniImagenet](https://drive.google.com/open?id=1HkgrkAwukzEZA0TpO7010PkAOREb2Nuk), and the split files `train.csv, test.csv, val.csv` from [here](https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet)

2. Put the MiniImagenet dataset to your project folder like this

   ```
   dataset/miniimagenet
   |-- images
   	|--- n0153282900000005.jpg
   	|--- n0153282900000006.jpg
   	...
   |-- test.csv
   |-- train.csv
   |-- val.csv
   ```

3. Run the python script to resize and split the whole dataset 

   ```
   cd scripts/image_classification
   python image_preprocess.py --dataset=miniimagenet
   ```

4. Modify the path to dataset in `scripts/image_classification/task_generator.py`

   ```python
   if self.dataset == 'miniimagenet':
       ...
       META_TRAIN_DIR = '../../dataset/miniImagenet/train'
       META_VAL_DIR = '../../dataset/miniImagenet/test'
       ...
   
   if self.dataset == 'omniglot':
       ...
           DATA_FOLDER = '../../dataset/omniglot'
           ... 
   ```
   
   
   
5. Run the main python script

   ```
   cd scripts/image_classification
   # For 5-way 1-shot on miniimagenet
   python main.py --dataset=miniimagenet --mode=train --n_way=5 --k_shot=1 --k_query=15
   # For 5-way 5-shot on miniimagenet
   python main.py --dataset=miniimagenet --mode=train --n_way=5 --k_shot=5 --k_query=15
   ```

   

## Omniglot Dataset

For Omniglot dataset, it will consume fewer computing resource and time

For 5-way 1-shot, 0.3 s for one training step

For 20-way 1-shot, 0.7 s for one training step

1. Download Omniglot dataset from [here](git clone git@github.com:brendenlake/omniglot.git) and extract the contents of `python/images_background.zip` and `python/images_evaluation.zip` to the `dataset/omniglot` it will looks like this:

   ```
   dataset/omniglot
   |-- Alphabet_of_the_Magi
   |-- Angelic
   ...
   ```

2. Run the python script to resize the images

   ```
   cd scripts/image_classification
   python image_preprocess.py --dataset=omniglot
   ```

3. Run the main python script 

   ```
   cd scripts/image_classification
   # For 5-way 1-shot on Omniglot
   python main.py --dataset=omniglot --mode=train --n_way=5 --k_shot=1 --k_query=1 --inner_lr=0.1
   # For 20-way 1-shot on Omniglot
   python main.py --dataset=omniglot --mode=train --n_way=20 --k_shot=1 --k_query=1 --inner_lr=0.1
   ```

   

## References

This project is, for the most part, a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in TensorFlow 2. The experiments are based on the paper

> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep
> networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]

If you want to cite this paper

```
@article{DBLP:journals/corr/FinnAL17,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {Model-{A}gnostic {M}eta-{L}earning for {F}ast {A}daptation of {D}eep {N}etworks},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```