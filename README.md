# MIA

Unofficial Pytorch implementation of paper: Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures

## NEW!
Code of my latest work on MIA "Z. Zhang, X. Wang, J. Huang and S. Zhang, "Analysis and Utilization of Hidden Information in Model Inversion Attacks," in IEEE Transactions on Information Forensics and Security, doi: 10.1109/TIFS.2023.3295942." is available at: https://github.com/zhangzp9970/Amplified-MIA

## Description

This this an **unofficial** pytorch implementation of paper: Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. In *Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security*  (*CCS '15*). Association for Computing Machinery, New York, NY, USA, 1322–1333. DOI:https://doi.org/10.1145/2810103.2813677

The official code from the author can be found at: https://www.cs.cmu.edu/~mfredrik/. It seems that the page sometimes doesn't load well.

The results from the original author can be found at https://github.com/mfredrik/facematch

## Usage

The repository contains both the code for attacking the logistic regression network and the multi-layer perception network with a hidden layer of 3000 neurals. Both networks are described in the paper. The code is written in PyTorch along with the [torchplus ](https://github.com/zhangzp9970/torchplus)toolkit library developed by me. It can be easily run both on CPU and GPU. GPU is prefered for better performance.

Change `mynet = Net(h*w, class_num).to(output_device).train(True)` to `mynet = MLP(h*w, class_num).to(output_device).train(True)` to change the network.

Download the dataset, install the dependences (below), Click run in any editor.

* main.py -- train the target network
* attack.py -- perform model inversion attack

## Third-party libraries

Anaconda is prefered and is the easiest way. Install Anaconda and install pytorch using the command in the official pytorch page. Then install torchplus using `conda install torchplus -c zhangzp9970`. Everything will be ok then.

* pytorch >= 1.8.1
* torchvision
* [torchplus](https://github.com/zhangzp9970/torchplus)
* tqdm

## Differents

Unlike the author, I use 8 images as a minibatch to train the network. From my perspective, it is not a good idea to train the network with only 1 images in each minibatch. The batch size can be any number less than the dataset size.

In attack.py, instead of Gradient Descent, SGD is used to achieve a high performance on gradient descent. See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD for more information.

In attack.py, torch.nn.CrossEntropyLoss is used as the cost function for better results, which still have the same meaning as the original

Denoise techniques such as ZCA are not implemented, so the results have little noisy

## Results

![figure](./MIA.svg)

## License

Copyright © 2021-2023 Zeping Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).
