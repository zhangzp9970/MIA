# MIA
Unofficial Pytorch implementation of paper: Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures

## Description
This this an <b>unofficial</b> pytorch implementation of paper: Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. 2015. Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. In <i>Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security</i> (<i>CCS '15</i>). Association for Computing Machinery, New York, NY, USA, 1322–1333. DOI:https://doi.org/10.1145/2810103.2813677

The official code from the author can be found at: https://www.cs.cmu.edu/~mfredrik/

The results from the original author can be found at https://github.com/mfredrik/facematch

## Usage
* main.py -- train the target network
* test.py -- test the target network
* attack.py -- perform model inversion attack

## Differents
In attack.py, instead of Gradient Descent, SGD is used to achieve a high performance on gradient descent. See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD for more information.

In attack.py, torch.nn.CrossEntropyLoss is used as the cost function for better results, which still have the same meaning as the original

Denoise techniques are not implemented, so the results have little noisy

## Results
![figure](./MIA.png)

## License

Copyright © 2021 Zeping Zhang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.