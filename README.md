# Self-Driving-Car
 The project is about implementing a self-driving car via behavioural cloning using Nvidia deep learning architecture.

## Table of Contents
- [About](#about)
- [Introduction](#introduction)
- [Installation](#installation)
- [Features](#features)
- [Contributing](#contributing)
- [FAQ](#faq)
- [License](#license)
---
## About
This resporitory provides materials for implementing and training self-driving car via behavior cloning.

---
## Introduction
You may approach the project via the following procedures
- Train to obtain model.h5
    >1: Walking through the <a href="https://github.com/JimYeung/Self-Driving-Car/tree/main/docs" target="_blank">**Behavioural_Cloning.ipynb**</a> with trained model on Google Colab
    /n OR
    /n >2: Train your own model.h5 with <a href="https://github.com/JimYeung/Self-Driving-Car/blob/main/train.py" target="_blank">**train.py**</a> 
- Laucnh your trained model with <a href="https://github.com/JimYeung/Self-Driving-Car/blob/main/Drive.py" target="_blank">**Drive.py**</a> and <a href="https://github.com/udacity/self-driving-car-sim/" target="_blank">**Udacity Cars simulator**</a>

---
## Installation
- GPU is utilized for training in this project. Refer to <a href="https://www.tensorflow.org/install/gpu/" target="_blank">**GPU tensorflow support**</a>for further setup information. 

- Install <a href="https://github.com/udacity/self-driving-car-sim/" target="_blank">**Udacity Cars simulator**</a>

- Cloning the project 
> !git clone https://github.com/JimYeung/Self-Driving-Car.git

- Refer to packages_lists More environment settings (e.g. packages)
---

## Reference
1. Bojarski, M., Del Testa, D., Dworakowski, D., Firner, B., Flepp, B., Goyal, P., Jackel, L.D., Monfort, M., Muller, U., Zhang, J. and Zhang, X., 2016. End to end learning for self-driving cars. arXiv preprint arXiv:1604.07316.

2. Udemy, R. Slim, A. Sharaf, J. Slim, S. Tanveer, 2020. The complete self-driving car course - Applied Deep Learning, Udemy.

## License
PROJECT LICENSE

This project was submitted by Yui Jim Yeung as part of the Nanodegree At Udemy.

Copyright (c) 2020 Yui Jim Yeung

Besides the above notice, the following license applies and this license notice
must be included in all works derived from this project.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
