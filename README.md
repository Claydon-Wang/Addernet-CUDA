## Training addernet accelerated by CUDA

### The original <a href="https://arxiv.org/abs/1912.13200">AdderNet Repo</a> considers using PyTorch for implementing add absed convolution, however it remains slow and requires much more runtime memory costs as compared to the variant with CUDA acceleration.

### This repository is partially referenced to <a href="https://arxiv.org/abs/2010.12785">shiftaddernet</a>.


You can ompile the Folder of "adder" to obtain CUDA vision of "adder2D", which can replace "Conv2D" with more efficient use of hardware.<br>

"adder2D-CUDA" can compress over ~10x traning time than non-CUDA version of "adder2D".

