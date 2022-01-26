# Single image dehazing via cycle-consistentadversarial networks with a multi-scale hybridencoder-decoder and global correlation loss

we propose novel cycle-consistent adversarial networks with a multi-scale hybrid encoder-decoder and global correlation loss for single image dehazing task. The requirement of paired training data is eliminated by combining two generators and discriminators into a cycle-consistent adversarial network.

## Architecture of generators network with a multi-scale hybrid encoder-decoder.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/frame.jpg)

## The detailed architecture of hybrid-encoding block.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/block_frame.png)

## Results

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/NYU.png)

*Fig. 1. Comparison of qualitative results on the NYU-Depth dataset.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/Midd.png)

*Fig. 2. Comparison of qualitative results on the Middlebury dataset.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/indoor.png)

*Fig. 3. Comparison of qualitative results on the Indoor SOTS dataset.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/OHAZE.png)

*Fig. 4. Comparison of qualitative results on the O-HAZE dataset.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/sea-fog.jpg)

*Fig. 5. Comparison of qualitative results on the Sea-fog dataset.

![](https://github.com/libiezhiwen/cycle-image-dehazing/blob/master/image/real.jpg)

*Fig. 6. Comparison of qualitative results on the real-world images.

## requirements

torch>=0.4.1
torchvision>=0.2.1
dominate>=2.3.1
visdom>=0.1.8.3

## Citation 
If you find this code useful, please cite:

Yao et al., Single image dehazing via cycle-consistentadversarial networks with a multi-scale hybridencoder-decoder and global correlation loss.

