# ReproducibilityChallenge2021
SYDE 671 - Final Report and Code 


## Adaptive Convolutions for Structure-Aware Style Transfer 

### Summary 

The following reproducibility challenge is based on the paper titled ’Adaptive Convolutions for Structure Aware Style
Transfer’ proposed by Prashanth et al., and Disney Research [1]. In essence, the investigated paper introduces Adaptive
Convolutions (AdaConv) as an extension of Adaptive Instance Normalization (AdaIN), that focuses on the neural
style transfer between images as an artistic application of CNNs, where given a content and style image the goal is to
synthesize an output image which has the high level structure of the content image and the artistic style as the style
image. As such the outcome of the proposed Adaptive Convolutions framework allows for the simultaneous transfer of
both statistical and structural styles.

### Code 

The code for the [AdaConv module ](https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/adaconv/adaconv.py/) and the [Kernel predictors](https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/adaconv/kernel_predictor.py/) was reused from the unofficial imeplementation by REIbars, 
which is MIT-licensed as open-source and for academic research. 

The code for AdaIN is also reused from the [Torch implementation with python](https://github.com/naoto0804/pytorch-AdaIN) of the original [Torch implementation with lua](https://github.com/xunhuang1995/AdaIN-style) by the authors. 

### Dataset 

Content - Click [here](https://cocodataset.org/#home) to go to official COCO dataset website. 

Style - Click [here](https://www.kaggle.com/antoinegruson/-wikiart-all-images-120k-link) to go to official Wiki-Art dataset on Kaggle.

### Method

By implementing the codes linked above we investigated the results and claims of the paper [1] in three ways: 
1. Outputted visuals of the content and style latent features across the different layers of the encoder-decoder architecture of AdaConv 
2. Compared AdaIN and AdaConv results on images and styles in test_images using different techniques (e.g. change in orientation of style image). 
3. Implemented AdaConv to raw content image taken on our phones. 

### Results 
#### AdaConv and AdaIN 
The following results were obtained for a qualitative analysis between the two frameworks, using different style image from 
test_images. 

!![AdaIN output](https://user-images.githubusercontent.com/38030229/145604938-059454fe-377e-4826-8b62-66d7ea34238c.jpg "AdaIN output") ![AdaConv output](https://user-images.githubusercontent.com/38030229/145605309-a107dcc5-809d-4ffd-9bd2-657ba5246d6d.png "AdaConv output ")

Fig: AdaIN output (left) and AdaConv output (right)

#### AdaConv 
The following results were obtained using images in the test_images folder. The content image is that of an engineering building at the University
of Waterloo clicked on our phone. Different style images were then applied to this content image using AdaConv method to synthesize the 
following output images. 

![content](https://user-images.githubusercontent.com/38030229/145607240-37b904ca-d39c-4356-beb0-c075c2ecce7c.jpg)

Fig: University of Waterloo as Content image

![alt-text-1](https://user-images.githubusercontent.com/38030229/145606791-d628e160-c6ad-4de2-baa9-7414c2e4df2a.jpg "title-1") ![alt-text-2](https://user-images.githubusercontent.com/38030229/145607598-e00dd4e2-baba-4ad0-b81e-4302d992d949.jpg "title-2") 

Fig: Style images

![alt-text-1](https://user-images.githubusercontent.com/38030229/145606927-da518a2a-8b53-49e2-a1f5-9b26d471bd40.png "title-1") ![alt-text-2](https://user-images.githubusercontent.com/38030229/145607670-019de6bd-7dca-4a58-bc92-a39aa6ecad8d.png "title-2") 

Fig: Output images 

### Discussion 

We analysed the images qualitatively, even though it can be subjective, one can see how well the style transfer framework of AdaConv 
simultaneously captured the statistical and structural features of the style image while preserving high level features of the content image, 
this corroborates one of the authors' claims in the paper that AdaConv improves the transfer of both statistical and structural style features in real time compared to AdaIN.


### Reference 
1. Prashanth Chandran, Gaspard Zoss, Paulo Gotardo, Markus Gross, and Derek Bradley. Adaptive convolutions
for structure-aware style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 7972–7981, 2021. [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.html) 
