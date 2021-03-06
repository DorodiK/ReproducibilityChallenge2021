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
both statistical and structural styles.![145637729-f40826c3-7a7c-4352-923e-b24342b4be40]

### Code 

The code for the [AdaConv module ](https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/adaconv/adaconv.py/) and the [Kernel predictors](https://github.com/RElbers/ada-conv-pytorch/blob/master/lib/adaconv/kernel_predictor.py/) was reused from an unofficial imeplementation by [REIbars](https://github.com/RElbers/ada-conv-pytorch).


The code for AdaIN is also reused from the [Torch implementation with python](https://github.com/naoto0804/pytorch-AdaIN) by [naoto0804](https://github.com/naoto0804/pytorch-AdaIN) of the original [Torch implementation with lua](https://github.com/xunhuang1995/AdaIN-style) by the [authors](https://github.com/xunhuang1995/AdaIN-style). 

Listed code repositories are MIT-licensed as open-source for academic research. 

### Dataset 

We used the pretrained weights of the AdaConv framework provided in the open-source repo. The weights were obtained by the author of the 
AdaConv code using the following datasets. 

Content - Click [here](https://cocodataset.org/#home) to go to official COCO dataset website. 

Style - Click [here](https://www.kaggle.com/antoinegruson/-wikiart-all-images-120k-link) to go to official Wiki-Art dataset on Kaggle.

Pretrained weights - Click [here](https://drive.google.com/file/d/17h-Hd08n-f_5D8cDV08dpB_-W1cs5jbt/view?usp=sharing) to download pretrained-weights to run experiments 

### Method

By implementing the codes linked above we investigated the results and claims of the paper [1] in three ways: 
1. Outputted visuals of the style latent features across the different layers of the encoder-decoder architecture of AdaConv along with the convolution results
3. Compared AdaIN and AdaConv results on images and styles in test_images using different techniques (e.g. change in orientation of style image). 
4. Implemented AdaConv to raw content image taken on our phones. 

### Results 

#### 1. AdaConv Framework Visuals
![style_01](https://user-images.githubusercontent.com/46634299/145637685-427deeeb-d769-48e9-a574-cb01767cc8a4.jpg)

A sample style image

![01_1](https://user-images.githubusercontent.com/46634299/145637729-f40826c3-7a7c-4352-923e-b24342b4be40.png)

The output of the VGG-19 network for the sample style image

![style_01](https://user-images.githubusercontent.com/46634299/145637812-6eb8be33-2d8c-4334-881b-ab6bbf675177.png)

The output of style encoder for the mentioned image

![K1](https://user-images.githubusercontent.com/38030229/145616390-4e1e3f05-d896-4ddf-8ec7-78b2b81b585b.png "K1") ![K2](https://user-images.githubusercontent.com/38030229/145616396-3f42f37c-ee52-4c96-80c4-9f19d5f48362.png "K2") ![K3](https://user-images.githubusercontent.com/38030229/145616404-9b924538-c9c7-4adf-8341-3470a1e8f35f.png "K3") ![K4](https://user-images.githubusercontent.com/38030229/145616408-d72c02b6-a482-4f8d-815c-7a0619db5929.png "K4") ![Final](https://user-images.githubusercontent.com/38030229/145616415-d9a945d6-d62b-40c3-bc1f-7319563d2355.png "Final")

Fig: Visuals from the AdaConv framework when predicted kernels are convolved (e.g. four kernels) with latent features of the content image at different levels of the decoder that results in the final output image. 


#### 2. AdaConv and AdaIN 

The following results were obtained by using different orientation of the style image on the same content image.

![AdaIN output](https://user-images.githubusercontent.com/38030229/145614967-d0a0c8a1-9333-4bfe-a22b-2f8dd595d486.gif "AdaIN output") 

Fig: AdaIN output as gif

![AdaConv output](https://user-images.githubusercontent.com/38030229/145614802-15588244-4be2-44c7-8d0d-9f921ecad4f9.gif "AdaConv output ")

Fig: AdaConv output as gif



#### 3. AdaConv 
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

### Acknowledgement 

We would like to thank the authors of the codes for making their codes open-source for academic research. 

### Reference 
1. Prashanth Chandran, Gaspard Zoss, Paulo Gotardo, Markus Gross, and Derek Bradley. Adaptive convolutions
for structure-aware style transfer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 7972–7981, 2021. [Link](https://openaccess.thecvf.com/content/CVPR2021/html/Chandran_Adaptive_Convolutions_for_Structure-Aware_Style_Transfer_CVPR_2021_paper.html) 
