# Introduction
Computer vision communities have produced multiple successful techniques to stylize photorealistic images. However, the reversed image translation problem: generate photorealistic images from paintings, is not as widely explored.
Generating photorealistic images from paintings is an interesting yet challenging task, due to the abstract nature in painting content, and the demanding details to make an image look photo-realistic. Although there are already some applications that explores the possibility of translating a Monet painting to a photo using CycleGAN, these techniques do not have a good understanding of the overall scene structure; they are only able to modify pixel colour instead of generating new realistic objects, thus requiring many further improvements in order to achieve successful results.
In this project, we present an approach that understands the semantic layout of a painted scene and utilizes such scene segmentation to generate a corresponding photorealistic image. Our contributions include: (i) a synthesized painting training dataset with segmentation labels, (ii) the demonstration of training HRNetV2 with and without OCR (Object Contextual Representations), and (iii) an additional discriminator in GuaGAN to distinguish photo-realistic images.

# Reference: GAU-GAN from NVIDIA
T. Park, M.-Y. Liu, T.-C. Wang, and J.-Y. Zhu. Semantic image synthesis with spatially-adaptive normalization. 2019.

# Sample Train and Test command
Run train.sh to train:\
Here is the sample command for training:\
python train.py --name orig_train --dataset_mode ade20k --dataroot ./datasets/ADEChallengeData2016

Add --continue_train to continue from last checkpoint.

Run test.sh to test:\
Here is the sample command for testing:\
python test.py --name orig_train --dataset_mode ade20k --dataroot ./datasets/vangogh_real

# Sample model can be downloaded from here
https://drive.google.com/drive/folders/10WlOkkZqJ9wiL7oa3wiFpZP5hzrVIGAd?usp=sharing
