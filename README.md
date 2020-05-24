# modified_GAU_GAN

# Reference: GAU-GAN from NVIDIA
# T. Park, M.-Y. Liu, T.-C. Wang, and J.-Y. Zhu. Semantic image synthesis with spatially-adaptive normalization. 2019.


Run train.sh to train:
Here is the sample command for training:
python train.py --name orig_train --dataset_mode ade20k --dataroot ./datasets/ADEChallengeData2016

Add --continue_train to continue from last checkpoint.

Run test.sh to test:
Here is the sample command for testing:
python test.py --name orig_train --dataset_mode ade20k --dataroot ./datasets/vangogh_real
