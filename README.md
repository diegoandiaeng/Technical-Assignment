The repository includes:
- Notebook for training the PointNet architecture used for field prediction PointNet training: PointNet_training.ipynb
- Notebook for training the Scalar Head using the pre-trained PointNet architecture (global encoder): PointNetScalar_training.ipynb
- Notebook for unsupervised ML techniques using the econded representation: encoder_study.ipynb 
- Notebook for an interactive UI for testing to load and show deformed component (intended for presentation): testing_results.ipynb
- Data folder with weight data including .pth files of the trained models, and .csv with geometry code and global values (max stress, 1st freq, mass).
  The point cloud data is a heavy file but it can be found and dowloaded in the following link https://drive.google.com/file/d/1bPaPRwL8Eu5IJuFOIRYBZSaDIoCUSqeB/view?usp=sharing
  it needs to be added inside the Data folder.


The PointNet architecture is based on the paper (https://arxiv.org/abs/2412.18362) where this architecture is benchmarked on the DeepJeb dataset:

<img width="1048" height="507" alt="image" src="https://github.com/user-attachments/assets/d454f421-667b-47ba-864e-92408d1a71fa" />

For the technical assignment the following modifications are done:
- Only the displacements/stress of the diagonal loading case are considered.
- The load and mass are not used as input only the point coordinates.
- An additional MLP is trained from the output of the global representation brach to predict 3 additional global scalars.

<img width="1432" height="582" alt="image" src="https://github.com/user-attachments/assets/496bf3bf-32ec-4eaa-bdc7-52b06fbc6bd1" />



Clarification regarding the 1D Convolutions: The kernel size used is 1 since the kernel slides point by point (points are unordered), the input features correspond to the channels (like RGB for an image).





Additional results:

<img width="1806" height="801" alt="image" src="https://github.com/user-attachments/assets/f6538e7c-9818-45e5-b2f7-6d5017c373ee" />
<img width="1797" height="751" alt="image" src="https://github.com/user-attachments/assets/42af55f2-5e15-406e-b915-6d0a770a592f" />
<img width="1799" height="706" alt="image" src="https://github.com/user-attachments/assets/ba74bcf6-6c18-4903-888d-4817f5b21470" />
