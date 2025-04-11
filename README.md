# PetNet: Multi-Task Learning for Pet Classification & Segmentation

This project explores a multi-task learning approach to simultaneously tackle hierarchical image classification and semantic segmentation. It is built around a dataset of pet images labeled with species (cat or dog), breed (37 total breeds), and pixel-level segmentation masks (foreground, background, boundary).

## Tasks
The model jointly learns three interconnected tasks:

- Species Classification
- Predict whether the pet in the image is a cat or a dog.
- Breed Classification
- Predict the top-3 most probable breeds, conditioned on the predicted species.
- Semantic Segmentation
Predict a per-pixel segmentation mask with three classes:
0 = foreground (pet)
1 = background
2 = boundary

  
## Model Architecture

The model consists of a shared CNN backbone and three separate heads:

- Backbone: 5 stacked convolutional blocks that compress the image to a compact representation.
- Segmentation Head: Upsampling using 5 transposed convolutional blocks to recover the original resolution.
- Species Classifier: A small MLP that outputs the species probability.
- Breed Classifier: A second MLP that outputs breed probabilities.
- The loss function is the sum of the individual task losses:
total_loss = segmentation_loss + species_loss + breed_loss

## Dataset

The dataset contains:

RGB images of cats and dogs (resized to 128x128)
Species and breed labels
Semantic segmentation masks with pixel-wise annotations

## Performance

Species Classification	Accuracy	96%
Breed Classification	Top-3 Accuracy	87%
Segmentation	Mean IoU	0.74
Min IoU	0.54

## Inference

The model exposes a simple interface through the predict method in model.py. Given a single preprocessed image (3 x 128 x 128), it returns:

species: a string, either 'cat' or 'dog'
breed: a tuple of top-3 predicted breed names (case-sensitive)
mask: a tensor of shape 128x128 with values {0, 1, 2}

# Load model
model.load_state_dict(torch.load("weights.pth", map_location=torch.device('cpu')))

# Predict
output = model.predict(image_tensor)
