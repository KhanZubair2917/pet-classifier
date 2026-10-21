# Pet Breed Classifier

Classifying 37 cat and dog breeds from photos using transfer learning with a pretrained ResNet18 in PyTorch.

## Dataset

Oxford-IIIT Pets, 3,680 training photos and 3,669 test photos across 37 breeds. Images were resized to 224x224 and normalized to match what the pretrained model expects.

## What I Did

- Loaded a ResNet18 pretrained on over a million images and froze all its layers
- Replaced the final layer with a new one that outputs 37 breeds, so only 18,981 of 11,195,493 parameters were trained
- Trained for 5 epochs with Adam on a Colab GPU
- Evaluated on the full test set and analyzed which breeds the model confuses

## Results

87% accuracy on 3,669 unseen photos, compared to about 2.7% for random guessing across 37 breeds.

## Key Findings

The biggest mix-ups were between look-alike breeds: American Pit Bull Terrier vs Staffordshire Bull Terrier (24 times), American Pit Bull Terrier vs American Bulldog (19), Birman vs Ragdoll (18), and British Shorthair vs Russian Blue (13). The top 5 confusions account for about 18% of the 485 total mistakes, so the rest are spread across many pairs.

## Next Steps

Fine-tune more layers of the model, train for more epochs, and add data augmentation.

## Tools Used

Python, PyTorch, torchvision, Google Colab