# Automatic Licence Plate Recognition

In February, 2017 I was selected as team leader for my senior design project. The problem was to design a system that could receive an image of a license plate, extract the registration numbers off of the plate and recognize the characters. This was a research project that was to include the use of neural networks for character classification. We decided to use Python 3.6, and experimented with Theano/Keras and scikit-learn for our machine learning libraries.

## Files

What is posted in this repository is a code sample from this project.

### alpr_som.py

This class represents a experimental implementation of a neuron gas self-organizing map. Although it was a successful implementation, we abandoned it to find a faster method.

### rectfinder.py

This is the character segmentation algorithm we designed. It essentially is a flood fill algorithm that tracks the boundaries of the areas being flooded. It analyzes each cluster of pixels and either accepts or rejects it as a meaningful region. It then extracts the accepted regions into a list of subimages. In order for this algorithm to be successful, the license plate must be preprocessed in a specific way - namely luminous grayscale, and inverse rounding. This is mentioned in a conference paper which we are in the process of writing about this project.
