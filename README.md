## 2D personalization to generate infarct images

The Python notebook provides the generation of 2D images with infarcts with two simple models:
- elliptical, represented by the intersection of one ellipse with the myocardium,
- iterative spherical, represented by the random number of spheres intersected with the myocardium.

The learning process of parameters of the models is presented by the optimization algorithm CMA-ES.

The detailed evaluation of such generations (the choice of losses, hyperparameters, initial values) was accepted for the publication in Pattern Recognition Letters.
