## Population-wise personalization to generate 2D synthetic myocardial infarct images

The Python notebook performs the population-wise personalization of simple geometrical models of myocardial infarct, providing as output a set of 2D synthetic images whose distribution matches the distribution of real images.

The current code is applied to the following models:
- elliptical, represented by the intersection of one ellipse with the myocardium,
- iterative spherical, represented by the union of a random number of spheres intersected with the myocardium,
- diffusion-based, mimicking the infarct front propagation from the endocardium.

Personalization is done by a learning process that optimizes the parameters of the models, with the algorithm CMA-ES (Covariance Matrix Adaptation - Evolution Strategy).

### Reference

If you decide to re-use this code, please acknowledge the following publications:

Regarding the detailed evaluation of elliptical and iterative personalizations (choice of losses, hyperparameters, initial values):

*Konik A, Clarysse P, Duchateau N. Detailed evaluation of a population-wise personalization approach to generate synthetic myocardial infarct images. Pattern Recognition Letters. 2025;188:8-14.*

Regarding the diffusion-based model of infarct:

*Konik A, Deleat-Besson R, Clarysse P, Duchateau N. Population-based personalization of a 2D diffusion-based model of myocardial infarct. Proc. International Conference on Functional Imaging and Modeling of the Heart (FIMH), LNCS 2025. In press.*

