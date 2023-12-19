# D4.2: Toolbox for joint processing of geophysical and remote sensing data

### Summary


> Development of a toolbox for the joint processing of geophysical and remote sensing data

## Introduction

This deliverable focuses on establishing a shared repository, equipped with tools and examples for reading, manipulating, and inverting various types of data commonly used during exploration phases. Emphasis has been placed on integrating a selection of code packages, primarily contributed by members of the Vector consortium, along with a new repository developed specifically for the Vector project.

In addition to the code essential for performing data inversion, we have included a comprehensive set of tutorials. These guides lead users through typical workflows, from data reading to final optimization processes. Due to intellectual property constraints, all tutorial examples are anonymized, and the datasets required to replicate these examples are exclusive to the Vector consortium. Efforts are underway to address these limitations, aiming to make all materials described in this deliverable freely accessible in the future.

The design of this toolbox emphasizes flexibility and extensibility, catering to the diverse datasets and scientific inquiries encountered during exploration. Recognizing the impracticality of creating a universally exhaustive toolbox, our approach has been to establish a robust framework. This framework can be expanded upon to accommodate specific use cases as they arise.


## Toolbox Overview

To data, the toolbox is divided into four main sections: readers and parsers, structural modeling, forward geopysics and inversion; and probabilistic modeling.

TODO: Explain a bit the whole concept

## Work Description

### Vector Geology
- [ ] Use of simpeg
-
#### Documentation

### GemPy
- [ ] Torch
    - [ ] Automatic differentiation. Activation function
- [ ] Pykeops
- [ ] Adam optimization
-
  GemPy is a Python-based open-source library for 3D geological modeling. It is designed to facilitate the construction of complex geological models, including the generation of 3D grids, the interpolation of geological contacts and faults, and the simulation of geological processes. GemPy was developed mostly as a PhD student project and was in need of a quite major overhaul to make it robust and reliable in production environments. The Vector project provided the opportunity to do this and hence a lot of the work necessary to produce this deliverable has been done by refactoring and updating the GemPy code base. The main changes are:

- [ ] TODO: **What is gempy exactly** Add small comment about gempy interpolation Gaussian Process and Kriging differences. Add concept of surface points and orientations


#### Updating GemPy


##### Tensor Backends

Something that happened over the life of the GemPy project has been the rise of deep learning and the development of a number of tensor backends for Python. These backends provide a way to do automatic differentiation and hence to do gradient-based optimization. Amidst this, the original Tensor library used by GemPy got deprecated. In order to combat this rapidly changing enviroment, we rearchitect GemPy to relatively easily being able to choose between different tensor backends. Currently, we support numpy, PyTorch and Pykeops. 

The use of PyTorch makes much easier to design complex probabilistic models - including probabilistic inversions - with an easy to use API and with very remarkable error messages while being able to get the gradients of the operations and even run the code on GPUS. Pykeops, on the other hand, is capable to do reductions operations with a very minimal memory footprint. Memory usage has been a major issue of Gaussian Process - i.e. the numerical algorithm used for the interpolation - and using PyKeops has allowed us to use GemPy with much larger number of parameters.


##### Stateless, Procedural design

The original GemPy code base was designed in a stateful, object-oriented way. This made it difficult to use in a production environment, where it is often necessary to run the same code multiple times with different parameters. The new GemPy code base is designed in a stateless, procedural way. This make the functional graph much mode clear and easier to extend. Altouhg we are in the early stages  of this new architecture, we have already been able to implement a number of minor but nevertheless useful new features, including:

- **External Implicit functions** to interpolate some of the structural group. Previously the interpolation of all the structues in a given geological model had to be done with the Gaussian Process Regression (a.k.a Krigin) even when the specific geometry was not very compatible with the type of geometries produced by that interpolator. With the new architecture now we can inject any implicit function to do the core interpolation while still using everything else in gempy. This is particularly useful for dykes, fault planes, blobs etc.
 
- **Fault zone**: Previously faults were always modeled as a 2D monifold in 3D space. In the new version of GemPy we can give them certain thickenss to define an extra domain mapped to a given fault. This is specially useful when the use of the geological model is flow dynamics or any other simulation sensitive to fault zones.

- TODO: Add more?


##### Nugget effect optimization

Nugget effect is a parameter that defines how much variance have a given parameter on an interpolation. In practice this value controls how "exact" (this is obviously this is an oximoron, in not mathematical terms would be better expressed as how close) the final interpolation has to honor each parameter. These parameters can be defined independently (TODO: Cite Jan's paper) but always the question has been how tune the nugget effects automatically. Here, levearing PyTorch framewor for optimization, ADAM (TODO: Add reference) we have optimized the nugget effects with respect the conditional number of the covariance matrix of the Gaussian Process. This process have been used in some of the examples yielding good results. In practice, areas of the model where the data was specially uncompatible with the rest of parameters were satisfactrily identified and the nugget effect was increased in those areas. This is a very promising approach that we are still exploring. (TODO: Link to the example).

This procedure is essential for probabilistic inversion since we need the model to behave as close to linear as possible. This is specially important for the inversion of geophysical data since the forward modeling is linear and hence the inversion is much more stable if the model is also linear. By minimizing the condition number tof the covariance matrix we are able to increase the local linearity of the model.

##### Optimized dependency management

As a complex libreary with a large range of uses, GemPy has always been tapping on many of the libraries on the open-source geoscientific packages and data science. As powerful the ecosystem is, depending on a large area of the ecosystem makes the package very fragile to any change in any of the multiple libraries used. To fight this, and to make gempy more robust and future proof, we have split the main library into multiple packages that require a subset of the functionality. Therefore, the gempy project entails:

  - `gempy` [here](): The main package. It is meant to be used as a wrapper of the other packages and to provide a clean and easy to use API. It also includes the documentation and the tutorials.
   
  - `gempy_engine` [here](https://github.com/gempy-project/gempy_engine): The core of the library. Includes the interpolation, boolean operations to handle geological discontinuities and the geophysical forward modeling.
   
  - `gempy_probability` [here](https://github.com/gempy-project/gempy_probability): A set of tools to do probabilistic modeling with gempy. For now only using Pyro. This includes the inversion of geophysical data and the generation of stochastic geological models.
   
  - `gempy_viewer` [here](https://github.com/gempy-project/gempy_viewer): A set of tools to visualize the geological models and the results of the forward modeling.

  - `gempy_plugins`[here](https://github.com/gempy-project/gempy_plugins): A set of plugins to extend the functionality of gempy. Each of this plugin is meant to be independent of the others and can be used in any combination.

### GemPy Probability

As mentioned above, GemPy Probability is a package meant to work in tandem with the rest of the gempy project packages to do probabilistic modeling (TODO: Add somewhere link). The core of the package is wrapping gempy functionality with Pyro, a probabilistic programming language based on PyTorch. This allows us to do probabilistic modeling with a very easy to use API and with very powerful tools for optimization and inference.

Once we are in a probabilistic programming framework, most of the parameters involved in a gempy model can be priors, i.e. random variables from where we want to learn the posterior distribution given a set of training data or observation.

This powerful framework enables to use any type of geophysical data we can simulate as observation data to learn the posterior distribution of the selected random variables. Naturally, choosing which parameters we want to invert and how is a quite involved step by itself and requires quite a lot of domain and Baysian knowledge. Here, we present a few examples of different probabilistic networks but in the end each model will have to be tailored for a specific scientific question and area.


### Subsurface

## Outlook

## Conclusion

## References