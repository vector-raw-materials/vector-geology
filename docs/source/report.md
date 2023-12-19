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

([TODO: Tomorrow])

## Work Description

### Vector Geology
- [ ] Use of simpeg

[TODO: Thursday]

Vector Geology is a collaborative platform integrating a suite of geophysical and remote sensing data processing tools developed by the `Vector consortium <https://vectorproject.eu//>`_ . This repository serves as the central hub for accessing, understanding, and utilizing a range of software packages designed for geological exploration and research. Its key features include:

* **Integrated Toolsets**: Collection of diverse code packages developed and maintained by Vector consortium members, offering a wide range of functionalities for geophysical data processing.

* **Comprehensive Tutorials**: Step-by-step guides demonstrating typical workflows, from data input to advanced inversion techniques.

* **Extensible Framework**: Designed to accommodate a variety of datasets and scientific inquiries, with the flexibility to adapt and expand to meet specific project needs.

* **Collaborative Development**: Opportunities for users to contribute, enhancing the repository with their expertise and feedback.

In the context of geophysical inversion, here we are including several workflows defined by different partners of the Vector consortium. These workflows are meant to be used as a starting point for the inversion of geophysical data. They are not meant to be used as a black box but rather as a starting point for the inversion of geophysical data. To date this workflows include examples done with `simpeg` and `gempy`. 

`simpeg` is a mature library for geolphysical inversion (TODO: cite) developed primarily by the `UBC geophysical inversion group < and hence here we show the integration of this library with the rest of the Vector tools and the use of this library for the inversion of geophysical data.

In the case of `gempy`since it is primarily developed by parters in the consoritum, we have included more details about the intrinsics of the library and how it can be used for the inversion of geophysical data. Furthermore, the integration with other vector libraries is a bit more advanced.

#### Documentation

For the documentation of the Vector Geology repository we have used `sphinx` and `sphinx-gallery`. Sphinx is a documentation generator that allows to write documentation in reStructuredText (reST) and generate HTML, PDF, EPUB, etc. files. Sphinx-gallery is a Sphinx extension that allows to automatically generate an example gallery from a project's documentation. The examples are written in Python and can be run directly from the documentation. The documentation is hosted on `readthedocs <https://vector-geology.readthedocs.io/en/latest/>`_.

By generating the documentation from the code, we ensure that the documentation is up to date and make it easier to update the examples and tutorials as the partners generate content. 

Furthermore, we can link external sphinx gallery examples in order to integrate the documentation of the different libraries. For example, we have linked the `gempy_probability` examples [TODO: Add link]. This flexibility is important in this type of projects since we cannot centralize all the developments of so many partners in a single repository.


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

-------------------------------------------------------------
##### Nugget Effect Optimization

The nugget effect is a critical parameter in geostatistics, influencing the variance of a given parameter during interpolation. In simpler terms, it determines how closely the interpolation honors each data point. This section explores the optimization of the nugget effect, leveraging advanced computational techniques.

Nugget effect parameters can be set independently ([TODO: Cite Jan's paper]), but the challenge lies in their automatic adjustment. Utilizing the PyTorch framework and ADAM optimization ([TODO: Add reference]), we have developed a method to fine-tune these parameters dynamically. This approach is based on optimizing the conditional number of the covariance matrix in a Gaussian Process.

Our experiments have demonstrated promising results. Specifically, the optimized nugget effect has effectively identified and adjusted for areas in the model where data were notably inconsistent with other parameters. This was particularly evident in regions with unusually high noisy values, where the nugget effect was increased to accommodate the discrepancies.

The application of this optimized nugget effect has been integral in probabilistic inversions where the linearity of the problem is crucial. For instance, in [Project Name or Example], we observed [specific improvements or results]. 

In summary, the optimization of the nugget effect represents a substantial advancement in geostatistical analysis. By harnessing cutting-edge computational techniques, we are paving the way for more accurate and efficient data interpretation in various geophysical contexts.


##### Optimized Dependency Management

GemPy, as a comprehensive and versatile library, has long relied on a wide array of open-source geoscientific and data science libraries. While this ecosystem is robust, dependency on numerous libraries can render GemPy vulnerable to changes in any of these dependencies. To address this, we have adopted a strategic approach to make GemPy more resilient and future-proof.

The key to our strategy is the modularization of GemPy. We have divided the main library into distinct packages, each catering to specific functionalities. This not only simplifies dependency management but also enhances the overall stability of the system. The modular structure includes:

- `gempy`: The primary package [here](). It functions as a wrapper for other packages, offering a user-friendly API, comprehensive documentation, and tutorials.

- `gempy_engine`: The core library [here](https://github.com/gempy-project/gempy_engine). It encompasses critical components like interpolation, handling geological discontinuities through boolean operations, and geophysical forward modeling.
 
- `gempy_probability` [here](https://github.com/gempy-project/gempy_probability): A set of tools to do probabilistic modeling with gempy. For now only using Pyro. This includes the inversion of geophysical data and the generation of stochastic geological models.

- `gempy_viewer` [here](https://github.com/gempy-project/gempy_viewer): A set of tools to visualize the geological models and the results of the forward modeling.

- `gempy_plugins`[here](https://github.com/gempy-project/gempy_plugins): A set of plugins to extend the functionality of gempy. Each of this plugin is meant to be independent of the others and can be used in any combination.


This modular approach offers several advantages:

1. **Stability and Reliability**: By reducing the dependency on external libraries, we enhance the stability and predictability of GemPy.

2. **Scalability and Flexibility**: Users can choose specific modules based on their needs, making the system more scalable and adaptable to various applications.

3. **Easier Maintenance and Updates**: Managing smaller, focused packages is more efficient, facilitating quicker updates and maintenance.

In conclusion, the optimized dependency management in GemPy represents a significant step towards a more robust, adaptable, and user-friendly geoscientific toolkit. These enhancements align with our commitment to providing cutting-edge tools to the scientific community.


### GemPy Probability

GemPy Probability is an integral part of the GemPy project, designed to facilitate probabilistic modeling in geosciences ([TODO: Add link]). At its core, this package integrates GemPy's functionalities with Pyro, a probabilistic programming language built on PyTorch. This combination offers a user-friendly API and robust tools for optimization and inference, making probabilistic modeling more accessible and powerful.

By leveraging Pyro's capabilities, GemPy Probability allows for sophisticated probabilistic modeling. This framework transforms most parameters within a GemPy model into priors – essentially, random variables. Our goal is to learn the posterior distribution of these variables based on given training data or observations.

This probabilistic approach is particularly beneficial in geophysical data analysis. It enables the use of various types of geophysical data as observational inputs to learn about the posterior distribution of selected random variables. This method is invaluable in understanding and predicting geological phenomena with greater accuracy.

Implementing this framework requires substantial knowledge in both the geosciences and Bayesian statistics. Selecting which parameters to invert and determining the inversion process are complex tasks that demand expertise in these domains. To aid in this process, we provide several examples of different probabilistic networks.

Each model within GemPy Probability is tailored to specific scientific inquiries and geographical areas. These models are not one-size-fits-all solutions but are instead customized to address unique challenges and questions in the field of geoscience.


### Subsurface

[TODO: Thursday]

## Outlook
[TODO: Friday]

## Conclusion
[TODO: Friday]

## References