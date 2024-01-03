# D4.2: Toolbox for joint processing of geophysical and remote sensing data

### Summary


> Development of a toolbox for the joint processing of geophysical and remote sensing data

## Introduction

This deliverable focuses on establishing a shared repository, equipped with tools and examples for reading, manipulating, and inverting various types of data commonly used during exploration phases. Emphasis has been placed on integrating a selection of code packages, primarily contributed by members of the Vector consortium, along with a new repository developed specifically for the Vector project.

In addition to the code essential for performing data inversion, we have included a comprehensive set of tutorials. These guides lead users through typical workflows, from data reading to final optimization processes. Due to intellectual property constraints, all tutorial examples are anonymized, and the datasets required to replicate these examples are exclusive to the Vector consortium. Efforts are underway to address these limitations, aiming to make all materials described in this deliverable freely accessible in the future.

The design of this toolbox emphasizes flexibility and extensibility, catering to the diverse datasets and scientific inquiries encountered during exploration. Recognizing the impracticality of creating a universally exhaustive toolbox, our approach has been to establish a robust framework. This framework can be expanded upon to accommodate specific use cases as they arise.


## Toolbox Overview

The toolbox is a comprehensive suite designed for the joint processing of geophysical and remote sensing data. Its structure is categorized into four distinct sections, each targeting a specific aspect of data processing and analysis. These include readers and parsers, structural modeling, forward geophysics and inversion, and probabilistic modeling.

### Readers and Parsers

In the intricate field of geophysics and remote sensing, the journey of effective data analysis begins with efficient data handling. This section is crafted to provide a comprehensive understanding of the diverse data formats encountered in this field, guiding users through the nuances of identifying, reading, and transforming these formats into analyzable structures, particularly within the context of the Vector Geology project. We introduce users to a selection of specialized libraries, each serving as a key to unlocking different data formats, allowing users to seamlessly transition from one data format to another. users will also learn to transform raw data into xarray structures, a powerful format for handling multidimensional datasets. This is specially useful to store dataset for later use and crucial link with the app LiquidEarth [TODO: Add mention to task and or deliverable].

Accompanying these technical aspects are interactive tutorials, which provide a hands-on experience through real-world scenarios. While our commitment to intellectual property rights necessitates the use of anonymized datasets in these tutorials, this does not diminish their educational value. As of now, the materials and datasets are exclusively available to the Vector consortium, but there is a vision for the future where these resources become widely accessible. 

### Structural Modeling

Structural modeling is an important aspect for many of the methods used for geophysical and other remote sensing processing. The Structural Modeling section of our toolbox is designed for professionals engaged in the intricate task of 3D subsurface modeling, offering a comprehensive suite of tutorials and tools focused on practical applications and in-depth understanding.

Structural modeling serves as a bridge between raw geophysical data and their transformation into detailed subsurface models. This segment is particularly valuable for those seeking to visualize and comprehend geological formations beneath the Earth's surface. Central to this section is `gempy`, a tool developed by our partners within the Vector consortium. The tutorials are crafted to accommodate varying levels of familiarity with gempy, from beginners to advanced users. By exploring these resources, users can master the use of gempy for creating intricate and accurate geological models. The tutorials are not just about tool usage but also provide some  insights and techniques to enhance modeling proficiency.

As part of our commitment to practical, real-world applications, this section of the toolbox is crafted to empower users with essential skills in geological modeling, crucial for advanced geophysical research. We understand the dynamic nature of geoscience and are dedicated to continually evolving and expanding this toolbox to address emerging needs and discoveries in the field.


### Forward Geophysics and Inversion

The "Forward Geophysics and Inversion" section of the D4.2 toolbox is dedicated to guiding users through the methodologies of geophysical forward modeling and inversion techniques. Our commitment to providing comprehensive learning resources is embodied in a series of detailed tutorials, utilizing state-of-the-art tools such as `gempy` and `simpeg`.

The first facet of this section delves into geophysical forward modeling. Users are introduced to the process of converting structural models into geophysical models, an essential skill in exploration geophysics.  Using `gempy`, we offer step-by-step guidance on simulating geophysical responses, ensuring users gain a thorough understanding of the underlying principles and practical applications.

We then transition to classical inversion techniques, employing the robust capabilities of the `simpeg` package. This segment is designed to equip users with the skills to apply inversion methods to geophysical data, thereby extracting valuable geological insights from observational data. Our tutorials are structured to provide both theoretical knowledge and practical skills, allowing users to confidently navigate the complexities of geophysical inversion.

As part of our ongoing commitment to practical and applied learning, we have currently implemented tutorials focusing on two key areas:

1. **Bouguer Gravity**: This section offers an in-depth exploration of Bouguer gravity. Users are taught the principles of this technique and how it's applied in geophysical modeling, providing a solid foundation in one of the most fundamental aspects of geophysical analysis.

2. **Gravity Gradiometry**: Here, we explore the application of gravity gradiometry. This advanced technique is crucial for obtaining detailed information about subsurface structures. Our tutorials guide users through the intricacies of gravity gradiometry, enhancing their ability to discern and interpret complex geological features.

It's designed not only as a learning platform but also as a stepping stone for future innovations in the field of exploration geophysics. As the project progresses, we aim to expand these resources, incorporating more techniques and examples to reflect the evolving landscape of geophysical exploration.


### Probabilistic Modeling

In subsurface exploration, uncertainty is a constant. The Probabilistic Modeling section of our toolbox confronts this challenge head-on, employing advanced Bayesian Statistics and probabilistic methodologies. This segment of the toolbox is dedicated to quantifying and integrating the inherent variability, or aleatoric uncertainty, present in subsurface data.

Central to this section is the application of Bayesian statistical methods, facilitated by Pyro, a cutting-edge probabilistic programming framework. This powerful tool enables users to encode and manage uncertainty within their geological models effectively. Our tutorials guide users through this process, ensuring a thorough understanding of handling complex geological data within a probabilistic framework. Further enhancing the toolbox's capabilities, we introduce the integration of structural uncertainty into modeling processes via GemPy. This strategy not only elevates the realism of geological interpretations but also bolsters their reliability. By weaving structural uncertainty into the fabric of our models, we open new avenues for understanding and interpreting subsurface structures.

Another pivotal feature of this section is the utilization of geophysical data in probabilistic inversions. This technique leverages geophysical measurements as observational constraints, allowing for a more nuanced and comprehensive understanding of subsurface formations. By combining geological knowledge with empirical data, our toolbox facilitates a deeper exploration of subsurface structures, paving the way for more informed decision-making in the field of geoscience.

## Work Description

The work done for this deliverable is framed within Task 4.1: [TODO: Add exact name]. The bulk of work for ths task is being divided into 
multiple code bases and repositories. The main repositories are:

### Vector Geology [New Package] (Deliverable [TODO: Add deliverable number])

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


### GemPy [Development]

  GemPy is a Python-based open-source library for 3D geological modeling. It is designed to facilitate the construction of complex geological models, including the generation of 3D grids, the interpolation of geological contacts and faults, and the simulation of geological processes. GemPy was developed mostly as a PhD student project and was in need of a quite major overhaul to make it robust and reliable in production environments. The Vector project provided the opportunity to do this and hence a lot of the work necessary to produce this deliverable has been done by refactoring and updating the GemPy code base. The main changes are:

- [ ] [TODO]: **What is gempy exactly** Add small comment about gempy interpolation Gaussian Process and Kriging differences. Add concept of surface points and orientations


Now we will go through the main changes in the GemPy code base:

#### Tensor Backends

Something that happened over the life of the GemPy project has been the rise of deep learning and the development of a number of tensor backends for Python. These backends provide a way to do automatic differentiation and hence to do gradient-based optimization. Amidst this, the original Tensor library used by GemPy got deprecated. In order to combat this rapidly changing enviroment, we rearchitect GemPy to relatively easily being able to choose between different tensor backends. Currently, we support numpy, PyTorch and Pykeops. 

The use of PyTorch makes much easier to design complex probabilistic models - including probabilistic inversions - with an easy to use API and with very remarkable error messages while being able to get the gradients of the operations and even run the code on GPUS. Pykeops, on the other hand, is capable to do reductions operations with a very minimal memory footprint. Memory usage has been a major issue of Gaussian Process - i.e. the numerical algorithm used for the interpolation - and using PyKeops has allowed us to use GemPy with much larger number of parameters.


#### Stateless, Procedural design

The original GemPy code base was designed in a stateful, object-oriented way. This made it difficult to use in a production environment, where it is often necessary to run the same code multiple times with different parameters. The new GemPy code base is designed in a stateless, procedural way. This make the functional graph much mode clear and easier to extend. Altouhg we are in the early stages  of this new architecture, we have already been able to implement a number of minor but nevertheless useful new features, including:

- **External Implicit functions** to interpolate some of the structural group. Previously the interpolation of all the structues in a given geological model had to be done with the Gaussian Process Regression (a.k.a Krigin) even when the specific geometry was not very compatible with the type of geometries produced by that interpolator. With the new architecture now we can inject any implicit function to do the core interpolation while still using everything else in gempy. This is particularly useful for dykes, fault planes, blobs etc.
 
- **Fault zone**: Previously faults were always modeled as a 2D monifold in 3D space. In the new version of GemPy we can give them certain thickenss to define an extra domain mapped to a given fault. This is specially useful when the use of the geological model is flow dynamics or any other simulation sensitive to fault zones.

- TODO: Add more?


#### Activation function

[TODO:]

#### Nugget Effect Optimization

The nugget effect is a critical parameter in geostatistics, influencing the variance of a given parameter during interpolation. In simpler terms, it determines how closely the interpolation honors each data point. This section explores the optimization of the nugget effect, leveraging advanced computational techniques.

Nugget effect parameters can be set independently ([TODO: Cite Jan's paper]), but the challenge lies in their automatic adjustment. Utilizing the PyTorch framework and ADAM optimization ([TODO: Add reference]), we have developed a method to fine-tune these parameters dynamically. This approach is based on optimizing the conditional number of the covariance matrix in a Gaussian Process.

Our experiments have demonstrated promising results. Specifically, the optimized nugget effect has effectively identified and adjusted for areas in the model where data were notably inconsistent with other parameters. This was particularly evident in regions with unusually high noisy values, where the nugget effect was increased to accommodate the discrepancies.

The application of this optimized nugget effect has been integral in probabilistic inversions where the linearity of the problem is crucial. For instance, in [Project Name or Example], we observed [specific improvements or results]. 

In summary, the optimization of the nugget effect represents a substantial advancement in geostatistical analysis. By harnessing cutting-edge computational techniques, we are paving the way for more accurate and efficient data interpretation in various geophysical contexts.


#### Optimized Dependency Management

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


### GemPy Probability [New Package]

GemPy Probability is an integral part of the GemPy project, designed to facilitate probabilistic modeling in geosciences ([TODO: Add link]). At its core, this package integrates GemPy's functionalities with Pyro, a probabilistic programming language built on PyTorch. This combination offers a user-friendly API and robust tools for optimization and inference, making probabilistic modeling more accessible and powerful.

By leveraging Pyro's capabilities, GemPy Probability allows for sophisticated probabilistic modeling. This framework transforms most parameters within a GemPy model into priors – essentially, random variables. Our goal is to learn the posterior distribution of these variables based on given training data or observations.

This probabilistic approach is particularly beneficial in geophysical data analysis. It enables the use of various types of geophysical data as observational inputs to learn about the posterior distribution of selected random variables. This method is invaluable in understanding and predicting geological phenomena with greater accuracy.

Implementing this framework requires substantial knowledge in both the geosciences and Bayesian statistics. Selecting which parameters to invert and determining the inversion process are complex tasks that demand expertise in these domains. To aid in this process, we provide several examples of different probabilistic networks.

Each model within GemPy Probability is tailored to specific scientific inquiries and geographical areas. These models are not one-size-fits-all solutions but are instead customized to address unique challenges and questions in the field of geoscience.


### Subsurface [Development]

The Subsurface module serves as a DataHub for geoscientific data, with two primary objectives:

1. **Data Unification**: Our aim is to unify geometric data into standardized data objects. These objects utilize numpy arrays for memory representation, ensuring compatibility across various packages in our stack.

2. **Basic Data Interactions**: We provide tools for essential interactions with these data objects, including:
- **Read/Write Operations**: Efficient and reliable data manipulation capabilities.
- **Categorized/Metadata Handling**: Organized and accessible data management.
- **Visualization Tools**: Intuitive and powerful visualization options to better understand and interpret the data.

This module represents a key component in our endeavor to streamline the handling of geoscientific data. By offering a unified approach to data management and interaction, Subsurface significantly enhances the user experience, leading to more efficient and effective data processing and analysis.

During the project we expanded the capabilities of the Subsurface module to be able to handle the data provided by the Vector partners. 

## Outlook and Conclusion

In this deliverable we have presented the current state of the Vector Geology toolbox. As the fist stage of this toolbox, the main focus has been on geophysics. In practice, a lot of the work has been foundational for the main goal of task 4.1 which is to provide a centralize place to integrate much of the different development of Work package 4. To this end, we have been created a new open-source repository - Vector Geology - including all the logic for authomatic documentation.

This deliverable has showcased the development and current state of the Vector Geology toolbox, marking a significant milestone in our journey. The initial phase of the toolbox has been centered on geophysics, laying the groundwork for Task 4.1's primary objective: to establish a centralized integration point for the diverse developments within Work Package 4. A pivotal achievement in this phase is the creation of the new open-source repository, Vector Geology, which includes extensive logic for automatic documentation.

*Ongoing open-source development*

The groundwork for this project dates back to the beginning of the last decade, rooted in the global open-source community's ongoing efforts to innovate geoscientific tools. Many of this tools developed by european partners in similar type of projects. The ambition of this endeavor is vast, extending beyond the scope of a single three-year project. Nonetheless, initiatives like Vector are instrumental in sustaining and advancing these codebases, moving us towards a more robust and reliable open-source ecosystem for geosciences, particularly in the exploration of critical raw materials.

*Future work*

In the upcoming months, our focus will be on further developing Vector Geology and its associated libraries as new data from Work Package 2 becomes available. This progress will be detailed in the forthcoming deliverable [TODO: Add deliverable title for Open-source Toolbox]. With the final datasets taking shape, our aim is to construct extensive probabilistic networks that incorporate as much of this data as possible. A special focus will be on integrating Hyperspectral datasets, a topic, to our knowledge, yet unexplored in such networks.


*Datasets*

Midway through the project, we await additional datasets and are in the process of defining subsequent steps. The current lack of final data and inherent uncertainties in scientific projects have posed challenges in setting requirements for the toolbox. However, this scenario also presented an opportunity to strive for generality in our approach, enhancing the toolbox's applicability for the broader community and future endeavors.

*Conclusion*

Reflecting on our progress to date, the Vector Geology toolbox represents a significant step forward in geoscientific collaboration and development. This project, while primarily aimed at meeting immediate needs within the consortium, also contributes to the broader field of geoscience by establishing a versatile foundation for ongoing and future research. Looking ahead, we are optimistic about the toolbox's potential to enhance the open-source geoscientific community. Our continued efforts are focused on deepening the understanding of the subsurface, a vital aspect of geoscientific exploration and research.

## References










