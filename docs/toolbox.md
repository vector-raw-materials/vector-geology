2. Overview of the Toolbox
## Repository Goals
   The primary goal of the Vector Geology repository is to provide a comprehensive platform that integrates various
   software tools and data processing workflows developed during the duration of the Horizon Europe project Vector. 
   This repository serves as a core component of WP4, focusing specifically on the integration of multisource data to facilitate 3D geological
   modeling. The repository provides a unified environment  for current and future researchers interested in the developments
    of the Vector project, ensuring that all necessary functionalities are accessible through a cohesive open-source framework.

   The toolbox offers several key functionalities, including data ingestion, geomodeling, geophysical analysis, 
   petrophysical analysis using an array of machine learning methods and visualization. 
   
**Data ingestion** tools help researchers import and preprocess geological and geophysical datasets,
   ensuring compatibility with subsequent modeling workflows. 

**Geomodeling** capabilities are provided through software
   such as GemPy, allowing users to construct 3D models of the subsurface, while 

**geophysical** analysis tools enable the
   interpretation of geophysical measurements to derive meaningful geological insights. Finally, visualization tools,
   including PyVista, are integrated to help users understand and present the results of their analyses effectively.

   The toolbox leverages both existing and newly developed software packages to create a comprehensive geoscientific
   stack. Notable components include _GemPy_ for geological modeling and probability, _subsurface_ for handling borehole data, _hylite_, _hklearn_ and _hycore_ for geophysical modeling, and PyVista for 3D visualization. By integrating these packages, the toolbox aims to provide a
   flexible, extensible platform that can adapt to the evolving needs of the geoscientific community.

  ## 2.2 Structure and Coordination
   The development of the Vector Geology repository is a collaborative effort involving multiple partners, each
   contributing their expertise to different aspects of the toolbox. The integration of software packages developed by
   various partners presents significant challenges, particularly in terms of ensuring compatibility and maintaining a
   cohesive user experience. TER has taken on the role of coordinating these efforts, streamlining the
   development and integration of different components to create a unified platform. This includes maintaining
the repository itself, setting up the documentation framework and adapting the partners' examples to the valid format

   The partners contributing to this deliverible include TER, CSIC and HZDR each focusing on specific aspects of the
   toolbox. In addition of coordinating _vector-gelogy_ itself, TER has generated a bunch of content in relation with
importing datasets, structural modeling, initial examples of using Bayesian framework for data integration and
   visualization. 
   
CSIC and HZDR have contributed by developing machine learning methods for data fusion, enabling the
   transformation of raw geological and geophysical data into actionable information. In the case of CSIC, the focus has
been on the application of machine learning of raw data to better understand the petrophysical properties at least on the cores 
analyzed in the case studies. HZDR on the other hand, have been developing novel deep learning methods for the interpretation of
hyperspectral data into petrophysics. In addition, HZDR has generated a set of examples of gravity inversions as reference for other partners.

   The collaborative nature of this project has been quite challenging since by the nature of this project, each partner
    has its own software stack and workflows. The aim of this Task was to set up a framework where each independent partner 
can decide how much of their work they want to share and how to share it. We believe that we have been able to gather enough resources to 
represent the a big part of the work developed in the WP4 of the Vector project. We are confident that over the next months,
until the conclusion of the project we will be able to add some extra examples and data sets as more results gets published

   ## 2.3 Update from Previous Deliverable (D4.2)
   Since Deliverable 4.2, significant progress has been made in the development of the Vector Geology toolbox. One of
   the key advances has been the release of GemPy and subsurface 2024.2  and the addition of the Machine Learning methods for petrophysics.
   ### GemPy 2024.2 and subsurface 2024.2
   This new version includes several improvements that make it easier for users to construct geological models.
Key new features developed during vector:
- PyTorch Backend
- LiquidEarth connection (Task -)
- TODO: Fill this table

Another major development has been the improvements made to the subsurface package, specifically in the area of well
data importing. These enhancements have made it more efficient to bring borehole data into the toolbox, allowing for
more accurate and detailed subsurface models. The ability to easily import well data and convert it in input for GemPy
ensures a reproducible workflow for constructing 3D geological models.

### Machine Learning Methods for Petrophysics
TODO: Add CSIC content here

   ## 2.4 Software Architecture
   The software architecture of the toolbox is designed to support the seamless integration of various geoscientific
   tools and data processing workflows. Since each partner has been developing its own packages, it was important that
_vector_geology_ would feel more like an integration hub that a package by itself. For this reason, specific auxiliary
functions used by each workflow were place close to the examples themselves since they were not meant to be general purpose.
In that sense a lot of the focus has been on the workflow examples and the documentation of the repository. In addition,
we leaned on the use of the typical data science stack in Python, including NumPy, SciPy, and pandas for data processing,
    GemPy for geological modeling, simpeg for geophysical inversion, and PyVista for visualization. The use of these
    open-source technologies ensures compatibility with the broader geoscientific community and allows for future
    extensions and contributions. This has special relevance for the future mainenance of the code once the project is over,
by leaning on other well established packages, we hope that the code will be maintained by the community.

   ## 2.5 Typical Data Workflow
   The workflow for reading, processing, and modeling data within the toolbox is designed to be both flexible and
   user-friendly. The data workflow begins with the ingestion of raw data, which may include geological, geophysical.
   Specialized readers and parsers are used to import these datasets into Python data structures that we can use
   that are compatible with the subsequent modeling steps.

Some of the workflows end here once we visualize the data (add links), others try to better understand better the data
(add links) while other are aim to construct 3D models either via interpolation or inversion.

   Once the data has been ingested, it is processed and prepared for modeling. This may involve tasks such as data
   cleaning, interpolation, and transformation, depending on the type of data and the specific requirements of the
   modeling workflow. The processed data is then used to construct 3D geological models, which represent the subsurface
   structures and properties of interest. These models are built using tools like GemPy, which allows users to define
   geological units, faults, and other features based on the available data.
   Geophysical forward modeling and inversion are also supported by the toolbox, enabling users to simulate geophysical
   responses based on their geological models and adjust the models to better match observed data. Visualization tools,
   such as PyVista, are used throughout the workflow to help users understand the data, evaluate the models, and
   communicate the results effectively.

   ## 2.6 Licensing and Compliance
   To ensure compliance with open-source policies and facilitate broad usage of the toolbox, all software packages
   included in the Vector Geology repository are licensed under one of three open-source licenses: EUPL, Apache License
   2.0, or MIT License. Each of these licenses has been chosen to provide an appropriate balance between openness and
   protection for both developers and users. The EUPL is particularly suitable for projects aligned with European open
   science initiatives, while the Apache and MIT licenses offer additional flexibility for reuse and contributions from
   the wider community.
   Steps have been taken to ensure that all contributions comply with the chosen licensing models, including regular
   reviews of the codebase and collaboration with legal experts to address any potential issues. By adhering to these
   open-source licenses, the Vector project ensures that the toolbox remains accessible, reusable, and adaptable for
   future research efforts.
