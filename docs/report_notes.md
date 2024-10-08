Description of work
-------------------

Task 4.1: Common Earth Models and machine learning data fusion (TER, CSIC, HZDR, GFZ)
Task 4.1 will provide an integrative platform to store, and further interpret the geological, geochemical, and
geophysical data produced in WP2 (TER). Ultimately, the goal is to create fully reproducible scripts to generate
one or several 3D “common earth models” - i.e., quantitative models of the subsurface consistent with as much
data as possible - for each case study of WP2 by the end of the project, including the 3D geometry of the most
relevant geological formations as well as geophysical, petrophysical and petrochemical distributions within each
unit.
- Integrate existing and new compatible software packages to the current open-source geoscientific stack
  packages (e.g. GemPy, subsurface, pyGIMLi, PyVista) (TER, HZDR).35
- Develop machine learning methods to process the raw data into information useful for the construction of the
  common earth models (CSIC, HZDR, GFZ).
- Apply soft labeling and adversarial learning to geophysical data to inform and validate the quality of
  petrophysical distributions over space (HZDR, CISC, GFZ). The joint data will be compared with
  petrophysical data to better understand possible relations between chemistry/rock, composition/mineralogy,
  and petrophysical properties. Resulting relationships will then be utilized to derive vectors toward
  mineralized zones as well as defining the uncertainties associated with these vectors
- Combine the information produced from the different measurements and computations in a Bayesian
  framework. The types of information comprise raw geophysical (Task 2.2), mineralogical (Task 2.3), and
  geochemical (Task 2.4) data, surrogate information obtained from supervised and unsupervised machine
  learning processes, expert geological interpretations (D2.3), and evolutionary simulations (Task 4.1).

Deliverable description
-----------------------
D4.3 Open-source toolbox for 3D modelling of multisource data (M28);

Vector Geology Repository
-------------------------

Overview
Welcome to the Vector Geology repository, a collaborative platform integrating a suite of geophysical and remote sensing data processing tools developed by the Vector consortium . This repository serves as the central hub for accessing, understanding, and utilizing a range of software packages designed for geological exploration and research.

Key Features
Integrated Toolsets: Collection of diverse code packages developed and maintained by Vector consortium members, offering a wide range of functionalities for geophysical data processing.

Comprehensive Tutorials: Step-by-step guides demonstrating typical workflows, from data input to advanced inversion techniques.

Extensible Framework: Designed to accommodate a variety of datasets and scientific inquiries, with the flexibility to adapt and expand to meet specific project needs.

Collaborative Development: Opportunities for users to contribute, enhancing the repository with their expertise and feedback.

Getting Started
Prerequisites
Basic understanding of geophysical exploration techniques and remote sensing data.

Familiarity with programming languages and tools used in the repository (e.g., Python).

Note!
Note that vector-geology is still in early days; do expect things to change.

Explore Our Guides
Galleries

Vector Geology Examples
Readers and Parsers
Structural Modeling
GeoPhysics and Inversion
Probabilistic Modeling
Bayesian Inference Theory
orphan:
Vector Geology Examples
Welcome to the Vector Geology Examples section. This space is dedicated to showcasing the diverse capabilities of the Vector platform. Here, we illustrate not only what you can achieve with Vector but also guide you on how to implement these techniques effectively. Our examples are categorized into distinct areas:

Readers and Parsers: Understand how to interpret and utilize various data formats.

Structural Modeling: Explore methodologies for constructing geological models.

Geophysical Forward Engines: Dive into the simulation of geophysical processes.

Probabilistic Modeling: Grasp the nuances of probabilistic approaches in geology.

Each tutorial is designed to provide a practical understanding of Vector’s functionalities. While our goal is not to cover every feature exhaustively, we aim to offer a comprehensive overview, demonstrating the potential applications and methodologies. As each geological setting may require unique approaches, we encourage you to delve into the repository’s documentation and codebase to uncover the most suitable tools and techniques for your specific needs.

🌱 A Living Gallery: We actively encourage contributions and suggestions. If you have an application, a novel use case, or improvements to existing examples, we welcome your input. This collaborative approach ensures that the gallery not only stays current with the latest developments but also resonates with the needs and discoveries of our users.

So, whether you’re here to learn, explore, or contribute, the Vector Geology Examples gallery is your starting point for a journey into the depths of geophysical data processing and analysis.

Readers and Parsers
Effective data analysis in geophysics begins with efficient data handling. This section is dedicated to demonstrating the use of specialized libraries to read and parse various data formats commonly encountered in the Vector Geology project. Here’s what you can expect:

Format Recognition: Learn how to identify and handle different data formats used in geophysical and remote sensing research.

Library Integration: Understand how to leverage specific libraries to read these formats. We cover a range of tools, emphasizing flexibility and ease of use.

Data Conversion: Discover methods to convert raw data into xarray structures, a powerful tool for handling multidimensional datasets. This step is crucial for streamlining post-processing activities.


Reading OMF Project Into Python

Reading Model 1 OMF Project into Subsurface

Reading OMF project and exporting it to Subsurface: Example 1

Gravity Data Visualization
Structural Modeling
Delve into the world of 3D subsurface modeling with our Structural Modeling tutorials. In this section, we focus on harnessing the capabilities of gempy, a powerful tool developed by a partner within the Vector consortium. Here’s what these tutorials offer:

3D Subsurface Modeling: Learn the intricacies of constructing detailed 3D geological models. We guide you through the process of transforming raw geophysical data into a comprehensive three-dimensional representation of the subsurface.

Using gempy: Discover how to effectively use gempy for your modeling needs. Whether you are new to this tool or looking to refine your skills, our tutorials are designed to cater to a range of expertise levels.

From Data to Insights: Our goal is to empower you with the knowledge to not just create models, but also to extract meaningful insights from them. These tutorials bridge the gap between raw data and the decision-making processes in geology.

Prepare to enhance your understanding and skills in geological modeling with a focus on practical, real-world application. By the end of this section, you will be equipped to construct and interpret complex structural models, a crucial skill in the field of geophysical research.


Constructing Structural Geological Model

Construct Model 1 with Helper Functions

Construct Spremberg: Importing borehole data

Construct Spremberg: Building initial model
GeoPhysics and Inversion
Our tutorials are crafted to guide you through the process of applying geophysical forward modeling and inversion techniques, using advanced tools like gempy and simpeg. Here’s what you can expect:

Geophysical Forward Modeling: Learn how to transform structural models into geophysical models. These tutorials will show you how to use gempy to simulate geophysical responses based on your structural models. Whether you are a beginner or looking to refine your modeling skills, these guides offer comprehensive insights.

Classical Inversion Techniques: Dive into the world of geophysical inversion using the powerful simpeg package. We’ll demonstrate how to apply classical inversion methods to geophysical data, turning observations into meaningful geological insights.

Current Implementations: As of now, we have explored two areas:

Bouguer Gravity: Understand the principles and application of Bouguer gravity in geophysical modeling.

Gravity Gradiometry: Explore the use of gravity gradiometry in deriving detailed information about subsurface structures.


Inversion of Full Tensor Gravity Gradiometry Data

Model 1 Forward Gravity

Predicting P-Wave Velocity from Hyperspectral Data
Probabilistic Modeling
In the dynamic and often uncertain realm of subsurface exploration, probabilistic modeling emerges as a key tool. This section delves into the application of Bayesian Statistics and probabilistic approaches to address the inherent uncertainties in subsurface data. Here’s what you’ll learn:

Embracing Uncertainty: Gain an understanding of how to quantify and incorporate uncertainty in geological models. We focus on aleatoric uncertainty, which represents the inherent variability in subsurface data.

Bayesian Methods with Pyro: Discover how to apply Bayesian statistical methods using Pyro, a powerful probabilistic programming framework. These tutorials will guide you through the process of encoding uncertainty in your models, offering a robust way to handle complex geological data.

Integrating Structural Uncertainty: Learn to integrate structural uncertainty into your models using GemPy. This approach enhances the realism and reliability of your geological interpretations.

Probabilistic Inversions: Explore how geophysical data can be used as observational constraints in probabilistic inversions. This technique allows for a more comprehensive understanding of subsurface structures by incorporating both geophysical measurements and geological knowledge.


