# Deliverable 4.3: Open-source Toolbox for 3D Modelling of Multisource Data

## Executive Summary
A brief overview summarizing the goals, development process, and key achievements of the open-source toolbox for 3D modeling. This section should outline the main contributions of the deliverable, highlighting the collaborative efforts, software packages developed, and the impact of the toolbox on open science and the geoscience community.

## 1. Introduction
1.1 Background and Objectives
- Overview of the Vector project and the goals of Work Package 4 (WP4).
- Importance of open-source tools for geoscientific research and alignment with Horizon Europe objectives.
- Task 4.1 aims to provide an integrative platform to store and interpret geological, geochemical, and geophysical data produced in WP2, ultimately creating reproducible scripts to generate 3D "common earth models" for each case study of WP2.
- Shortcomings. Data licensing has been an issue and a limitation. Also some of the workflows in Vector have been using proprietary software and therefore could not be included in this repository.

1.2 Commitment to Open Science
- Statement on open science and its importance for transparency and accessibility in research.
- Description of the licensing structure (EUPL, Apache 2.0, MIT) and the rationale for choosing these licenses.

## 2. Overview of the Toolbox
2.1 Repository Goals
- Detailed description of the goals of the repository, specifically in the context of WP4 and geological modeling.
- Overview of the different functionalities: data ingestion, geomodeling, geophysical analysis, etc.
- The toolbox aims to integrate existing and new compatible software packages into the current open-source geoscientific stack, including tools like GemPy, subsurface, pyGIMLi, and PyVista.

2.2 Structure and Coordination
- Explanation of the contributions from multiple partners and the challenges of integrating different packages.
- Role of Terranigma in coordinating and streamlining package development and integration.
- Partners contributing to Task 4.1 include TER, CSIC, HZDR, and GFZ, focusing on various aspects such as machine learning, data fusion, and geophysical data analysis.

2.3 Update from Previous Deliverable (D4.2)
- Summary of the progress made since Deliverable 4.2, highlighting key advances in the toolbox development.
  - GemPy v3 full release
  - Subsurface improvements on well data importing
- Developments include the integration of machine learning methods to process raw data for constructing common earth models, and advancements in soft labeling and adversarial learning to validate the quality of petrophysical distributions.

2.4 Software Architecture
- Overview of the architecture of the toolbox, including the relationships between different packages.
- Explanation of the technologies used (e.g., programming languages, frameworks, libraries).
- The toolbox is designed to be extensible, allowing the integration of additional datasets and software tools to meet evolving project needs.

2.5 Data Workflow
- Description of the workflow for reading, processing, and modeling data.
- Explanation of how the toolbox supports geological modeling and geophysics tasks, including examples such as reading and parsing data formats, constructing structural models, and performing geophysical forward modeling and inversion.

2.6 Licensing and Compliance
- Details on the licensing models used for different packages.
- Steps taken to ensure compliance with open-source policies and usage rights.

## 3. Work Description
3.1 Development of Individual Sections
- Description of the individual software packages, their functionalities, and their integration into the toolbox.
  - **Readers And Parsers (TER)**: Reading dataset provided by the different partners. Including OMF, csv containing datasets etc
  - **Structural Modeling (TER)**: Being able to build models from row data or modify structural models created by other software.
  - **Geophysical Forward Modeling (TER, HZDR)**: Differnent methods for geophysical forward modeling.
  - **Probabilistic Modeling (TER)**: Novel the use of probabilistic modeling for geological modeling as integration of different data sources.
  - **Machine learning methods for geophysical data (CSIC)**: Analysis of geophysical data using machine learning methods.
- Integration of tools like GemPy for 3D subsurface modeling, simpeg for geophysical inversion, and PyVista for visualization.

3.2 Integration of Multisource Data
- Explanation of how different types of geoscientific data are integrated using the toolbox.
- Integration includes geological, geochemical, and geophysical data produced in WP2, combined in a Bayesian framework to create common earth models. The general numerical framework has been 
 developed but there are still multiple challenges regarding model complexity and computational cost necessary to be addressed before can be properly implemented in the case studies.
- Challenges and solutions related to multisource data modeling, including handling data from different measurements and computations.

3.3 Collaboration and Contributions
- Overview of contributions from each partner.
- Contributions include developing machine learning methods (CSIC, HZDR, GFZ), applying soft labeling and adversarial learning to geophysical data (HZDR, CSIC, GFZ), and coordinating software integration (TER).
- Description of the collaboration process and tools used to ensure seamless development and integration.
- The input data sets at the moment are only available after asking for them to the Vector consortium.

## 4. Outlook and Conclusion
4.1 Demonstrations and Use Cases
- Examples of use cases where the toolbox was applied to real-world geological modeling problems.
  - All the code used in vector-geology has been applied to Stonepark, Collinston, and Spremberg case studies. To avoid innecessary IP issues the data has been slightly annonimized
- Demonstrations showing the capabilities of the developed software, including applications in structural modeling, geophysical inversion, and probabilistic modeling.

4.2 Impact on the Scientific Community
- Discussion on the impact of making these tools open-source.
- Feedback from early users and partners, emphasizing the collaborative and evolving nature of the repository.

4.3 Future Work
- Planned future developments and improvements for the toolbox.
- Future work includes enhancing machine learning algorithms for better data fusion, expanding the toolbox to include more geophysical data types, and improving user guides and documentation.
- Opportunities for collaboration and community involvement, encouraging contributions to the repository.

4.4 Conclusion
- Summary of the achievements of Deliverable 4.3.
- Restatement of the importance of open-source development and the broader goals of Horizon Europe.

## 7. References
- List of references used in the report, including scientific articles, documentation, and other relevant sources.

## 8. Appendices
8.1 Technical Documentation
- Detailed documentation for each software package.
- Technical details on using tools like GemPy, simpeg, and PyVista.

8.2 Partner Contributions
- Specific contributions from each partner organization.
- Details on the roles of TER, CSIC, HZDR, and GFZ in Task 4.1.

8.3 Glossary
- Explanation of technical terms and acronyms used in the report.
