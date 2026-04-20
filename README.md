# CMILK_GA
Genetic Algorithm-Enhanced CMILK for Robust Thermal Facial Data Imputation


# Dataset Access
Use this link to download datasets used in this project:
https://compvis.site.hw.ac.uk/dataset/cmilk-thermal-facial-temperature-dataset/

# Notes
The code was created in Matlab R2023b, any older versions were not verified.

# Required Packages:
1) Global Optimization Toolbox
2) Parallel Computing Toolbox
3) Statistics and Machine Learning Toolbox

# How to re-run experiments
1) Download dataset above and make sure it is in the same directory as cmilk_ga.m
2) Run experiments.m in Matlab.
3) Results will be displayed in command window.

# How to use VAE/GAIN/MiceForest imputation
1) Download dataset above and make sure it is in the same directory as imputation.py.
2) Add GAIN-master to path in imputation.py.
3) Run code below for imputation.
   ```python
   pyhton imputation.py
   ```
4) Results will be saved in the current directory.

# Referencing
If you make use of this code or CMILK in your work. Please cite this work as:

TFD68:

Yean Chun Ng, Alexander G. Belyaev, F. C. M. Choong, Shahrel Azmin Suandi, Joon Huang Chuah, and Bhuvendhraa Rudrusamy. 2025. TFD68: A Fully Annotated Thermal Facial Dataset with 68 Landmarks, Pose Variations, Per-Pixel Thermal Maps, Visual Pairs, Occlusions, and Facial Expressions. In SIGGRAPH Asia 2025 Technical Communications (SA Technical Communications ’25), December 15–18, 2025, Hong Kong, China. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3757376.3771410

CMILK:

Ng Yean Chun, Alexander G. Belyaev, F. C. M. Choong, Joon Huang Chuah, Shahrel Azmin Suandi, Bhuvendhraa Rudrusamy, "CMILK: correlation-based multiple imputation with local k-neighbour matching for missing thermal facial data ," Proc. SPIE 13993, Eighth International Conference on Artificial Intelligence and Pattern Recognition (AIPR 2025) , 1399334 (18 December 2025); https://doi.org/10.1117/12.3093944

CMILK_GA:

Y. C. Ng, A. G. Belyaev, F. C. M. Choong, S. A. Suandi, J. H. Chuah and B. Rudrusamy, "Genetic Algorithm-Enhanced CMILK for Robust Thermal Facial Data Imputation," in IEEE Access, vol. 14, pp. 49192-49206, 2026, doi: https://10.1109/ACCESS.2026.3678483
