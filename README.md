
~~~
    ______                  _____   _  _     _____  ______       _____ 
   / \  / \           /\   |_   _| | || |   |  __ \|  ____/\    / ____|
  /   /\   \         /  \    | |   | || |_  | |__) | |__ /  \  | (___  
 /   /  \   \       / /\ \   | |   |__   _| |  ___/|  __/ /\ \  \___ \ 
 \--/----\--/      / ____ \ _| |_     | |   | |    | | / ____ \ ____) | 
  \/______\/      /_/    \_\_____|    |_|   |_|    |_|/_/    \_\_____/   
~~~
# AI for PFAS
A deep learning expedition into PFAS toxicity. The preprint can be accessed [here](https://chemrxiv.org/articles/preprint/Uncertainty-Informed_Deep_Transfer_Learning_of_PFAS_Toxicity/14397140).

##### Contents
 1. [Brief](#brief)
 1. [Context](#context)
 2. [Dataset](#dataset)
 3. [Installation](#installation)

## Brief
This repository contains code, instructions, and data used in the LDRD deep learning expedition on PFAS toxicity.

##### Repository structure
~~~
ai4pfas/
+-- src/
¦   +-- dataset.py
¦   +-- experimental_setup.py
¦   +-- helpers.py
¦   +-- models.py
¦   +-- preprocess_data.py
+-- notebooks/
¦   +-- benchmarks/
¦   ¦   +-- benchmarks.ipynb
¦   ¦   +-- tuning-dnn-mordred.ipynb
¦   ¦   +-- tuning-dnn-ecfp.ipynb
¦   ¦   +-- tuning-gcn.ipynb (todo)
¦   ¦   +-- tuning-rf-mordred.ipynb
¦   ¦   +-- tuning-nmf.ipynb (todo)
¦   ¦   +-- tuning-gp.ipynb (todo)
¦   +-- transfer-learning.ipynb
¦   +-- uncertainty.ipynb
¦   +-- selective-net.ipynb
+-- data/
¦   +-- ldtoxdb/
¦   ¦   +-- ldtoxdb.csv
¦   ¦   +-- ldtoxdb-classified.csv
¦   ¦   +-- ldtoxdb-mordred.csv
¦   ¦   +-- preprocessed/
¦   ¦   ¦   +-- random
¦   ¦   ¦   +-- stratified
¦   ¦   +-- benchmark-models/
¦   ¦   +-- deep-ensemble/
¦   ¦   +-- latent-space/
¦   ¦   +-- transfer-learning/
¦   +-- results/
¦       +-- benchmarks
+-- environment.yml
+-- README.md
~~~

## Context
 - **Polyfluoroalkyl and perfluoroaklyl** *(PFAS)* - a family of organic industrial synthetics of increasing federal concern for unknown hazards and widespread contamination of DOD & DOE sites
 - **Acute oral rat toxicity** *(LD50)* - a nonspecific quantification of toxicity which measures the minimum 24-hr drug dose to induce 50% lethality in a sample of labratory rats
 - **LDToxDB** - shorthand for the compiled LD50 dataset (described below)

## Dataset
LDToxDB was created by compiling LD50 datasets from the [EPA Toxicity Estimation Software Tool (TEST)](https://www.epa.gov/chemical-research/toxicity-estimation-software-tool-test), [NIH Collaborative Acute Toxicity Modeling Suite (CATMoS)](https://ntp.niehs.nih.gov/iccvamreport/2019/technology/comp-tools-dev/catmos/index.html), and National Toxicology Program (NTP). Values represent experimental LD50 measurements in -log(mol/kg). Chemical duplicates were identified by InChIKey and removed.

## Installation
### Requirements
The workflow requires Python 3.7 and relevant packages. Install Anaconda and run the following terminal commands to create an environment based on the repository requirements:
~~~
(base) $ conda create -n ai4pfas -f environment.yml
(base) $ conda activate ai4pfas
~~~

### Running the repo
Start a Jupyter Notebook instance from the terminal and navigate within the UI to execute `.ipynb` files.
~~~
(ai4pfas) $ jupyter notebook
~~~

Python files are directly executable from within the terminal.
~~~
(ai4pfas) $ python ./src/deep-ensemble.py
~~~

### How to cite ?
If you are using the AI4PFAS workflow  in your research paper, please cite us as
```
@article{ai4pfas,
author = "Jeremy Feinstein and ganesh sivaraman and Kurt Picel and Brian Peters and Alvaro Vazquez-Mayagoitia and Arvind Ramanathan and Margaret MacDonell and Ian Foster and Eugene Yan",
title = "{Uncertainty-Informed Deep Transfer Learning of PFAS Toxicity}",
year = "2021",
month = "4",
url = "https://chemrxiv.org/articles/preprint/Uncertainty-Informed_Deep_Transfer_Learning_of_PFAS_Toxicity/14397140",
doi = "10.26434/chemrxiv.14397140.v1"
}
```

## Acknowledgements
This  material  is  based  upon  work  supported  by  Laboratory  Directed  Research  and  Development (LDRD) funding from Argonne National Laboratory, provided by the Director, Office of Science, of the U.S. Department of Energy under Contract No. DE-AC02-06CH11357.
