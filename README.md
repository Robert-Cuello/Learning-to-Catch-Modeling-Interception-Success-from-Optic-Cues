# Learning-to-Catch-Modeling-Interception-Success-from-Optic-Cues


This repository contains the full set of scripts and models developed for my Master's thesis:  
**“Learning to Catch: Modeling Interception Success from Optic Cues.”**

The project analyzes human behavior during interception of virtual balls under varying gravity conditions, using data from Aguado & López-Moliner (2025). It applies machine learning techniques to predict interception success based on early optic variables such as pitch (vertical elevation), and θ (retinal size).

---

## Project Objectives

- **Train a classification model** (logistic regression, neural network) to predict whether a ball will be intercepted (`catch` vs. `missh`) based on early optic input.
- **Reduce temporal input** to discover the minimal information required for accurate prediction.
- **Analyze the role of gravity** as a modulator of interception behavior.

---

## Repository Structure

├── Data Extraction.Rmd # R script to generate CML_data.csv from RData
├── interception_utils.py # Python utility functions
├── Modularized ML classification model.ipynb # Main Jupyter notebook (ML pipeline)
├── CML_data.csv # Generated dataset for ML (not included in repo)
└── README.md # This document

---

## Data

The data originates from the VR interception experiment published in:

> Aguado, B., & López-Moliner, J. (2025). *The predictive outfielder: a critical test across gravities*. Royal Society Open Science, 12: 241291.  
> [Link to article](https://doi.org/10.1098/rsos.241291)

The raw and processed datasets are hosted on OSF:

**Download `final_db.RData` here:**  
(https://osf.io/bcp28/files/osfstorage)
> This GitHub repo only includes code and no raw participant data. All analysis uses anonymized, preprocessed input.

---

## How to Run

1. **Download the required scripts:**
   - `Data Extraction.Rmd`
   - `Modularized ML classification model.ipynb`
   - `interception_utils.py`

2. **Run the preprocessing (R):**
   - Open `Data Extraction.Rmd` in RStudio.
   - Make sure `final_db.RData` is downloaded and in the working directory.
   - Run the script to generate `CML_data.csv`.

3. **Run the ML model (Python):**
   - Open `Modularized ML classification model.ipynb` in Jupyter Notebook.
   - Make sure `CML_data.csv` is in the same directory.
   - Run all cells to reproduce the model training, evaluation, and plots.

---

## Acknowledgements

- Based on experimental data by **Borja Aguado & Joan López-Moliner**
- Supervised by **Dr. Joan López-Moliner**
- Machine learning model and analysis designed and implemented by the author for a Master's thesis in Cognitive Science

---

## License

This repository is shared under the [MIT License](LICENSE).  
All code is free to use and adapt with attribution.  
Participant data is hosted externally on OSF and reused under ethical guidelines.
