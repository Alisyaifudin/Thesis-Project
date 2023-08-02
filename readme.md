# Thesis Project
Title: MODEL DISTRIBUSI MATERI DI SEKITAR MATAHARI: STUDI PERBANDINGAN HIPOTESIS MATERI GELAP DAN MOND DENGAN MENGGUNAKAN DATA GAIA DR3
English: MODELING DARK MATTER DISTRIBUTION IN THE SOLAR NEIGHBORHOOD: A COMPARATIVE STUDY OF DARK MATTER AND MOND HYPOTHESES USING GAIA DR3 DATA
Last updated: 3 August 2023 

This is my thesis repo! Follow this repo to reproduce my thesis report.

## Project Structure
```
.
├── 1_Collecting_Data
│   ├── 1.1. Gaia DR3 and 2MASS.ipynb
│   ├── 1.2. RAVE6.ipynb
│   ├── 1.3. APOGEE (SDSS17).ipynb
│   ├── 1.4. GALAH DR3.ipynb
│   ├── 1.5. LAMOST DR7.ipynb
│   ├── gaia_script
│   │   └── ...
│   ├── images
│   │   └── ...
│   └── script
│       ├── create_procedure.py
│       ├── gaia-tmass.py
│       └── procedures
│           └── ...
├── 2_Cleaning
│   ├── 2.0. Preprocessing-gaia.ipynb
│   ├── 2.1. Combine.ipynb
│   ├── 2.2. Best Parameters.ipynb
│   ├── 2.3. Neighbour.ipynb
│   ├── 2.4. GUMS.ipynb
│   ├── 2.5. Color-Class.ipynb
│   ├── 2.6. Clusters.ipynb
│   └── script
│       ├── best-pars.py
│       └── combine.py
├── 3_Vertical_Number
│   ├── 3.1. Survey Completeness.ipynb
│   ├── 3.2. Effective Completeness.ipynb
│   ├── 3.3. Effective Volume.ipynb
│   └── img
│       └── ...
├── 4_Vertical_Velocity
│   ├── 4.1. Middleplane Vertical Velocity Distribution.ipynb
│   └── img
│       └── ...
├── 5_mcmc
│   ├── 1_mock_dm
│   │   ├── 5.1. gravity.ipynb
│   │   ├── 5.2 density.ipynb
│   │   ├── 5.3. fit.ipynb
│   │   ├── 5.4. figures.ipynb
│   │   ├── img
│   │   │   └── ...
│   │   └── script
│   │       ├── generate_mock.py
│   │       └── procedure
│   │           └── ...
│   ├── dddm
│   │   └── ...
│   ├── dm
│   │   └── ...
│   ├── mond
│   │   └── ...
│   └── no
│       └── ...
├── 6_Analisis
│   └── ...
├── LICENSE
├── note.md
├── readme.md
├── tes.ipynb
├── tree.txt
├── undefined_Gravity
│   └── gravity.md
└── utils
    ├── __init__.py
    ├── append_name.py
    ├── check_df.py
    ├── completeness.py
    ├── delete_directory.py
    ├── launch_job.py
    ├── load_spectral_types.py
    ├── runcmd.py
    ├── rust_utils.py
    ├── safe_mkdir.py
    ├── style.py
    └── timeout.py
```

## 1. Collecting Data
<details>
<summary>Click to expand!</summary>

* [1.1. Gaia DR3 and 2MASS](<1. Collecting Data/1.1. Gaia DR3 and 2MASS.ipynb>)
* [1.2. RAVE6](<1. Collecting Data/1.2. RAVE6.ipynb>)
* [1.3. APOGEE (SDSS17)](<1. Collecting Data/1.3. APOGEE (SDSS17).ipynb>)
* [1.4. GALAH DR3](<1. Collecting Data/1.4. GALAH DR3.ipynb>)
* [1.5. LAMOST DR7](<1. Collecting Data/1.5. LAMOST DR7.ipynb>)
* [1.6. simple 2MASS](<1. Collecting Data/1.6. simple 2MASS.ipynb>)
</details>

## 2. Vertical Number Density Distribution
<details>
<summary>Click to expand!</summary>

* [2.1. Spectral Class](<2. Vertical Number Density/2.1. Spectral Class.ipynb>)
* [2.2. Cutting](<2. Vertical Number Density/2.2. Cutting.ipynb>)
* [2.3. Survey Completeness](<2. Vertical Number Density/2.3. Survey Completeness.ipynb>)
* [2.4. Effective Completeness](<2. Vertical Number Density/2.4. Effective Completeness.ipynb>)
* [2.5. Effective Volume](<2. Vertical Number Density/2.5. Effective Volume.ipynb>)
* [2.6. Vertical Number Density](<2. Vertical Number Density/2.6. Vertical Number Density.ipynb>)
</details>

## 3. Vertical Velocity Distribution
<details>
<summary>Click to expand!</summary>

* [3.1. Combine](<3. Vertical Velocity Distribution/3.1. Combine.ipynb>)
* [3.2. Filter rvs](<3. Vertical Velocity Distribution/3.2. Filter rvs.ipynb>)
* [3.3. Best parameters](<3. Vertical Velocity Distribution/3.3. Best parameters.ipynb>)
* [3.4. Cutting](<3. Vertical Velocity Distribution/3.4. Cutting.ipynb>)
* [3.5. Vertical Velocity Distribution](<3. Vertical Velocity Distribution/3.5. Vertical Velocity Distribution.ipynb>)

</details>

## 4. ??

## 5. Profit