# 📘 RadioTherapyProject — User Guide


## How to clone the repo 

---
*git clone git@github.com:TalLychtenstein/RadioTherapyProject.git* 

or

*git clone https://github.com/TalLychtenstein/RadioTherapyProject.git* 

---

## How to activate the env 
---

*Option 1 : through the repo*


*cd RadioTherapyProject*

*conda env create -f arvous.yml*

*conda activate arvous*


*Option 2 : through the yml file in /home/shared/env_for_RT_project* just in case doesn't work

open the terminal 

*cd /home/shared/env_for_RT_project*

*ls*

you should see arvous.yml 
   
*conda env create -f arvous.yml*
   
*conda activate arvous*


---




## How to Use the Interface

### 1. Configure Settings
- Click **Settings**.  
- Choose:
  - **Resampling type**: `shape` or `spacing`.  
  - **Resampling value** (e.g., `512 × 512 × 512`).  
- Select the **Output folder** where all results (tables, NIfTI files, DVHs) will be saved.  


---

### 1. Batch Processing 
- Click **Batch Processing**.  
- Choose the root dataset folder (must be organized as in the full resampled folder).  
- ARVOUS will automatically process **all patient sessions** in that dataset and add them to the output folder.  

---

---

### 2. Load Data
- **Single Patient**: Click **Load Patient Data** and select one patient session folder.  

⚠️ The dataset **must keep the current folder structure** , **order of dcm files is not importent**:  

Patient/
├── CT/
├── RTDOSE/
└── RTSTRUCT/
└── RTPLAN 

---

---
### 3. Extract Metrics
- Click **Extract Metrics**.  
- This generates the **ROI Summary Table** (volumes, DVH metrics, indices).  
- The table is displayed in the interface **and** saved automatically to the output folder.  

---

### 4. Create NIfTI Volumes
- Click **Create NIfTI Volumes** to export CT, dose, and ROI masks as `.nii.gz` files.  
- These appear in the output folder and can be opened in external tools (e.g., 3D Slicer, FSL).  

---

### 5. Show Visualization
- Click **Show Visualization** to open the visualization window.  
- This allows you to:
  - View **CT slices** with dose overlays.  
  - See **ROI masks** overlaid on anatomical images.  
  - Interactively scroll through slices for spatial validation.  

--- 




<img width="1427" height="893" alt="image" src="https://github.com/user-attachments/assets/118d5846-fa71-4784-af3a-9dbdfb2e281a" />


## 🖥️ ARVOUS Interface Overview

When you run the application, the interface is organized into several key panels:

---

### 🔧 Settings Panel (top section)
- Displays the **resampling configuration** you selected (e.g., shape/spacing, voxel size).  
- Shows the chosen **output path** where results will be saved.  

---

### 🩺 Processing Patient Record
- Provides **patient/session details**, such as:
  - Patient ID  
  - Treatment type (e.g., SRS, WBRT)  
  - Targeted region (e.g., right frontal, multiple mets)  
  - Session date  
- Displays the **input path** for the dataset being processed.  

---

### 📊 Metadata Table (left side)
- Contains **general treatment metadata**:
  - Prescription dose  
  - Number of fractions  
  - Dose per fraction  

---

### 📑 ROI Metrics Table (right side)
- Lists all **Regions of Interest (ROIs)**.  
- Provides detailed metrics per ROI:
  - **Volume (cc)**  
  - **Minimum, Maximum, Mean, Median, Mode, and Standard Deviation dose (Gy)**  
  - **D2, D50, D98 values** (Gy)  
  - **Homogeneity Index (HI)** and **Conformity Index (CI)**  

---

### 📈 DVH Plots (bottom)
- Shows **Dose–Volume Histograms (DVHs)** for selected ROIs.  
- You can:
  - Select which ROIs to display from the **ROI list** on the left.  
  - Overlay multiple DVHs for comparison.  
  - Adjust axis definitions:
    - **X-axis** → *absolute dose* or *relative dose [% Rx]*  
    - **Y-axis** → *absolute volume (cc)* or *relative volume [%]*  

---

✅ In short:  
- **Top = settings & patient info**  
- **Middle-left = metadata table**  
- **Middle-right = ROI metrics**  
- **Bottom = DVH plots with ROI selection**  


---
## Steps to Run arvous.py in PyCharm 

This is in case cloning from git doesn't work

1. Open the Project in PyCharm
   
	Launch PyCharm.

	Go to File → Open.

      Browse to: /home/shared/RadioTherapyProject_tal_and_daniel
   
      Click OK
   
PyCharm will now treat this as the project folder. You should see arvous.py in the file tree.

2.	Configure the Conda Interpreter (arvous)

	In PyCharm, open:
	
	File → Settings →Python → Interpreter

	click on Add Interpreter → choose select existing for Environment ( Top)→
	
	choose conda for Type → choose arvous for Environment (bottom)→click OK

	click Apply then OK .


<img width="1176" height="749" alt="seting up interpreter" src="https://github.com/user-attachments/assets/d0790919-b64b-4753-9a2e-2d799b85ff62" />


Here is an example of how it should look , the path to conda will be different but that’s okay :)



