# 📘 RadioTherapyProject — User Guide


## How to clone the repo 

---

*git clone https://github.com/<USER_OR_ORG>/RadioTherapyProject.git*

---

## How to activate the env 
---
*Option 1 : through the yml file in /home/shared/env_for_RT_project*

open the terminal 

1. Navigate to the folder with the Yml file
   
	*cd /home/shared/env_for_RT_project*

	*ls*

you should see arvous.yml 

3. Create the environment
   
	*conda env create -f arvous.yml*

5. Activate the environment
   
	*conda activate arvous*

*Option 2 : through the repo *

*cd RadioTherapyProject*

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
- Choose the root dataset folder (must be organized as in the Radiation_plans_TABM folder).  
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

### 6. Visualize DVHs
- Use the **ROIs to Display** panel to select structures of interest.  
- Selected ROIs appear in the DVH plot.  
- Change axes as needed:
  - **X-axis** →  *dose* or *relative dose*.  
  - **Y-axis** →  *volume* or *relative volume*.  




<img width="1427" height="893" alt="image" src="https://github.com/user-attachments/assets/118d5846-fa71-4784-af3a-9dbdfb2e281a" />




