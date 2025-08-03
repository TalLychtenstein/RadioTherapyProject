import sys
import os
import numpy as np
import pandas as pd
import matplotlib
# Force Qt5 backend for all plotting (avoids mixing TkAgg)
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem, QMessageBox
)
from PyQt5.QtCore import Qt
import nibabel as nib

from new_utils_3 import (
    load_RT_data, load_CT_data, extract_dose_data, extract_ROIs_data,
    create_ROIs_volume, extract_group1_metadata, extract_group2_metrics,
    get_group2_roi_names, resample_array, preprocess_Dose_to_CT
)
from visualization import plot_Dose_on_CT

class RadiotherapyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiotherapy Metrics Viewer")
        self.resize(1400, 900)

        # Data containers
        self.CT_data = {}
        self.Dose_data = {}
        self.ROIs_data = {}
        self.group1 = {}
        self.group2 = {}
        self.roi_names = []

        # Main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Top buttons
        btn_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Patient Data")
        self.load_button.clicked.connect(self.load_data)
        self.show_vis_button = QPushButton("Show Visualization")
        self.show_vis_button.clicked.connect(self.show_visualization)
        self.export_button = QPushButton("Export Tables as CSV")
        self.export_button.clicked.connect(self.export_tables)
        self.save_nifti_button = QPushButton("Save Volumes as NIfTI")
        self.save_nifti_button.clicked.connect(self.save_volumes)
        for b in (self.load_button, self.show_vis_button, self.export_button, self.save_nifti_button):
            btn_layout.addWidget(b)
        main_layout.addLayout(btn_layout)

        # Status bar
        self.status_label = QLabel("Status: Waiting for user action.")
        main_layout.addWidget(self.status_label)

        # Middle: tables
        mid_layout = QHBoxLayout()
        main_layout.addLayout(mid_layout)

        # ROI metrics table
        left = QVBoxLayout()
        left.addWidget(QLabel("ROI Metrics"))
        self.group2_table = QTableWidget()
        self.group2_table.setColumnCount(7)
        self.group2_table.setHorizontalHeaderLabels([
            "ROI", "Volume [mm3]", "D2% [Gy]", "D50% [Gy]",
            "D98% [Gy]", "HI", "CI"
        ])
        self.group2_table.cellDoubleClicked.connect(self.plot_dvh)
        left.addWidget(self.group2_table)
        mid_layout.addLayout(left, 3)

        # Plan metadata table
        right = QVBoxLayout()
        right.addWidget(QLabel("Plan Metadata"))
        self.group1_table = QTableWidget()
        self.group1_table.setColumnCount(2)
        self.group1_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        right.addWidget(self.group1_table)
        mid_layout.addLayout(right, 2)

        # DVH plot canvas
        main_layout.addWidget(QLabel("DVH Plot:"))
        self.figure, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def load_data(self):
        try:
            self.status_label.setText("Status: Selecting CT DICOM folder...")
            QApplication.processEvents()
            ct_folder = QFileDialog.getExistingDirectory(self, "Select CT DICOM Folder")
            self.status_label.setText("Status: Selecting RT DICOM folder...")
            QApplication.processEvents()
            rt_folder = QFileDialog.getExistingDirectory(self, "Select RT DICOM Folder")
            if not ct_folder or not rt_folder:
                self.status_label.setText("Status: Folders not selected.")
                return

            self.status_label.setText("Status: Loading RT data...")
            QApplication.processEvents()
            RD, RS, RP = load_RT_data(rt_folder)
            self.status_label.setText("Status: Loading CT data...")
            QApplication.processEvents()
            CT = load_CT_data(ct_folder)

            self.status_label.setText("Status: Extracting dose data...")
            QApplication.processEvents()
            Dose = extract_dose_data(RD)
            self.status_label.setText("Status: Extracting ROI data...")
            QApplication.processEvents()
            ROIs = extract_ROIs_data(RS)
            self.status_label.setText("Status: Building ROI volume...")
            QApplication.processEvents()
            ROIs["Volume"] = create_ROIs_volume(ROIs, CT)

            self.status_label.setText("Status: Resampling volumes...")
            QApplication.processEvents()
            vol = CT["Volume"]
            shape = np.array(vol.shape)
            zoom = np.array((512,512,512)) / shape
            CT["Volume"] = resample_array(vol, zoom, order=1)
            ROIs["Volume"] = resample_array(ROIs["Volume"], zoom, order=0)
            CT["Spacing"] = np.array(CT["Spacing"]) / zoom

            self.status_label.setText("Status: Aligning dose with CT...")
            QApplication.processEvents()
            Dose = preprocess_Dose_to_CT(Dose, CT)

            # Store
            self.CT_data, self.Dose_data, self.ROIs_data = CT, Dose, ROIs

            self.status_label.setText("Status: Computing plan metadata...")
            QApplication.processEvents()
            self.group1 = extract_group1_metadata(RP, RD)
            self.display_group1_metrics()

            self.status_label.setText("Status: Computing ROI metrics...")
            QApplication.processEvents()
            pres = self.group1.get("Prescription Dose [Gy]")
            self.group2 = extract_group2_metrics(Dose, ROIs, prescribed_dose=pres)
            self.roi_names = get_group2_roi_names(ROIs)
            self.populate_group2_table()

            self.status_label.setText("Status: Data loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.status_label.setText(f"Error: {str(e)}")

    def display_group1_metrics(self):
        self.group1_table.setRowCount(len(self.group1))
        for i,(k,v) in enumerate(self.group1.items()):
            item_k = QTableWidgetItem(str(k))
            val = QTableWidgetItem(str(round(v,3)) if isinstance(v,(int,float)) else str(v))
            item_k.setFlags(item_k.flags() ^ Qt.ItemIsEditable)
            val.setFlags(val.flags() ^ Qt.ItemIsEditable)
            self.group1_table.setItem(i,0,item_k)
            self.group1_table.setItem(i,1,val)

    def populate_group2_table(self):
        self.group2_table.setRowCount(len(self.roi_names))
        for r,roi in enumerate(self.roi_names):
            d = self.group2[roi]
            row_items = [
                QTableWidgetItem(roi),
                QTableWidgetItem(f"{d['Volume [mm3]']:.3f}" if d['Volume [mm3]'] else "N/A"),
                QTableWidgetItem(f"{d['D2% [Gy]']:.3f}" if d['D2% [Gy]'] else "N/A"),
                QTableWidgetItem(f"{d['D50% [Gy]']:.3f}" if d['D50% [Gy]'] else "N/A"),
                QTableWidgetItem(f"{d['D98% [Gy]']:.3f}" if d['D98% [Gy]'] else "N/A"),
                QTableWidgetItem(f"{d['Homogeneity Index']:.3f}" if d['Homogeneity Index'] else "N/A"),
                QTableWidgetItem(f"{d['Conformity Index']:.3f}" if d['Conformity Index'] else "N/A")
            ]
            for c,item in enumerate(row_items):
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.group2_table.setItem(r,c,item)

    def plot_dvh(self, row, _):
        try:
            roi = self.group2_table.item(row,0).text()
            self.status_label.setText(f"Status: Plotting DVH for {roi}...")
            QApplication.processEvents()
            dvh = self.group2[roi]['DVH']
            bins,counts = dvh['dose_bins'],dvh['voxel_counts']
            samples = [b for b,c in zip(bins,counts) for _ in range(c)]
            if not samples:
                QMessageBox.warning(self,"Warning","Empty ROI.")
                self.status_label.setText("Status: Empty ROI.")
                return
            self.ax.clear()
            edges = np.arange(0,max(bins)+1,1)
            self.ax.hist(samples,bins=edges,edgecolor='black')
            self.ax.set_xlabel("Dose [Gy]")
            self.ax.set_ylabel("Number of Voxels")
            self.ax.set_title(f"Dose Histogram: {roi}")
            self.ax.set_ylim(bottom=0)
            mean_val,med_val = np.mean(samples),np.median(samples)
            self.ax.text(0.95,0.9,f"Mean = {mean_val:.3f}",ha='right',va='top',transform=self.ax.transAxes)
            self.ax.text(0.95,0.8,f"Median = {med_val:.3f}",ha='right',va='top',transform=self.ax.transAxes)
            self.figure.tight_layout()
            self.canvas.draw()
            self.status_label.setText("Status: DVH plotted.")
            QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self,"DVH Error",str(e))
            self.status_label.setText(f"Error: {str(e)}")
            QApplication.processEvents()

    def show_visualization(self):
        try:
            self.status_label.setText("Status: Showing CT/Dose visualization...")
            QApplication.processEvents()
            plot_Dose_on_CT(self.CT_data, self.Dose_data, self.ROIs_data)
            plt.show(block=False)
            self.status_label.setText("Status: Visualization displayed.")
            QApplication.processEvents()
        except Exception as e:
            QMessageBox.critical(self,"Visualization Error",str(e))
            self.status_label.setText(f"Error: {str(e)}")
            QApplication.processEvents()

    def export_tables(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Tables", "", "CSV files (*.csv)")
        if path:
            self.status_label.setText("Status: Exporting tables...")
            QApplication.processEvents()
            try:
                # Save plan metadata
                plan_df = pd.DataFrame(list(self.group1.items()), columns=["Parameter", "Value"])
                plan_df.to_csv(path.replace('.csv', '_plan.csv'), index=False)
                # Save ROI metrics
                roi_df = (
                    pd.DataFrame.from_dict(self.group2, orient='index')
                      .reset_index()
                      .rename(columns={'index': 'ROI'})
                )
                roi_df.to_csv(path.replace('.csv', '_roi.csv'), index=False)
                QMessageBox.information(self, "Export Successful", "Tables exported successfully.")
                self.status_label.setText("Status: Tables exported.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.status_label.setText(f"Error: {str(e)}")
            QApplication.processEvents()

    def save_volumes(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Save NIfTI Files")
        if folder:
            self.status_label.setText("Status: Saving volumes...")
            QApplication.processEvents()
            try:
                nib.save(nib.Nifti1Image(self.CT_data["Volume"], np.eye(4)), os.path.join(folder, "CT_volume.nii.gz"))
                nib.save(nib.Nifti1Image(self.Dose_data["Volume"], np.eye(4)), os.path.join(folder, "Dose_volume.nii.gz"))
                nib.save(nib.Nifti1Image(self.ROIs_data["Volume"], np.eye(4)), os.path.join(folder, "ROIs_volume.nii.gz"))
                QMessageBox.information(self, "Save Successful", "Volumes saved as NIfTI.")
                self.status_label.setText("Status: Volumes saved.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.status_label.setText(f"Error: {str(e)}")
            QApplication.processEvents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RadiotherapyApp()
    window.show()
    sys.exit(app.exec_())