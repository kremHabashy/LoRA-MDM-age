import os

# Define the base folder path
base_folder = "/u1/khabashy/LoRA-MDM/humanact12_motion_analysis"

# List the subdirectories where you need __init__.py files
subdirectories = [
    "",  # For the base folder itself
    "kinematics",
    "kinematics/range_of_motion",
    "kinematics/velocity_acceleration",
    "foot_contact",
    "foot_contact/contact_percentage",
    "foot_contact/contact_intervals",
    "motion_energy",
    "motion_energy/frame_energy",
    "intra_class_variability",
    "intra_class_variability/trajectory_variance",
    "umap",
    "umap/motion_umap"
]

# Create an empty __init__.py file in each listed folder
for sub in subdirectories:
    folder = os.path.join(base_folder, sub)
    init_file = os.path.join(folder, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Package initialization\n")
print("Created __init__.py files in all required folders.")
