import synapseclient
import synapseutils
import os

syn = synapseclient.Synapse()
syn.login(authToken="---")
entity = syn.get("syn60868042")

download_path = "./"
files = os.listdir(download_path)
print(files)
entity = syn.get("syn60868042", downloadFile=False)
print(entity)

children = syn.getChildren("syn60868042")
for child in children:
    print(child)


print("################################################")
    
# Synapse IDs
images_folder_id = "syn64871114"
expert_seg_id = "syn64871175"
metadata_id = "syn64854989"
split_csv_id = "syn60880777"

#  Create output directories
os.makedirs("MAMA-MIA/images/precontrast", exist_ok=True)
os.makedirs("MAMA-MIA/images/postcontrast", exist_ok=True)
os.makedirs("MAMA-MIA/segmentations_expert", exist_ok=True)
os.makedirs("MAMA-MIA/metadata", exist_ok=True)

#  1. Load expert segmentations
expert_files = list(syn.getChildren(expert_seg_id))
expert_patient_ids = {f['name'].replace(".nii.gz", "") for f in expert_files}
expert_dict = {f['name'].replace(".nii.gz", ""): f['id'] for f in expert_files}

#  2. Filter image folders by valid prefixes and segmentation match
image_folders = list(syn.getChildren(images_folder_id))
valid_prefixes = ("DUKE", "ISPY1", "ISPY2", "NACT")
image_dict = {f['name']: f['id'] for f in image_folders if f['name'].startswith(valid_prefixes)}

matched = {name: id for name, id in image_dict.items() if name in expert_patient_ids}
skipped = {name: id for name, id in image_dict.items() if name not in expert_patient_ids}

#  3. Download phase 0 and phase 1 images only
for name, folder_id in matched.items():
    print(f" Checking {name}...")
    child_files = list(syn.getChildren(folder_id))

    for f in child_files:
        if "_0000.nii.gz" in f['name']:
            print(f" Precontrast: {f['name']}")
            syn.get(f['id'], downloadLocation="MAMA-MIA/images/precontrast")
        elif "_0001.nii.gz" in f['name']:
            print(f" Postcontrast: {f['name']}")
            syn.get(f['id'], downloadLocation="MAMA-MIA/images/postcontrast")


    # Download expert segmentation
    print(f"  Segmentation: {name}.nii.gz")
    syn.get(expert_dict[name], downloadLocation="MAMA-MIA/segmentations_expert")

#  4. Download metadata
syn.get(metadata_id, downloadLocation="MAMA-MIA/metadata")
syn.get(split_csv_id, downloadLocation="MAMA-MIA/metadata")

#  5. Summary
print("\n Summary:")
print(f" Tumor image folders downloaded: {len(matched)}")
print(f" Non-tumor or unmatched folders skipped: {len(skipped)}")

