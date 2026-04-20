import os 

dataset_path = "/Users/brianmason/Desktop/SafeSleep/datafiles/images"

for filename in os.listdir(dataset_path):
    if filename.endswith(".jpeg"):
        new_filename = filename.replace(".jpeg", ".jpg")
        os.rename(os.path.join(dataset_path, filename), os.path.join(dataset_path, new_filename))

print("All .jpeg files renamed to .jpg")