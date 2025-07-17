import os
import h5py
import numpy as np
import pandas as pd

class MRIClassifier:
    """MRI data classifier for anatomy and acceleration"""
    
    def __init__(self, brain_width_range=(390, 400), mask_threshold=95):
        self.brain_width_range = brain_width_range
        self.mask_threshold = mask_threshold
    
    def classify(self, h5_path):
        """Classify MRI data into anatomy and acceleration types"""
        with h5py.File(h5_path, 'r') as f:
            kspace = f['kspace'][:]
            mask = f['mask'][:]
        
        # Anatomy classification
        width = kspace.shape[-1]
        anatomy = 'brain' if self.brain_width_range[0] <= width <= self.brain_width_range[1] else 'knee'
        
        # Acceleration classification
        ones_count = int(np.sum(mask))
        acceleration = 'acc4' if ones_count >= self.mask_threshold else 'acc8'
        
        return {
            'anatomy': anatomy,
            'acceleration': acceleration, 
            'class_label': f"{acceleration}-{anatomy}"
        }

def classify_and_index(input_folder, output_base, brain_width_range=(390, 400), mask_threshold=95, print_freq=50):
    """
    Classify MRI files and create index files for each class
    
    Returns:
        dict: {class_label: [file_list]}
        list: paths to generated index files
    """
    classifier = MRIClassifier(brain_width_range, mask_threshold)
    
    # Step 1: Classify all files
    records = []
    files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    
    print(f"Classifying {len(files)} files...")
    for idx, fname in enumerate(sorted(files)):
        if idx % print_freq == 0:
            print(f"Progress: {idx}/{len(files)}")
        
        path = os.path.join(input_folder, fname)
        classification = classifier.classify(path)
        records.append({
            'file': fname,
            'anatomy': classification['anatomy'],
            'acceleration': classification['acceleration'],
            'class': classification['class_label']
        })
    
    # Step 2: Group by class and save index files
    df = pd.DataFrame(records)
    class_groups = df.groupby('class')['file'].apply(list).to_dict()
    
    os.makedirs(output_base, exist_ok=True)
    index_files = []
    
    for class_label, file_list in class_groups.items():
        index_path = os.path.join(output_base, f"{class_label}.txt")
        index_files.append(index_path)
        
        with open(index_path, 'w') as f:
            for fname in file_list:
                f.write(fname + '\n')
        
        print(f"Created index: {index_path} ({len(file_list)} files)")
    
    # Step 3: Print statistics
    total = sum(len(files) for files in class_groups.values())
    print("\n=== MoE Classification Statistics ===")
    for class_label, file_list in class_groups.items():
        count = len(file_list)
        percentage = (count/total)*100
        print(f"{class_label}: {count} files ({percentage:.1f}%)")
    print("=====================================")
    
    # Save summary CSV
    csv_path = os.path.join(output_base, "classification_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved classification summary: {csv_path}")
    
    return class_groups, index_files

# Backward compatibility (기존 함수명 유지)
def classify_and_split(input_folder, output_base, brain_width_range=(390, 400)):
    """Backward compatibility wrapper"""
    return classify_and_index(input_folder, output_base, brain_width_range)

# Example usage
# if __name__ == '__main__':
#     input_folder = "/root/Data/train/kspace"
#     output_base = "/root/Data/train/class_indices"
#     class_groups, index_files = classify_and_index(input_folder, output_base)
