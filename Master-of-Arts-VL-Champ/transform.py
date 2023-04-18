img_dir = 'train/'

# Define the directory to save the transformed images
save_dir = 'train/'
import cv2
# Define the transformations to apply
transformations = [
    ('rotate_90', Image.ROTATE_90),
    ('rotate_180', Image.ROTATE_180),
    ('rotate_270', Image.ROTATE_270),
    ('flip_left_right', Image.FLIP_LEFT_RIGHT),
    ('flip_top_bottom', Image.FLIP_TOP_BOTTOM),
    ('vertical_flip', cv2.flip(img, 0)),
    ('horizontal_flip', cv2.flip(img, 1)),
    ('zoom', cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)),
    ('outzoom', cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC))
    ('transpose', Image.TRANSPOSE)
]

# Iterate over all images in the directory
for img_file in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_file)
    with Image.open(img_path) as img:
        # For each transformation, apply it to the image and save the result
        for t_name, t_func in transformations:
            t_img = img.transpose(t_func)
            t_img.save(os.path.join(save_dir, f'{t_name}_{img_file}'))

# Define the csv file that contains the labels for each image
csv_path = 'train.csv'

# Read the labels from the csv file into a pandas dataframe
df = pd.read_csv(csv_path, sep='\t')

# Create a list of the original image filenames
orig_files = [fname for fname in df['image_name'].tolist()]

# Create a list of the transformed image filenames
trans_files = []
for t_name, _ in transformations:
    trans_files += [f'{t_name}_{fname}' for fname in orig_files]

# Create a list of the labels for the transformed images
trans_labels = [label_id for label_id in df['label_id'].tolist() for _ in range(len(transformations))]

# Add the transformed images and their labels to the dataframe
df_trans = pd.DataFrame({'image_name': trans_files, 'label_id': trans_labels})
df = pd.concat([df, df_trans], ignore_index=True)

# Write the updated dataframe to the csv file
df.to_csv(csv_path, sep='\t', index=False)

class ArtDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):

        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["label_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = Image.open(self.files[idx]).convert('RGB')
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target
