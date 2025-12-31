import os  # file directory
import numpy as np  # for image and feature arrays
from sklearn import SVC  # for character classifier
from sklearn.model_selection import cross_val_score  # modelevaluation
from sklearn.externals import joblib  # load trained model
from skimage.io import imread  # read images
from skimage.filters import threshold_otsu  # binarzation

letters = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def read_training_data(training_directory):
    image_data = []  # flattened images
    target_data = []  # correct labels ie. A, O , 3 etc
    for each_letter in letters:
        for each in range(10):  # 10 training images per each letter
            image_path = os.path.join(
                training_directory, each_letter, each_letter + "_" + str(each) + ".jpg"
            )
            # Reads each image for each character
            img_details = imread(image_path, as_gray=True)
            # converts each character into binary
            binary_image = img_details < threshold_otsu(img_details)
            # converts our image into a 1D shape
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)
    return (np.array(image_data), np.array(target_data))


def cross_validation(model, num_of_fold, train_data, train_label):
    acuracy_result = cross_val_score(model, train_data, train_label, cv=num_of_fold)
    print(f"Cross validation for {str(num_of_fold)} -fold")
    print(acuracy_result * 100)


current_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(current_dir, "train")
image_data, target_data = read_training_data(training_dataset_dir)

# the kernel can be 'linear', 'poly' or 'rbf'
svc_model = SVC(kernal="linear", probability=True)
cross_validation(svc_model, 4, image_data, target_data)

# train the model with input data
svc_model.fit(image_data, target_data)

save_directory = os.path.join(current_dir, "models/svc/")
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
joblib.dump(svc_model, save_directory + "/svc.pkl")
