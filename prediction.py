import os
import segmentation
import joblib

# load model
current_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(current_dir, "models/svc/svc.pkl")
model = joblib.load(model_dir)

classification_result = []
for each_character in segmentation.character:
    # converting to 1d image
    each_character = each_character.reshape(1, -1)
    result = model.predict(each_character)
    classification_result.append(result)

print(classification_result)

plate_string = ""
for each_predict in classification_result:
    plate_string += each_predict[0]

print(plate_string)

# possibility we have the wrong order, so we are gonna sort it
coloumn_list_copy = segmentation.coloumn_list[:]
segmentation.coloumn_list.sort()
rightplate_string = ""
for each in segmentation.coloumn_list:
    rightplate_string += plate_string[coloumn_list_copy.index(each)]

print(rightplate_string)
