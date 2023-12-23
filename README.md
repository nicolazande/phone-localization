# phone-localization
![find_phone](https://github.com/nicolazande/phone-localization/assets/115359494/53ed21c6-8fb6-4b45-acfe-8e9fe9fc580d)

## Instructions to run the code
1. Run "train_phone_finder.py": takes as input the path of the folder where the images training set is and, eventually, builds the model, trains it and saves new weights. At the beginning of the script there are few parameters (save_weights, start_training, prepare_patches, load_weights, apply_data_augmentation) which are by default to False. This is due to the fact that I already included all the components to run "find_phone.py" in the folder but by setting those parameters to True you can create new patches from the original test set, tain the model (standard 10 epochs) and saving the weights for "find_phone.py".
2. Run "find_phone.py": takes as input the path of the test image and gives as output the relative coordinates of the center of the phone. This script simply builds the same model as the one in "train_phone_finder.py", loads the previously obtained weights and makes the prediction.

## Libraries used
- all standard python libraries (numpy, json, PIL, ...)
- tensorflow
- keras

## Explanation of choices
To solve this problem I decided to perform a weakly supervised localization with CAM because the prediction involved just the center of the phone (no segmentation or bounding box was needed).
To do this I divided the original images in patches with and without phone inside and performed a simple binary image classification. I decided not to do transfer learning because it was not necessary and I wanted the net to be small (the results using VGG16 as encoder were almost the same).
After obtaining almost perfect accuracy and low loss on classification (I used only the images you gave me even if I think that by adding also different images in the "no-phone-set" the results could be better) I used the GAP layer after the convolutional part to obtain the heatmap that indicates where most likely the phone is. After resizing the eatmap I selected the indeces of the numpy array that contained the highest values because they correspond to the pixels in the original image where the phone should be. By averaging and normalizing tham I obtain the relative coordinates ot he phone center.
