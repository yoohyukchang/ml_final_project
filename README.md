Due to model size, we could not upload trained models to the repo. Therefore, we uploaded it to OneDrive and required a JHU email to access it. We did this to protect the file as
best as possible while allowing faculty and TAs access. They can edit this folder, so please reach out to rmcgove3@jhu.edu if you cannot see either trained model.
Link to model files: https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/rmcgove3_jh_edu/EjZ2-JlkbWtCifG25DSOPCUBuXxe5bojtOBJIAhaZ1aSnQ?e=dbOwEI

# Project Setup
- We recommend using `virtualenv` library from python. If it's not installed on your computer, you can `pip install virtualenv` or `pip3 install virtualenv` if you are using Python3.
- After the installation, on the root directory, type `virtualenv env` to create a virtual environment folder
- Then, do `source env/bin/activate` to activate your virtual environment.
- Then please install libraries using `pip install -r requirements.txt`.
- You can always deactivate your virtual environment using `deactivate` command.

# Data
Inside the `data` directory from the root, you will see there is a 'Test' directory. You need to visit `https://livejohnshopkins-my.sharepoint.com/personal/rmcgove3_jh_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Frmcgove3%5Fjh%5Fedu%2FDocuments%2FML%5FTest%5FFiles&ga=1` to download the test data. These test data are the actual unseen data. To get the result, you need to download Cropped_Legions, Cropped_MRIs, Legions, MRIs, Prostates data and place inside the Test directory correspondingly.

# Models
Please go to [`https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/rmcgove3_jh_edu/EjZ2-JlkbWtCifG25DSOPCUBuXxe5bojtOBJIAhaZ1aSnQ?e=dbOwEI`](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/rmcgove3_jh_edu/EjZ2-JlkbWtCifG25DSOPCUBuXxe5bojtOBJIAhaZ1aSnQ?e=dbOwEI) to download `UNET_Transformer_model_prostate.pt` and put it inside the `model1_prostate_segmentation` directory. Just like that, download `UNET_Transformer_model_legion.pt` and put it inside the `model2_lesion_segmentation` directory.

# Running the Model (Get Result)
To run the model, you need to run `predict.py` files.

To get the result of our first model (prostate_segmentation_model), you should run `python model1_prostate_segmentation/predict.py`. After running this script, you will have `predicted_labels` directory inside `model1_prostate_segmentation` directory. From this, `segmentation` is the predicted segmentation of the prostate. If you want to view the result visually, we recommend you download the `3D Slicer` software from https://www.slicer.org/. Along with the resulting segmentation, you will also get the dice score and Haufdorff distance metric in the terminal. You can also find the metric result inside the `predicted_labels` directory, which is `evaluation_metrics.txt`. Dice and Haufdorff are used as our metric.

To get the rsult of our second model(lesion_segmentation_model), you should run `python model2_lesion_segmentation/predict.py`. After running this script, you will have `predicted_labels` directory inside `model2_lesion_segmentation` directory. From this, `segmentation` is the predicted segmentation of the lesion. If you want to view the result visually, we recommend you download the `3D Slicer` software from https://www.slicer.org/. Along with the resulting segmentation, you will also get the dice score and Haufdorff distance metric in the terminal. You can also find the metric result inside the `predicted_labels` directory, which is `evaluation_metrics.txt`. Dice and Haufdorff are used as our metric.