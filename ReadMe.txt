This zip file contains source code Of CMEpatch. The code uses a Transformer-based model (PatchTST) to classify solar activity over a 24-hour time window.

The usage instructions are as follows:
	python3 CME_patchTST.py patchTST 24 0

The first argument "CME_patchTST.py" is Python script filename. # Main Python script containing the complete pipeline — including data loading, training, Testing, and prediction . 

The second argument "patchTST"  Specifies the model to use for prediction 
The third argument "24" Predicts CMEs occurring within the next 24 hours.

The fourth argument 0 / 1 -> Controls model behavior:

0 : Loads and uses a pre-trained model (patchTST-24-model.h5 or lstm-24-model.h5).
1 : Retrains the model before making predictions.

After execution, the output predictions are stored in:
patchTST-24-output.csv



- Setup Python Environment: program is run on Python version 3.9.7 

- Install Required Packages: To install all the necessary packages run "pip install -r requirements.txt" avaibale in the zip file.

- Handling AdamW Error (PatchTST): If you encounter an error related to AdamW while using the PatchTST model, install TensorFlow Addons: "pip install tensorflow-addons" 