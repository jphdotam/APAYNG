# APAYNG
## Anatomy pics are all you need on Gadgetron

This is a re-implementation of our study ["Automated analysis and detection of abnormalities in transaxial anatomical cardiovascular magnetic resonance images: a proof of concept study with potential to optimize image acquisition"](https://link.springer.com/article/10.1007/s10554-020-02050-w) in the The International Journal of Cardiovascular Imaging.

The Onnx model is available ["here"](https://james.dev/apayng_80_0.80656.pt.onnx)

`inference.pynb` shows an example where a folder containing a typical anatomy sequence is supplied to the model:
![Inference example](https://james.dev/apayn_example.png)

The following files are also available in the repository:

* `1_train.py` - Training code to train the Pytorch model

* `2_export_onnx.py` - Code to export the Onnx model (link to above)

* `3_inference_on_all_studies.py` - Batch inference on a series of studies, with results exported as a CSV file.

* `4_compare_predictions_to_groundtruth.py` - The results from Step 3 can be plotted as a scatter plot versus gold standard predictions. Plots from our data are shown below:
