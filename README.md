# APAYNG
## Anatomy pics are all you need on Gadgetron

This is a re-implementation of our study ["Automated analysis and detection of abnormalities in transaxial anatomical cardiovascular magnetic resonance images: a proof of concept study with potential to optimize image acquisition"](https://link.springer.com/article/10.1007/s10554-020-02050-w) in the The International Journal of Cardiovascular Imaging.

![Method summary](https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs10554-020-02050-w/MediaObjects/10554_2020_2050_Fig2_HTML.png?as=webp)

Within the first couple of minutes of a CMR scan, a series of transaxial images are routinely acquired, commonly termed the “anatomy” sequences.

Here we use a neural network to analyse and segment each slice of these images, before combining these slices into a 3D model. This 3D model is then used to estimate common cardiac measurements made by expert humans. The hope is we can get good estimates of these measures right at the start of the scan, so any unexpected findings can be flagged up early.

The Onnx trained model used for inference is available [here for CPU](https://james.dev/apayng_models/78_0.79991.pt_cpu.onnx) and [here for CUDA](78_0.79991.pt_cuda.onnx)

`inference.pynb` shows an example where a folder containing a typical anatomy sequence is supplied to the model:
![Inference example](https://james.dev/apayn_example.png)

The following files are also available in the repository:

* `1_train.py` - Training code to train the Pytorch model

* `2_export_onnx.py` - Code to export the Onnx model (link to above)

* `3_inference_on_all_studies.py` - Batch inference on a series of studies, with results exported as a CSV file.

* `4_compare_predictions_to_groundtruth.py` - The results from Step 3 can be plotted as a scatter plot versus gold standard predictions.
