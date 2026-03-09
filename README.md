## xai group assignment repo

data used for the project is under data/ dir, we included the license found online for the .arff to .csv convertor (though you can fetch this data with sklearn directly also)

scripts are all under their respective names, you should only run main.py to get the results of each fine-tuned classifier, as well as the results of the application of XAI to these algorithms. 

The neural network uses LIME and SHAP, the RF uses treeSHAP

Code started from a template we manually wrote through looking at the documentation, then Claude Sonnet 4.5 was used to extend it to include RF and decision trees, to save time. Agent also helped setting up LIME and SHAP where applicable. 

