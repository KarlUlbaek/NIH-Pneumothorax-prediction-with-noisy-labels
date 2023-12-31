## Deep learning with noisy labels

The code is based on Ninas repo and in particular her "prediction.py" i.e. her training loop.

The code is pure spaghetti as I spend no time to clean it up and will most likely 
not run on your machine without a bit of tinkering. The reason is simple: I spend way too much time as is
and do not intend spend time on cleaning code and making sure it is reproducible on other machines when I 
doubt that anyone is actually gonna want to run in anyway. However if it is the case that you are interested
in running my code ill gladly help out explaining it or making it run.

Below i have pointed out the content of important/relevant implementations in the code.

\runsNIH-Pneumothorax-fp50-npp1-rs0 contains train.version_0.csv, test.version_0.csv and vali.version_0.csv
which are the precomputed csvs with the dataset metadata used for all experiments.

\NIH\preproc_224x224NIH\preproc_224x224 should contain all the NIH xrays resized to 224x224. As of now it
only contains 4 placeholder images.

\Nina repo\prediction\disease_prediction_ALL.py implements the main training loops and supports all
methods described in the report depending on which flag is set. For instance "method=mixup".

\Nina repo\prediction\gmm_and_mix_class.py implements gmm, mixup and gmm with mixup

\Nina repo\prediction\CL_kfold.ipynb is based on an older version of "disease_prediction_ALL.py" and 
performs 5-fold cross validation and stores the results which subsequently are intended to be used in 
find_outliers.ipynb

\Nina repo\prediction\find_outliers.ipynb is used to find outliers in the data based on confident learning.
These outliers have to be manually saved to "\Nina repo\prediction\CL_FOUND_OUTLIERS.py" as 
list variables. They may then 
be imported as global variables and used in the training in "\Nina repo\prediction\disease_prediction_ALL.py"
to omit (or invert the labels) of said IDS in the training.


\Nina repo\prediction\models.py contains the convnext model imported from hugginface and adapted.

NIH Pneumothorax prediction with noisy labels








