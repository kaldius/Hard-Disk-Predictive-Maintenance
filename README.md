# Hard-Disk-Predictive-Maintenance

Repository of models trained for CS3244 Machine Learning term project. See project paper [here](Report/CS3244_Group_20_Project_Report.pdf).

## Logistic Regression (LogisticRegression.ipynb)
* Attributes used: 'smart_12_normalized', 'smart_189_normalized', 'smart_190_normalized', 'smart_193_normalized',
             'smart_199_normalized', 'smart_240_normalized', 'smart_242_normalized', 'smart_5_normalized',
             'smart_187_normalized', 'smart_188_normalized', 'smart_197_normalized', 'smart_198_normalized'
* Randomly sampled 836 negative examples (non-failed hard disks) because we only have 836 positive examples
* Train: 80%; Test: 20% 
* Score: 0.7014925373134329
* **Lots of false negatives**

## AdaBoost
* Attributes used: all except 'date_x','serial_number','model','failure_x','date_actual_fail'
* Used various oversample/undersample methods (SMOTE, SMOTETomek, ADASYN)
* For AdaBoostSMOTETOMEK(biased data).ipynb, negative:positive ratio is 1:5
