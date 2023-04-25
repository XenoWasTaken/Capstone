# FirstResponder Assist Bot

## Important Files
- audioSVM.py : loads an SVM model and feeds audio into it. Prints to terminal with success 
- CO2.py : 2 minute initialization, then pings user for next 2 minute reading period.  Reports difference in mean and maximum from current 2 minute period.
- model.joblib : SVM model used for classifying human sounds.  Medium success rate, very simple model.  Some areas of dead code left for TODO/improving later
- test_data.txt : folds used for building model
  
### Testing directories
- ambient : short ambient sounds
- voicelayer : human voice recordings over short ambient sounds
- nonvoicelayer : nonvoice human sounds (coughing, breathing), over short ambient sounds
- quietambient : same as ambient but with less gain, used for short sound removal (currently not in model)


## Misc.
AlgoPrep.ipynb : working ttest document for building model and testing purposes -- sandbox