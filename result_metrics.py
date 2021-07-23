from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

def calculatemetrics(y_orig,y_pred):
  
  resultados = {}
  
  resultados['accuracy'] = accuracy_score(y_orig, y_pred)
  resultados['precision'] = precision_score(y_orig, y_pred, average=None)
  resultados['recall'] = recall_score(y_orig, y_pred,average=None)
  resultados['f1'] = f1_score(y_orig, y_pred,average='weighted')
  resultados['kappa'] = cohen_kappa_score(y_orig, y_pred)

  return resultados