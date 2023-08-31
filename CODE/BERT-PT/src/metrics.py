import numpy as np
from scipy.special import expit
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from transformers.trainer_utils import PredictionOutput
from src.utils import ProblemTypeError


class CustomMetrics:
    
    PROBLEM_TYPES = {"binary": "binary", "multi_class": "micro", "multi_label": "micro"}
    
    def __init__(self, problem_type: str):
        """
        Compute classification metrics for text classification models during fine-tuning for selected problem type.
        
        Parameters
        ----------
        problem_type : str {'binary', 'multi_class', 'multi_label'}
            model classification problem type to perform.
        """
        if problem_type not in self.PROBLEM_TYPES:
            raise ProblemTypeError(f"unsupported problem_type value. available values: {self.PROBLEM_TYPES}")
        else:
            self.problem_type = problem_type
            self.average = self.PROBLEM_TYPES.get(self.problem_type)

    def metrics(self, y_true: np.array, y_pred: np.array) -> dict:
        """
        Return dict with classification metrics for selected problem type.
        
        Parameters
        ----------
        y_true : 1d array-like, or label indicator array / sparse matrix
            ground truth (correct) labels.
        y_pred : 1d array-like, or label indicator array / sparse matrix
            predicted labels, as returned by a classifier.
            
        Returns
        -------
        dict
        """
        metrics_dict = {
            "accuracy": accuracy_score(y_true, y_pred, normalize=True),
            "hamming loss": hamming_loss(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=self.average, zero_division=1),
            "recall": recall_score(y_true, y_pred, average=self.average, zero_division=1),
            "f1 score": f1_score(y_true, y_pred, average=self.average, zero_division=1)
        }
        return metrics_dict
    
    def activation_function(self, logits: np.array) -> np.array:
        """
        Get activation function for selected problem type.
        
        Parameters
        ----------
        logits : array-like of shape (n_samples, n_classes)
            test data computed logits
        
        Returns
        -------
            np.array
        """
        if self.problem_type in ("binary", "multi_class"):
            return softmax(logits, axis=1)
        else:
            return expit(logits)
        
    def compute_prediction(self, y_pred_proba: np.array) -> np.array:
        """
        Get prediction for selected problem type.
        
        Parameters
        ----------
        y_pred_proba : array-like of shape (n_samples, n_classes)
            test data computed probabilities
        
        Returns
        -------
            np.array
        """
        if self.problem_type in ("binary", "multi_class"):
            return np.argmax(y_pred_proba, axis=1)
        else:
            return np.where(y_pred_proba < 0.5, 0, 1)
    
    def compute_metrics(self, eval_predictions: PredictionOutput) -> np.array:
        """
        Compute classification metrics for text classification models during fine-tuning for selected problem type.
        
        Parameters
        ----------
        eval_predictions : PredictionOutput
            huggingface models prediction output
        
        Returns
        -------
            np.array
        """
        logits = eval_predictions.predictions
        y_pred_proba = self.activation_function(logits)
        y_pred = self.compute_prediction(y_pred_proba)
        y_true = eval_predictions.label_ids
        print(classification_report(y_true, y_pred))
        return self.metrics(y_true, y_pred)
