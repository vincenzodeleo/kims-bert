import numpy as np
from scipy.special import expit
from scipy.special import softmax

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch import qint8
from torch.nn import Linear
from torch.quantization import quantize_dynamic
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from src.utils import ProblemTypeError

from typing import Any
from typing import Union


class AutoBertClassifier:
    
    PROBLEM_TYPES = {"binary": "binary", "multi_class": "weighted", "multi_label": "samples"}
    
    def __init__(self, model_path: str, problem_type: str):
        """
        Load trained model and perform text classification on new data.
        
        Parameters
        ----------
        model_path : str
            trained model path
        problem_type : str {'binary', 'multi_class', 'multi_label'}
            model classification problem type to perform.
        """
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if problem_type not in self.PROBLEM_TYPES:
            raise ProblemTypeError(f"unsupported problem_type value. available values: {self.PROBLEM_TYPES}")
        else:
            self.problem_type = problem_type
            self.average = self.PROBLEM_TYPES.get(self.problem_type)

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
            model computed probabilities
        
        Returns
        -------
            np.array
        """
        if self.problem_type in ("binary", "multi_class"):
            return np.argmax(y_pred_proba, axis=1)
        else:
            return np.where(y_pred_proba < 0.5, 0, 1)
        
    def __compute_proba_labels(self, y_pred_proba: np.array) -> dict:
        """
        Compute model probability labels.
        
        Parameters
        ----------
        y_pred_proba : array-like of shape (n_samples, n_classes)
            model computed probabilities
            
        Returns
        -------
        dict
        """
        return dict(zip(self.model.config.id2label.values(), y_pred_proba))
    
    def __compute_pred_labels(self, y_pred) -> Union[np.array, list]:
        """
        Compute model prediction labels.
        
        Parameters
        ----------
        y_pred : np.array
            model computed predictions
            
        Returns
        -------
        np.array | list
        """
        if self.problem_type in ("binary", "multi_class"):
            return np.vectorize(pyfunc=self.model.config.id2label.get)(y_pred)
        else:
            y_pred = list(map(self.__compute_proba_labels, y_pred))
            return [[k for k, v in d.items() if v==1] for d in y_pred]
        
    def predict_proba(self, texts: list, batch_size: int = 16, labels: bool = False) -> Union[np.array, list]:
        """
        Probability estimates.
        
        Parameters
        ----------
        texts : list
            input texts to classify
        batch_size: int, default=16
            the number of samples to load at once
        labels : bool, default=False
            if True, returns probabilities with labels
            
        Returns
        -------
        np.array | list
        """
        if isinstance(texts, str):
            texts = [texts]     
        y_pred_probas = []
        data_loader = DataLoader(texts, batch_size=batch_size)
        with torch.no_grad():
            for batch in tqdm(data_loader):
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                logits = self.model(**inputs).logits
                y_pred_probas.append(logits.cpu())
        y_pred_probas = np.concatenate(y_pred_probas)
        y_pred_proba = self.activation_function(y_pred_probas)
        if labels:
            return list(map(self.__compute_proba_labels, y_pred_proba))
        else:
            return y_pred_proba
        
    def predict(self, texts: list, batch_size: int = 16, labels: bool = False) -> Union[np.array, list]:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        texts : list
            input text to classify
        batch_size : int, default=16
            the number of samples to load at once.
        labels : bool, default=False
            if True, returns predicted labels
            
        Returns
        -------
        np.array | list
        """
        y_pred_proba = self.predict_proba(texts=texts, batch_size=batch_size)
        y_pred = self.compute_prediction(y_pred_proba)
        if labels:
            return self.__compute_pred_labels(y_pred)
        else:
            return y_pred


def quantize_model(model: Any) -> Any:
    """
    Converts a float model to dynamic (i.e. weights-only) quantized model.
    
    Parameters
    ----------
    model : Any
        model to quantize
    
    Returns
    -------
        Any
    """
    return quantize_dynamic(model=model.to("cpu"), qconfig_spec={Linear}, dtype=qint8)


def save_model_and_tokenizer(model: Any, tokenizer: Any, save_directory: str) -> None:
    """
    Save a model and its configuration with tokenizer files to a directory.
    
    Parameters
    ----------
    model : Any
        model to save
    tokenizer : Any
        tokenizer to save
    save_directory : str
        directory to which to save (will be created if it doesn't exist)

    Returns
    -------
        None
    """
    model.save_pretrained(save_directory=save_directory)
    tokenizer.save_pretrained(save_directory=save_directory)
