# Import classes
from mmexofast.gridsearches import EventFinderGridSearch, AnomalyFinderGridSearch, ParallaxGridSearch
from mmexofast.results import MMEXOFASTFitResults, FitRecord, AllFitResults, GridSearchResult
from mmexofast.mmexofast import MMEXOFASTFitter, WorkflowStep, OutputConfig
from mmexofast.classifier import AnomalyClassifier

from os import path

MODULE_PATH = path.abspath(__file__)
for i in range(3):
    MODULE_PATH = path.dirname(MODULE_PATH)

path_1 = path.join(MODULE_PATH, 'data')
if path.isdir(path_1):
    DATA_PATH = path_1
else:
    DATA_PATH = path.join(path.dirname(__file__), 'data')
