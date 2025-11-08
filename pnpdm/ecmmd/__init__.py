from pnpdm.ecmmd.knn import KNN
from pnpdm.ecmmd.ecmmd import ECMMD

def get_knn(k):
    return KNN(k=k)

def get_ecmmd(**kwargs):
    return ECMMD(**kwargs)
