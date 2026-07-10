"""
Source: Impact4cast (Max Planck Institute)
Original code from the Max Planck Institute for Informatics / Max Planck Institute for Security and Privacy research group.

Modifications: Bug fixes for checkpoint resume mechanism and dataset adaptation for scientific entity impact prediction.
"""

import gzip
import pickle

file_path = r"./data/concept_citation\concept_part_150.gz"

#  ... pickle
with gzip.open(file_path, 'rb') as f:
    data = pickle.load(f)

#  ... 
print(f" ... : {type(data)}")
print(f" ... : {data}")