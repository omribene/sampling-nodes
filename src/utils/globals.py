"""
The globals module loads relevant information, including paths to the datasets we used.
We note that the datasets are not included here -- please refer to the relevant references.
We also include default values to the parameters used by our algorithms. These values
are configurable (see the `example_config' folder for examples).
"""

from collections import namedtuple
from pathlib import Path

# Setting up paths to datasets
Dataset = namedtuple("Dataset", ["path", "sep", "title", "directed"])

base = Path(__file__).resolve().parent.parent.parent / "data" / "networks"

datasets = [Dataset(base / f"soc-epinions" / f"soc-Epinions1.txt", "\t", "Epinions", True),
            Dataset(base / f"soc-slashdot" / f"Slashdot0902.txt", "\t", "Slashdot", True),
            Dataset(base / f"com-dblp" / f"com-dblp.ungraph.txt", "\t", "DBLP", False),
            Dataset(base / f"com-youtube" / f"com-youtube.ungraph.txt", "\t", "Youtube", False),
            Dataset(base / f"soc-twitter-higgs" / f"higgs-social_network.edgelist", " ", "Twitter-Higgs", True),
            Dataset(base / f"soc-pokec" / f"soc-pokec-relationships.txt", "\t", "Pokec", True),
            Dataset(base / f"soc-sinaweibo" / f"soc-sinaweibo.mtx", " ", "SinaWeibo", True),
            Dataset(base / f"forest-fire" / f"forest_fire_1M_0_37_0_3.txt", "\t", "ForestFire", False)
            ]

all_num_nodes = {}

nums_L0 = {"Epinions": 3000,
           "Slashdot": 3000,
           "DBLP": 30000,
           "Youtube": 30000,
           "Twitter-Higgs": 25000,
           "Pokec": 200000
           }

nums_L0_plus = {"Epinions": 1000,
                "Slashdot": 2000,
                "DBLP": 20000,
                "Youtube": 10000,
                "Twitter-Higgs": 10000,
                "Pokec": 100000
                }

nums_L1_up = {"Epinions": 1000,
              "Slashdot": 1000,
              "DBLP": 3000,
              "Youtube": 3000,
              "Twitter-Higgs": 3000,
              "Pokec": 3000,
              }

nums_L1_up_plus = {"Epinions": 1000,
                   "Slashdot": 1000,
                   "DBLP": 1000,
                   "Youtube": 1000,
                   "Twitter-Higgs": 1000,
                   "Pokec": 1000,
                   }

nums_L2_reaches = {"Epinions": 200,
                   "Slashdot": 200,
                   "DBLP": 200,
                   "Youtube": 200,
                   "Twitter-Higgs": 200,
                   "Pokec": 200
                   }

nums_L2_reaches_plus = {"Epinions": 100,
                        "Slashdot": 100,
                        "DBLP": 100,
                        "Youtube": 100,
                        "Twitter-Higgs": 100,
                        "Pokec": 100
                        }
