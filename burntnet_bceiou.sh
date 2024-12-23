#!/bin/bash

# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=coral
# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=cyan
# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=grey
# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=lime
# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=pink
# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=purple


# python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=magenta dataset.batch_size=6 trainer.devices=\[0\]

python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=0 dataset.batch_size=6 trainer.devices=\[0\]
python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=1 dataset.batch_size=6 trainer.devices=\[0\]
python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=2 dataset.batch_size=6 trainer.devices=\[0\]
python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=3 dataset.batch_size=6 trainer.devices=\[0\]
python main.py model_name=burntnet_bceiou model=burntnet_bceiou dataset.test_set=4 dataset.batch_size=6 trainer.devices=\[0\]
