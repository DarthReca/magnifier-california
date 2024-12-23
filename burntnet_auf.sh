#!/bin/bash

# python main.py model_name=burntnet_auf model=burntnet_auf dataset.test_set=coral dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset.test_set=cyan dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset.test_set=grey dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset.test_set=lime dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset.test_set=pink dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset.test_set=purple dataset.batch_size=6 trainer.devices=\[1\]

python main.py model_name=burntnet_auf dataset=europe model=burntnet_auf dataset.test_set=magenta dataset.batch_size=6 trainer.devices=\[1\]


# python main.py model_name=burntnet_auf model=burntnet_auf dataset=california2 dataset.test_set=0 dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset=california2 dataset.test_set=1 dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset=california2 dataset.test_set=2 dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset=california2 dataset.test_set=3 dataset.batch_size=6 trainer.devices=\[1\]
# python main.py model_name=burntnet_auf model=burntnet_auf dataset=california2 dataset.test_set=4 dataset.batch_size=6 trainer.devices=\[1\]
