python ./proxy_dataset_poisoning.py --proxy_epoch 0


python ./proxy_dataset_poisoning.py --proxy_dataset imdb --proxy_max_len 256 --dataset imdb --max_len 256
python ./proxy_dataset_poisoning.py --proxy_dataset imdb --proxy_max_len 256 --dataset imdb --max_len 256 --training_method freelb > freelb_log.txt
