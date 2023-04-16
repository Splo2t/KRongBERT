deepspeed --num_gpus=4 --num_nodes 1 --master_addr 110 bert_train.py --deepspeed ds_config.json
