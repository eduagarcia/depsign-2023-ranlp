cache_dir=./RTD
max_seq_length=512
data_dir=$cache_dir/reddit/spm_$max_seq_length

Task=RTD
tag=deberta-v3-large

parameters=" --model_config rtd_large.json \
	--warmup 1000 \
	--learning_rate 5e-5 \
	--train_batch_size 256 \
	--eval_batch_size 32 \
	--predict_batch_size 32 \
	--init_generator deberta-v3-large/pytorch_model.generator.bin \
	--init_discriminator deberta-v3-large/pytorch_model.fix.bin \
	--decoupled_training True \
	--fp16 True \
	--accumulative_update 8"

python -m DeBERTa.apps.run --model_config config.json  \
	--tag $tag \
	--do_train \
	--max_seq_len $max_seq_length \
	--dump 500 \
	--task_name $Task \
	--data_dir $data_dir \
	--vocab_path $cache_dir/spm.model \
	--vocab_type spm \
	--output_dir ./models/${tag}-fix-v4-2 \
	$parameters \
	--num_train_epochs 1.0
#	--num_training_steps 20000


  
