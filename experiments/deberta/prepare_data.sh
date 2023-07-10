cache_dir=./RTD

max_seq_length=512
data_dir=$cache_dir/reddit/spm_$max_seq_length

corpora_path=/raid/juliana/depsign/external_data/reddit-corpora

mkdir -p $data_dir
python ./prepare_data.py -i ${corpora_path}-train.txt -o $data_dir/train.txt --max_seq_length $max_seq_length
python ./prepare_data.py -i ${corpora_path}-test.txt -o $data_dir/valid.txt --max_seq_length $max_seq_length
cp $data_dir/valid.txt $data_dir/test.txt