rm -r ./mongodb/log
rm -r ./mongodb/mlpfolder
mkdir ./mongodb/mlpfolder
mkdir ./mongodb/log
mongod --fork --logpath ./mongodb/log/log_mlp_A.txt --dbpath ./mongodb/mlpfolder/

python2 /home/sq/GAN/script/Spearmint/spearmint/main.py --config mlp_config.json ./
