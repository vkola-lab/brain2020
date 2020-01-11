
rm -r mongodb/dbfolder
mkdir mongodb/dbfolder
rm    mongodb/log/log.txt
mongod --fork --logpath mongodb/log/log.txt --dbpath mongodb/dbfolder/

#rm -r mongodb/dbfolder82
#mkdir mongodb/dbfolder82
#rm    mongodb/log/log82.txt
#mongod --fork --logpath mongodb/log/log_82.txt --dbpath mongodb/dbfolder82/

#rm -r mongodb/dbfolder3
#mkdir mongodb/dbfolder3
#rm    mongodb/log/log3.txt
#mongod --fork --logpath mongodb/log/log_3.txt --dbpath mongodb/dbfolder3/

python2 /home/sq/GAN/script/Spearmint/spearmint/main.py --config fcn_config.json ./
