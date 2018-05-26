echo "Downloading VOC2007 trainval..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar

echo "Downloading VOC2007 test data ..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

echo "Done downloading."

echo "Extracting trainval ..."
tar -xvf VOCtrainval_06-Nov-2007.tar

echo "Extracting test ..."
tar -xvf VOCtest_06-Nov-2007.tar

echo "removing tars ..."

rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar