echo "Downloading VOC 2012 trainval..."
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

echo "Done downloading."

echo "Extracting trainval ..."
tar -xvf VOCtrainval_11-May-2012.tar

echo "removing tars ..."

rm VOCtrainval_11-May-2012.tar