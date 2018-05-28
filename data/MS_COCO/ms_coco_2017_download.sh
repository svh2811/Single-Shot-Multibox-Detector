cd ./Images

echo "Downloading MS COCO 2017 training dataset..."
wget http://images.cocodataset.org/zips/train2017.zip

echo "Downloading MS COCO 2017 validation dataset..."
wget http://images.cocodataset.org/zips/val2017.zip

cd ../Annotations/

echo "Downloading MS COCO 2017 training and validation annotations dataset..."
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Done downloading."

cd ../

echo "Extracting training dataset..."
unzip ./Images/train2017.zip -d ./Images

echo "Extracting validation dataset..."
unzip ./Images/val2017.zip -d ./Images

echo "Extracting training and validation annotations dataset..."
unzip ./Annotations/annotations_trainval2017.zip -d ./Annotations

rm ./Images/train2017.zip
rm ./Images/val2017.zip
rm ./Annotations/annotations_trainval2017.zip
