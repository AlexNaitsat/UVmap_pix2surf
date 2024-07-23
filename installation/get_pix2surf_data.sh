#download data
cd ~/research/datasets
# ShapeNet-plain data (5G)
wget http://download.cs.stanford.edu/orion/xnocs/shapenetplain_v1.zip
unzip shapenetplain_v1.zip
rm shapenetplain_v1.zip

# dataset for pix2surf visualizaion (840MB)
wget http://download.cs.stanford.edu/orion/pix2surf/pix2surf_viz_dataset.zip
unzip pix2surf_viz_dataset.zip
rm  pix2surf_viz_dataset.zip




##ShapeNet-COCO data (170G)
#wget http://download.cs.stanford.edu/orion/xnocs/shapenetcoco_v1.zip
#unzip shapenetcoco_v1.zip
#rm shapenetplain_v1.zip

#link data with the code (use absolute path to data)
cd ~/research/pix2surf_master/resource/data
ln -s /home/ubuntu/research/datasets/shapenetplain_v1/ ./shapenet_plain
#ln -s /home/ubuntu/research/datasets/shapenet_coco_v1/ ./shapenet_coco
ln -s /home/ubuntu/research/datasets/pix2surf_viz/ ./pix2surf_viz

# do some minor modification to the dataset structure
cd ~/research/pix2surf_master/resource/utils
source activate pix2surf
python make_coco_validation.py