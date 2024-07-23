#download the code
cd ~/research 
git clone https://github.com/JiahuiLei/Pix2Surf pix2surf_master
mkdir   ~/research/pix2surf_master/resource/weight
mkdir   ~/research/pix2surf_master/resource/data 
mkdir   ~/research/pix2surf_master/log

#download pre-trained weights
mkdir weights_temp
cd weights_temp
mkdir pix2surf
cd pix2surf

#weight trianed on plain dataset
wget  http://download.cs.stanford.edu/orion/pix2surf/plain-weight.zip
mkdir plain-weight
unzip plain-weight.zip -d plain-weight 
mv plain-weight/*.model ~/research/pix2surf_master/resource/weight/

#weight trianed for visualization outputs (pix2surf_viz dataset)
wget http://download.cs.stanford.edu/orion/pix2surf/viz-weight.zip
mkdir viz-weight
unzip viz-weight.zip -d viz-weight
mv viz-weight/*.model ~/research/pix2surf_master/resource/weight/

#weight trianed on coco dataset 
wget http://download.cs.stanford.edu/orion/pix2surf/coco-weight.zip
mkdir coco-weight
unzip coco-weight.zip -d coco-weight
mv coco-weight/*.model ~/research/pix2surf_master/resource/weight/