cd ~/research

# create virtual env (/home/ubuntu/anaconda3/envs/pix2surf)
conda create -n pix2surf python=3.6
source activate pix2surf

# install pytorch with corresponding cuda version, for example we use cuda 10.0 here
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch

# install other requirements
cd pix2surf_master
pip install -r ./requirements.txt
# install tk3dv with --ignore-installed flag to avoid Error: Cannot uninstall 'certifi'
pip install  --ignore-installed git+https://github.com/drsrinathsridhar/tk3dv.git 

#substitute tk3dv file that cause OpenGL error
sudo cp ~/drawing.py ~/anaconda3/envs/pix2surf/lib/python3.6/site-packages/tk3dv/common/drawing.py 