#ICAT: Image Cluster Analysis Tool
A tool for exploring and labelling images that have been clustered.
<img src="library/icat.png">
### Instruction
- make a npz-file with clusters with the following fields
  - 'image_paths': list with path to images
  - 'x': array with x-coordinates of images in cluster-space
  - 'y': array with y-coordinates of images in cluster-space
- Install requirements
```
pip install -r requirements
```
- Run icat:
```
python icat.py -f YOUR_FILE_WITH_DATA.npz
```


### Self-supervision + TSNE clustering
- An example of how to create good clusters is provided in the folder _**demo_ssl_and_clustering**_
- The steps (0,1,2) will take you through the process of 
  1. defining a pytorch dataset
  2. running self-supervised training (SSL) with BYOL or SimCLR on your data
  3. clustering the learned features using TSNE
- Cd into the folder, edit and run each steps 



- Here is a good resource about doing tsne clustering in a good way: 
- https://www.bioinformatics.babraham.ac.uk/training/10XRNASeq/Dimension%20Reduction.pdf
- Folder 'from_sthalles_github' contains cloned repos:
https://github.com/sthalles/PyTorch-BYOL, https://github.com/sthalles/SimCLR, done on january 25th, 2022. Minus a few un-nessecary files.


### Todos:
- Make a CLI for icat
- default to lasso-tool on scatter-plot
- indicate which color that belongs to which class 
- hide labelled points in scatter
- add color to points in catter when labelled
- Test SWAV as SSL pretraing
- Add requirements.txt
