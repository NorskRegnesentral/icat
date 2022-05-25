
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

  
### Todos
- Test SWAV as SSL pretraing
- Test pl library for ssl
- Test PAWS for SemiSupervised L.