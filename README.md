# NAPH-Napari-Anotation-Helper-
NAPH  is a Napari plugin that was created as part of my Ph.D. project. Its objective is to help developers reduce annotation time and errors. 

Napari Annotation Helper or NAHP is a Python-developed Napari plugin using magicgui (Magicgui, 2020) and pyqt5 (Riverbank Computing, n.d.) Python packages. NAHP’s objective is to assist users in performing annotation faster and more easily by providing a set of functions that will help them avoid annotation errors, object-label mismatch, label repetition, and confusion when tracking the labels added to an image.

NAHP uses 2D segmented image stacks (preferably with instance labels) as input to generate 3D annotations. As this 2D segmentation step is not incorporated into NAHP, users are free to use any segmentation method they want as a starting point. Typically, I used my trained StarDist 2D model to get these 2D segmented image stacks.

NAHP consists of three main sections: Labels layer features, Functions to facilitate annotation and Annotation helper extra functions

## Labels layer features
NAHP requires loading two images, the one to annotate and the one containing the previously 2D segmentations. The image loading functions are available through two widgets called "Fluorescence" and "Labels". Accessible through a drop-down button, the user can choose one base color (red, blue, magenta, orange, yellow, or white) that will be used for all the 2D labels. Simultaneously, a look-up table (LUT) is created whose purpose is to generate colors that are visually as far away from the chosen base color. Thanks to it, every new 3D label will have a different color that will be distinguishable from the 2D segmentation. In spirit of inclusion, I also added the possibility to generate a color-blind LUT.

<img src="https://github.com/user-attachments/assets/4d0c81e6-8695-405f-a092-08f8003c6481" width=50% height=50%>

## Functions to facilitate annotation 
To develop a plugin that makes annotating easier, I wanted to address practical situations that frequently resulted in errors. I mainly identified three situations: label tracking, label localization, and error localization. 

**Label tracking**:I added a widget that keeps track of all the 3D labels already added and that only displays the last one added. In addition, I added a second widget whose purpose is to identify 2D labels that still must be re-labeled as 3D labels in the current frame. The idea is to help identify labels in crowded regions or very small labels (a few pixels) that are otherwise very hard to locate.

**Label localization.**: During annotation, it is often necessary to find a specific annotated item, whether to correct it, delete it, or continue with its annotation. The “Label coordinates” widget prints the first five coordinates of any label chosen by the user, whether it be 2D or 3D. 

<img src="https://github.com/user-attachments/assets/9dccfd2e-13c1-47bc-9f7e-6a651c40da68" width=60% height=60%> 

**Error localization**: Juggling with labels makes it very easy for human errors. The two most common errors are two objects sharing the same label or a missing label within a labeled object. These errors are easy to identify, as one would expect one connected component per object. I therefore implemented a function that screens all the 3D labels, identifies the ones that are not contiguous in the x, y, or z direction, and prints them

<img src="https://github.com/user-attachments/assets/55f82192-8c27-462b-aeae-41a7f3db252b" width=40% height=40%>

I expect that these 4 functions will help reduce both the number of errors and the time spent annotating.

## Annotation helper extra functions

**Nuclei individualization**: Being able to individualize the labeled nuclei is important as it can help with several tasks such as examining the labeling quality or generating simulations. Therefore, I decided to add a function that individualizes the nuclei and cleverly organizes them for visualization. This function requires the number of labels to individualize and generates 3 layers: one for the label itself, one for the corresponding part of the fluorescence images and one for the bounding box

<img src="https://github.com/user-attachments/assets/9688d4f5-8984-48ff-9d5a-33b8f5f1097d" width=40% height=40%>

**Nuclei localization**: Jointly to the individualization process, I added the possibility of localizing the individualized nuclei directly in the original fluorescence image stack. This is done by duplicating the label layer and setting all the non-individualized labels to 0, making them invisible.

**Properties table**: This function computes, for each 3D label, several quantitative measurements. It uses both the labeled and fluorescence image stacks and displays in a panda-compatible table several features: label id, area, centroid, bbox (bounding box), intensity maximum, intensity mean, and intensity minimum. 

**Centroid localization**: Finally, this function uses the centroid column computed in the “Properties table” and creates a new point layer, allowing the display of the centroids directly on the fluorescence image stack.

<img src="https://github.com/user-attachments/assets/ae112dea-77f6-4e93-9f5e-b173209581f0" width=40% height=40%>


**If you want to read my thesis**: https://theses.hal.science/tel-03982040/

**To cite this repository** 
Xareni Galindo.3Ddeep-learningbasedquantificationofthenucleidistributionforlight-sheetmi-
croscopy.Bioinformatics[q-bio.QM].UniversitédeBordeaux,2022.English..NNT:2022BORD0275..
.tel-03982040.
