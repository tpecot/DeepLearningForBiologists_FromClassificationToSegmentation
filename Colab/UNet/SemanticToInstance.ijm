/*
 * Macro template to process multiple images in a folder
 */

// input parameters
#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tiff") suffix

// call to the main function "processFolder"
processFolder(input);

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	///////////// initial cleaning /////////////////
	// close all images
	run("Close All");

	///////////// apply pipeline to input images /////////////////
	// get the files in the input folder
	list = getFileList(input);
	list = Array.sort(list);
	// loop over the files
	for (i = 0; i < list.length; i++) {
		// if there are any subdirectories, process them
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		// if current file ends with the suffix given as input parameter, call function "processFile" to process it
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
	
	///////////// save the extracted features /////////////////
	saveAs("Results", output+"/Results.csv");
}

function processFile(input, output, file) {

	///////////// parameters /////////////////////
	threshold_on_background = 0.95;
	threshold_on_nuclei_minus_contours = 0.3;
	///////////// define nuclei segmentation masks as ROIs /////////////////
	// open image
	open(input+"/"+file);
	// rename image
	rename("Input");
	// split channels
	run("Split Channels");
	// select channel 3 (background), high scores correspond to objects 
	selectWindow("C3-Input");
	setThreshold(0, threshold_on_background);
	run("Convert to Mask");
	rename("Object component");
	// remove small objects (<10 pixels)
	run("Gray Scale Attribute Filtering", "operation=Opening attribute=Area minimum=10 connectivity=4");
	// divide intensity by 255 to get a binary image
	run("Divide...", "value=255");
	rename("ObjectComponent");

	// subtract nuclei component from nuclei contours
	imageCalculator("Subtract create 32-bit", "C2-Input","C1-Input");
	// threshold the subtraction
	setThreshold(threshold_on_nuclei_minus_contours, 2.0000);
	run("Convert to Mask");
	// extract connected components to identify individual nuclei
	run("Connected Components Labeling", "connectivity=4 type=[16 bits]");
	// apply 3D Voronoi to expand nuclei components 
	run("3D Watershed Voronoi", "radius_max=0");
	// multiply Voronoi by nuclei binary image to get individual nuclei without contours that are shared between touching nuclei
	imageCalculator("Multiply create 32-bit", "ObjectComponent","VoronoiZones");
	// reove small objects (<10 pixels)
	run("Gray Scale Attribute Filtering", "operation=Opening attribute=Area minimum=10 connectivity=4");
	rename("IndividualNucleiWithoutContours");

	// dilate individual nuclei to get back shared contours
	run("Duplicate...", " ");
	run("Morphological Filters", "operation=Dilation element=Square radius=2");
	rename("DilatedIndividualNuclei");

	// binarize individual nuclei with missing contours
	selectImage("IndividualNucleiWithoutContours");
	run("Duplicate...", " ");
	setThreshold(0.5, 200000);
	run("Convert to Mask");
	run("Divide...", "value=255");
	rename("BinaryNucleiWithoutContours");

	// subtract object component from binary nuclei without contours to obtain the contours
	imageCalculator("Subtract create 32-bit", "ObjectComponent","BinaryNucleiWithoutContours");
	rename("ContoursToAdd");
	// multiply the contours by the dilated nuclei to get the correct id
	imageCalculator("Multiply create 32-bit", "ContoursToAdd","DilatedIndividualNuclei");
	// add contours to nuclei without contours
	imageCalculator("Add create 32-bit", "Result of ContoursToAdd","IndividualNucleiWithoutContours");
	run("Grays");

	///////////// save image to visually inspect the results /////////////////
	// save the image and the rois for visual inspection
	saveAs("tiff", output + File.separator + file);

	///////////// clear everything /////////////////
	// close all images
	run("Close All");
}
