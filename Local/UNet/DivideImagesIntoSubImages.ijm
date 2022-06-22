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
	
}

function processFile(input, output, file) {

	// parameters
	// number of sub-images across width
	width_divide = 2;
	// number of sub-images across height
	height_divide = 2;
	
	// open image
	open(input + File.separator + file);
	// rename image
	rename("input");
	// get location
	getLocationAndSize(locX, locY, sizeW, sizeH); 
	// get width and heigh of image
	width = getWidth(); 
	height = getHeight(); 
	// compute tiles' width and height
	tileWidth = width / width_divide; 
	tileHeight = height / height_divide; 
	// loop over y coordinates
	for (y = 0; y < height_divide; y++) { 
		// compute Y offset
		offsetY = y * height / height_divide; 
		// loop over x coordinates
	 	for (x = 0; x < width_divide; x++) { 
	 		// compute X offset
			offsetX = x * width / width_divide; 
			// select input image
			selectImage("input");
			// define cuurent locations
			call("ij.gui.ImageWindow.setNextLocation", locX + offsetX, locY + offsetY); 
			// duplicate image
			run("Duplicate...", "duplicate");
			// define rectangle of sub-image and crop it
			makeRectangle(offsetX, offsetY, tileWidth, tileHeight); 
 			run("Crop"); 
 			// save sub-image
 			saveAs("tiff", output + File.separator + file + "_" + x + "_" + y + ".tiff");
		} 
	} 

	///////////// clear everything /////////////////
	// close all images
	run("Close All");
}
