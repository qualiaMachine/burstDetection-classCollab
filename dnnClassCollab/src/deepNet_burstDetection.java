/**
 * @Author: Yuting Liu and Jude Shavlik.
 *
 * Copyright 2017.  Free for educational and basic-research use.
 *
 * The main class for deepNet_burstDetection of cs638/838.
 *
 * Reads in the image files and stores BufferedImage's for every example.  Converts to fixed-length
 * feature vectors (of doubles).  Can use RGB (plus grey-scale) or use grey scale.
 *
 * You might want to debug and experiment with your Deep ANN code using a separate class, but when you turn in deepNet_burstDetection.java, insert that class here to simplify grading.
 *
 * Some snippets from Jude's code left in here - feel free to use or discard.
 *
 *
 *
 *
 * Code adapted, for Lab3, by Chris Endemann, Dave Sescleifer, and Newton Wolfe, 3/15/17
 */


import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.*;
import javax.imageio.ImageIO;

public class deepNet_burstDetection {

	public static int     imageWidth = 351, imageHeight = 19; // Images are imageSize x imageSize.  The provided data is 128x128, but this can be resized by setting this value (or passing in an argument).
	// You might want to resize to 8x8, 16x16, 32x32, or 64x64; this can reduce your network size and speed up debugging runs.
	// ALL IMAGES IN A TRAINING RUN SHOULD BE THE *SAME* SIZE.
	public static int     imageContextWidth = 1750;
	public static int 		wideKernelSize = 10;
	private static enum    Category { positive, negative };  // We'll hardwire these in, but more robust code would not do so.

	private static final Boolean    useRGB = false; // If true, FOUR units are used per pixel: red, green, blue, and grey.  If false, only ONE (the grey-scale value).
	public static       int unitsPerPixel = (useRGB ? 4 : 1); // If using RGB, use red+blue+green+grey.  Otherwise just use the grey value.

	private static final String    modelToUse = "deep"; // Should be one of { "perceptrons", "oneLayer", "deep" };  You might want to use this if you are trying approaches other than a Deep ANN.
	public static int       inputVectorSize;         // The provided code uses a 1D vector of input features.  You might want to create a 2D version for your Deep ANN code.
	// Or use the get2DfeatureValue() 'accessor function' that maps 2D coordinates into the 1D vector.
	// The last element in this vector holds the 'teacher-provided' label of the example.
	public static final boolean dropOut = false, normalizeKernelOutputByKernelSum = false; // turns dropout on/off for ALL layers (except output, of course); normalizeKernelOutputByKernelSum was to see effect of normlizign kernel's summed output by sum of kernel weights, recommended by several sources
	public static double eta       =    0.01, fractionOfTrainingToUse = .1, hiddenDropoutRate = 0.0, inputDropoutRate = 0.0; // To turn off drop out, set dropoutRate to 0.0 (or a neg number).

	private static int    maxEpochs = 1; // Feel free to set to a different value.
	public static final boolean printEpochErrorPercentages = true;
	protected static  final double  shiftProbNumerator                = 6.0; // 6.0 is the 'default.'
	protected static  final double  probOfKeepingShiftedTrainsetImage = (shiftProbNumerator / 48.0); // This 48 is also embedded elsewhere!
	protected static  final boolean perturbPerturbedImages            = false;
	private   static  final boolean createExtraTrainingExamples       = false;
	private   static  final boolean confusionMatricies                = true;
	public static final double windowIndicator = 10000;
	private static final int trialSize = imageContextWidth; // may want to extract programmically in the future
	public static boolean useWideKernel = false;
	public static boolean useContext = true;
	public static Map<String, Integer> typesOfImages;
	public static int nTypes;
	public static ArrayList<String> typesList;
	public static String setType = "windowVecAndStimTimesAdded";

	public static boolean runTrained = true;
	static double[][][] features;



	public static void main(String[] args) {		
		String exType;
		if(useContext == true){
			exType = "contextExs";
		}else{
			exType = "noContextExs";
		}
		// Check dropOut params
		if(hiddenDropoutRate < 0.0 || inputDropoutRate < 0.0) {
			System.err.println("Dropout rate can't be set below 0.  Set to 0 to turn off, or > 0 to turn on.");
			System.exit(1);
		}else if(dropOut == false && (hiddenDropoutRate != 0 || inputDropoutRate != 0)){
			System.err.println("Dropout boolean does not agree with dropout rates.  Set rates to zero for dropOut flag == false.");
			System.exit(1);
		}

		if(args.length > 5) {
			System.err.println("Usage error: java deepNet_burstDetection setType exType wideKernel maxEpochs eta");
			System.exit(1);
		}

		if (args.length >= 1) {  setType = args[0]; }
		if (args.length >= 2) {
			exType = args[1];
			if (args[1].equals ("contextExs")) {
				useContext = true;
			}else{
				useContext = false;
			}
		}
		if (args.length >= 3) {
			if (args[2].equals ("true")) {
				useWideKernel = true;
			}else{
				useWideKernel = false;
			}
		}
		if (args.length >= 4) {  maxEpochs     = Integer.parseInt(args[3]); }
		if (args.length >= 5) {  eta     = Double.parseDouble(args[4]); }
		if (args.length >= 6) {  
			if (args[5].equals ("true")) {
				runTrained = true;
			}else{
				runTrained = false;
			}
		}

		if(dropOut){
			System.out.println("Note: Dropout is on for all layers except output.");
			System.out.println("inputDropoutRate = " + inputDropoutRate);
			System.out.println("hiddenDropoutRate = " + hiddenDropoutRate);
		}
		if(normalizeKernelOutputByKernelSum){
			System.out.println("Note: flag - NormalizeKernelOutputByKernelSum is set to true.");
		}
		// Print other parameter settings so we have a hardcopy saved to dribble file
		System.out.println("maxEpochs = " + maxEpochs);
		System.out.println("fractionOfTrainingToUse = " + fractionOfTrainingToUse);
		System.out.println("eta = " + eta);
		System.out.println("setType = " + setType);
		System.out.println("exType = " + exType);
		System.out.println("useWideKernel = " + useWideKernel);
		System.out.println();

		//	String trainDirectory = "../allProjectTrainTestTuneSets-zipped/" + setType + "/train/" + exType + "/";
		//	String  tuneDirectory = "../allProjectTrainTestTuneSets-zipped/" + setType + "/tune/" + exType + "/";
		//	String  testDirectory = "../allProjectTrainTestTuneSets-zipped/" + setType + "/test/" + exType + "/";
		String trainDirectory = "/Users/tammi/Desktop/burstDetection/java/deepNet_burstDetection/dataForProgram/allProjectTrainTestTuneSets-zipped/" + setType + "/train/" + exType + "/";
		String  tuneDirectory = "/Users/tammi/Desktop/burstDetection/java/deepNet_burstDetection/dataForProgram/allProjectTrainTestTuneSets-zipped/" + setType + "/tune/" + exType + "/";
		String  testDirectory = "/Users/tammi/Desktop/burstDetection/java/deepNet_burstDetection/dataForProgram/allProjectTrainTestTuneSets-zipped/" + setType + "/test/" + exType + "/";


		// Here are statements with the absolute path to open images folder
		File trainsetDir = new File(trainDirectory);
		File tunesetDir  = new File( tuneDirectory);
		File testsetDir  = new File( testDirectory);

		// create three datasets
		// Dataset trainset = new Dataset();
		// Dataset  tuneset = new Dataset();
		// Dataset  testset = new Dataset();
		Vector<Example> trainset = new Vector<Example>();
		Vector<Example>  tuneset = new Vector<Example>();
		Vector<Example>  testset = new Vector<Example>();


		// Initializa Mapping
		typesOfImages = new HashMap<String, Integer> ();
		nTypes = 0;
		typesList = new ArrayList<String> ();

		// Load in images into datasets.
		long start = System.currentTimeMillis();
		double fractionOfFilesToUse = 0.001/3;
		loadDataset(trainset, trainsetDir, trainDirectory,fractionOfFilesToUse);
		System.out.println("The trainset contains " + comma(trainset.size()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		fractionOfFilesToUse = 0.001;
		loadDataset(tuneset, tunesetDir, tuneDirectory,fractionOfFilesToUse);
		System.out.println("The tuneset contains " + comma( tuneset.size()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		start = System.currentTimeMillis();
		fractionOfFilesToUse = 0.001;
		loadDataset(testset, testsetDir, testDirectory,fractionOfFilesToUse);
		System.out.println("The testset contains " + comma( testset.size()) + " examples.  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + ".");

		if (deepNet_burstDetection.useContext) {
			deepNet_burstDetection.features = new double[imageHeight][imageContextWidth][unitsPerPixel];
		} else {
			deepNet_burstDetection.features = new double[imageHeight][imageWidth][unitsPerPixel];
		}

		createDribbleFile("results/"
				+ modelToUse
				+ "_extraExamples"
				+ Boolean.toString(createExtraTrainingExamples)
				+ "_inputDropoutRate"
				+ truncate(  10 * inputDropoutRate,    0)
				+ "_hiddenDropoutRate"
				+ truncate(  10 * hiddenDropoutRate,    0)  // Feel free to decide what you wish to include, but aim to NOT print decimal points, since this is a file name
				+ "_eta"          + truncate(1000 * eta, 0)
				+ "_trainPercent" + truncate(100 * fractionOfTrainingToUse, 0)
				+ "_numberHUs"    + numberOfHiddenUnits
				+ ".txt");

		// Now train a Deep ANN.  You might wish to first use your Lab 2 code here and see how one layer of HUs does.  Maybe even try your perceptron code.
		// We are providing code that converts images to feature vectors.  Feel free to discard or modify.
		start = System.currentTimeMillis();
		trainDeep(trainset, tuneset, testset);
		println("\nTook " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to train.");
		if(runTrained){
			String networkPath = "/Users/tammi/Desktop/burstDetection/java/";
			String unlabelledDataDirectory = "/Users/tammi/Desktop/burstDetection/java/deepNet_burstDetection/dataForProgram/unlabelledTests";
			fractionOfFilesToUse = 1;

			// Loop through animal/index/stim#
			File unlabelledExsDir = new File(unlabelledDataDirectory + "/animalData-16trial-trialWithFeats");
			File[] listOfAnimalFolders = unlabelledExsDir.listFiles();

			for (int i = 0; i < listOfAnimalFolders.length; i++) {
				if (listOfAnimalFolders[i].isFile()) {
//					System.out.println("File " + listOfAnimalFolders[i].getName());
				}else if (listOfAnimalFolders[i].isDirectory()) {
//					System.out.println("Directory " + listOfAnimalFolders[i].getName());
					File listOfDates = new File(unlabelledDataDirectory + 
							"/animalData-16trial-trialWithFeats/" + listOfAnimalFolders[i].getName());
					File[] listOfDateFolders = listOfDates.listFiles();

					for (int j = 0; j < listOfDateFolders.length; j++) {
						if (listOfDateFolders[j].isFile()) {
//							System.out.println("File " + listOfDateFolders[j].getName());
						} else if (listOfDateFolders[j].isDirectory()) {
//							System.out.println("Directory " + listOfDateFolders[j].getName());
							File listOfIndices = new File(unlabelledDataDirectory + 
									"/animalData-16trial-trialWithFeats/" + listOfAnimalFolders[i].getName() +
									"/" + listOfDateFolders[j].getName());
							File[] listOfIndiceFolders = listOfIndices.listFiles();
							for (int k = 0; k < listOfIndiceFolders.length; k++) {
								if (listOfIndiceFolders[k].isFile()) {
//									System.out.println("File " + listOfIndiceFolders[k].getName());
								} else if (listOfIndiceFolders[k].isDirectory()) {
//									System.out.println("Directory " + listOfIndiceFolders[k].getName());
//									System.out.println(unlabelledDataDirectory + 
//											"/animalData-16trial-trialWithFeats/" + listOfAnimalFolders[i].getName() +
//											"/" + listOfDateFolders[j].getName() + "/" +
//											listOfIndiceFolders[k].getName());
									File listOfStims = new File(unlabelledDataDirectory + 
											"/animalData-16trial-trialWithFeats/" + listOfAnimalFolders[i].getName() +
											"/" + listOfDateFolders[j].getName() + "/" +
											listOfIndiceFolders[k].getName());
									File[] listOfStimFolders = listOfStims.listFiles();
									for (int l = 0; l < listOfStimFolders.length; l++) {
										if (listOfStimFolders[l].isFile()) {
//											System.out.println("File " + listOfStimFolders[l].getName());
										} else if (listOfStimFolders[l].isDirectory()) {
//											System.out.println("Directory " + listOfStimFolders[l].getName());
											File imageDataFiles = new File(unlabelledDataDirectory + 
													"/animalData-16trial-trialWithFeats/" + listOfAnimalFolders[i].getName() +
													"/" + listOfDateFolders[j].getName() + "/" +
													listOfIndiceFolders[k].getName() + "/" + listOfStimFolders[l].getName());
											File[] listOfImageDataFiles = imageDataFiles.listFiles();
											
											String path = unlabelledDataDirectory + 
													"/animalData-16trial-trialWithFeats/" + listOfAnimalFolders[i].getName() +
													"/" + listOfDateFolders[j].getName() + "/" +
													listOfIndiceFolders[k].getName() + "/" + listOfStimFolders[l].getName();
											for (int m = 0; m < listOfImageDataFiles.length; m++) {
												Vector<UnlabelledEx>  unlabelledSet = new Vector<UnlabelledEx>();
												if (listOfImageDataFiles[m].isFile()) {
													File file = listOfImageDataFiles[m];
													String name = file.getName();
//													System.out.println("File " + name);
//													System.out.println("path " + path);
													for(int windowIndex = 0; windowIndex < deepNet_burstDetection.trialSize; windowIndex++){
														UnlabelledEx unlabelEx = new UnlabelledEx(path, name, deepNet_burstDetection.windowIndicator,windowIndex);
														unlabelEx.getFeatures();
														unlabelledSet.add(unlabelEx);
													}
													double [] predictions = new double[deepNet_burstDetection.trialSize];
													runNetworkOnSet(unlabelledSet,predictions,networkPath);
												}
											}
										}
									}
								}
							}
						}
					}

				}
			}
		}
		closeDribbleFile();

	}
	
/* Run network on 351 examples (one for each timepoint of trial; output results to vector; save vector out in the end
 * 
 * 
 */
	private static void runNetworkOnSet(Vector<UnlabelledEx> unlabelledSet, double[] predictions, String networkPath) {
		DeepNet network = loadNetwork(networkPath);
		
		// Loop through unlabelled ex's and store predictions
		int predInt = 0;
		for (UnlabelledEx example : unlabelledSet) {
//			String name = getType (example.name);
//			int k = typesOfImages.get (name);
			int label = network.getLabel(example.getFeatures());
			if(label == 0){
				System.out.println("Hey look, a brust");
			}
//			System.out.println(label);
			predictions[predInt] = label;
			predInt++;
		}
		// Save predictions to same folder as file was retrieved from
		String savePath = unlabelledSet.get(0).path;
		String fileName = unlabelledSet.get(0).name;
		System.out.println(fileName);
		String stringToCutOut = "trialWithFeats";
		String predsFileName = "predictions" + fileName.substring(stringToCutOut.length(),(fileName.length()-3)) + "txt";
		System.out.println(predsFileName);
		try
		{
			String outfile = savePath + "/" + predsFileName;
		    PrintWriter pr = new PrintWriter(outfile);    

		    for (int i=0; i<predictions.length ; i++)
		    {
		        pr.println(predictions[i]);
		    }
		    pr.close();
		}
		catch (Exception e)
		{
		    e.printStackTrace();
		    System.out.println("No such file exists.");
		}
		
	}

	public static String getType (String filename) {
		//	filename = "posEx-normBurstEdgeCase1 - Resh-16408-006-Stim5-trials109t109.txt";
		//	filename = "negEx-noisey10-burst1 - Nun-15d22-004-Stim4-trials56t56.txt";
		//	filename = "negEx-reduced10-burst1 - Nun-15d22-004-Stim4-trials56t56";
		String ret = filename.substring (filename.indexOf ("-") + 1); // fn_noLabel
		ret = ret.substring (0, ret.indexOf ("-")).trim (); //fn_noLabel_noTrialInfo

		if(ret.contains("normBurstEdgeCase")){
			ret = "normBurstEdgeCase";
		} else if (ret.contains ("normBurst")) {
			ret = "normBurst";
		} else if (ret.contains ("shuffledBurst")) {
			ret = "shuffled";
		} else if (ret.contains ("Shift")) {
			String shiftSize = ret.substring (ret.indexOf ("shiftSize") + 9); 
			ret = "shift" + shiftSize;
		}else if(ret.contains("noisey")) {
			String noiseLevel = ret.substring (ret.indexOf ("noisey") + 6);
			ret = "noisey" + noiseLevel + "Perc";	
		}else if(ret.contains("reduced")){
			String redLevel = ret.substring (ret.indexOf ("reduced") + 7);
			ret = "reducedBy" + redLevel + "Perc";
		} else {
			int id1 = filename.indexOf ("-");
			int id2 = filename.substring (id1 + 1).indexOf ("-") + id1 + 1;
			int id = filename.substring (id2 + 1).indexOf ("bursts") + id2 + 1;
			int id3 = filename.substring (id2 + 1).indexOf ("-") + id2 + 1;
			int id4 = filename.substring (id3 + 1).indexOf ("-") + id3 + 1;
			int id5 = filename.substring (id4 + 1).indexOf ("-") + id4 + 1;
			ret = filename.substring (id2 + 1, id).trim () + filename.substring (id4 + 1, id5).trim ();
		}
		return ret;
	}

	public static void loadDataset(Vector<Example> dataset, File dir, String path, double fractionOfFilesToUse) {
		int seed = 10;
		Random generator = new Random(seed);
		int neg = 0,pos = 0;
		String name = null;
		for(File file : dir.listFiles()) {
			double random = generator.nextDouble();
			//			System.out.println(random);
			if(random < fractionOfFilesToUse){
				try {
					FileInputStream fis = new FileInputStream(file);
					name = file.getName ();
					String type = getType (name);
					int label = 0;
					if (name.contains ("negEx") || name.contains ("NegEx")) {
						neg ++;
						if (!typesOfImages.containsKey (type)) {
							typesOfImages.put (type, nTypes);
							nTypes ++;
							typesList.add (type);
						}
						label = 1;
					} else if (name.contains ("posEx") || name.contains ("PosEx")) {
						pos ++;
						if (!typesOfImages.containsKey (type)) {
							typesOfImages.put (type, nTypes);
							nTypes ++;
							typesList.add (type);
						}
						label = 0;
					} else{
						int x = 1;
						x = x + 3;
					}

					dataset.add (new Example (path + name, label, name));
					fis.close ();
				} catch (Exception e) { }
			}
		}
		System.out.println ("Contain " + pos + " positive inputs and " + neg + " negative inputs");
	}
	
	/**
	 * Converts a string of the category name to a Category enum
	 * @param name The name of the category
	 * @return An enum containing the appropriate type.
	 */
	private static Category convertCategoryStringToEnum(String name) {
		if ("positive".equals(name))   return Category.positive; // Should have been the singular 'airplane' but we'll live with this minor error.
		if ("negative".equals(name))   return Category.negative;
		throw new Error("Unknown category: " + name);
	}

	/**
	 * Creates weight initialization weights based on the StackOverflow formula.
	 * @param fanin The number of incoming weights.
	 * @param fanout The number of nodes this node connects to (i.e. outgoing weights)
	 * @return a weight for initialization.
	 */
	public static double getRandomWeight(int fanin, int fanout) { // This is one 'rule of thumb' for initializing weights.  Fine for perceptrons and one-layer ANN at least.
		double range = Math.max(Double.MIN_VALUE, 4.0 * Math.sqrt(6.0 / (fanin + fanout)));
		return (2.0 * random() - 1.0) * range;
	}

	// Map from 2D coordinates (in pixels) to the 1D fixed-length feature vector.
	private static double get2DfeatureValue(Vector<Double> ex, int x, int y, int offset) { // If only using GREY, then offset = 0;  Else offset = 0 for RED, 1 for GREEN, 2 for BLUE, and 3 for GREY.
		return ex.get(unitsPerPixel * (y * imageWidth + x) + offset); // Jude: I have not used this, so might need debugging.
	}


	private static void fillImageArray(double[][][] imageArray, Instance image) {
		for (int x = 0; x < imageHeight; x++) {
			for (int y = 0; y < imageWidth; y++) {
				if (useRGB) {
					imageArray[x][y][0] = image.getRedChannel()[x][y] / 255.0;
					imageArray[x][y][1] = image.getGreenChannel()[x][y] / 255.0;
					imageArray[x][y][2] = image.getBlueChannel()[x][y] / 255.0;
					imageArray[x][y][3] = image.getGrayImage()[x][y] / 255.0;
				} else {
					imageArray[x][y][0] = image.getGrayImage()[x][y] / 255.0;
				}
			}
		}
	}

	/**
	 * Creates feature vectors from the dataset's images for one layer and perceptron processing.
	 * @param featureVectors The declared feature vectors to be filled.
	 * @param dataset The dataset from which to fill the feature vectors.
	 */
	private static void fillFeatureVectors(Vector<Vector<Double>> featureVectors, Dataset dataset) {
		for (Instance image : dataset.getImages()) {
			featureVectors.addElement(convertToFeatureVector(image));
		}
	}

	/**
	 * Converts an image to a Vector of doubles.
	 * @param image the image to convert
	 * @return a vector of doubles encoding the image
	 */
	private static Vector<Double> convertToFeatureVector(Instance image) {
		Vector<Double> result = new Vector<Double>(inputVectorSize);

		for (int index = 0; index < inputVectorSize - 1; index++) { // Need to subtract 1 since the last item is the CATEGORY.
			if (useRGB) {
				int xValue = (index / unitsPerPixel) % image.getWidth();
				int yValue = (index / unitsPerPixel) / image.getWidth();
				//	System.out.println("  xValue = " + xValue + " and yValue = " + yValue + " for index = " + index);
				if      (index % unitsPerPixel == 0) result.add(image.getRedChannel()  [xValue][yValue] / 255.0); // If unitsPerPixel > 4, this if-then-elseif needs to be edited!
				else if (index % unitsPerPixel == 1) result.add(image.getGreenChannel()[xValue][yValue] / 255.0);
				else if (index % unitsPerPixel == 2) result.add(image.getBlueChannel() [xValue][yValue] / 255.0);
				else                     result.add(image.getGrayImage()   [xValue][yValue] / 255.0); // Seems reasonable to also provide the GREY value.
			} else {
				int xValue = index % image.getWidth();
				int yValue = index / image.getWidth();
				result.add(                         image.getGrayImage()   [xValue][yValue] / 255.0);
			}
		}
		result.add((double) convertCategoryStringToEnum(image.getLabel()).ordinal()); // The last item is the CATEGORY, representing as an integer starting at 0 (and that int is then coerced to double).

		return result;
	}

	/**
	 * Augment the dataset by creating more images from the training set.
	 * @param trainImage The image from which we're going to create our new data.
	 * @param probOfKeeping Probability that we'll keep each generated image and add it to our dataset.
	 */
	private static void createMoreImagesFromThisImage(Instance trainImage, double probOfKeeping, Dataset trainsetExtras) {
		if (!"airplanes".equals(  trainImage.getLabel()) &&  // Airplanes all 'face' right and up, so don't flip left-to-right or top-to-bottom.
				!"grand_piano".equals(trainImage.getLabel())) {  // Ditto for pianos.

			if (trainImage.getProvenance() != Instance.HowCreated.FlippedLeftToRight && random() <= probOfKeeping) trainsetExtras.add(trainImage.flipImageLeftToRight());

			if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't flip to-to-bottom.
					!"flower".equals(   trainImage.getLabel()) &&  // Ditto for flowers.
					!"starfish".equals( trainImage.getLabel())) {  // Star fish are standardized to 'point up.
				if (trainImage.getProvenance() != Instance.HowCreated.FlippedTopToBottom && random() <= probOfKeeping) trainsetExtras.add(trainImage.flipImageTopToBottom());
			}
		}
		boolean rotateImages = true;
		if (rotateImages && trainImage.getProvenance() != Instance.HowCreated.Rotated) {
			//    Instance rotated = origTrainImage.rotateImageThisManyDegrees(3);
			//    origTrainImage.display2D(origTrainImage.getGrayImage());
			//    rotated.display2D(              rotated.getGrayImage()); waitForEnter();

			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  3));
			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -3));
			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  4));
			if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -4));
			if (!"butterfly".equals(trainImage.getLabel()) &&  // Butterflies all have the heads at the top, so don't rotate too much.
					!"flower".equals(   trainImage.getLabel()) &&  // Ditto for flowers and starfish.
					!"starfish".equals( trainImage.getLabel())) {
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  5));
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -5));
			} else {
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees(  2));
				if (random() <= probOfKeeping) trainsetExtras.add(trainImage.rotateImageThisManyDegrees( -2));
			}
		}
		// Would be good to also shift and rotate the flipped examples, but more complex code needed.
		if (trainImage.getProvenance() != Instance.HowCreated.Shifted) {
			for (    int shiftX = -3; shiftX <= 3; shiftX++) {
				for (int shiftY = -3; shiftY <= 3; shiftY++) {
					// Only keep some of these, so these don't overwhelm the flipped and rotated examples when down sampling below.
					if ((shiftX != 0 || shiftY != 0) && random() <= probOfKeepingShiftedTrainsetImage * probOfKeeping) trainsetExtras.add(trainImage.shiftImage(shiftX, shiftY));
				}
			}
		}
	}

	////////////////////  Some utility methods (cut-and-pasted from JWS' Utils.java file). ///////////////////////////////////////////////////

	private static final long millisecInMinute = 60000;
	private static final long millisecInHour   = 60 * millisecInMinute;
	private static final long millisecInDay    = 24 * millisecInHour;
	public static String convertMillisecondsToTimeSpan(long millisec) {
		return convertMillisecondsToTimeSpan(millisec, 0);
	}
	public static String convertMillisecondsToTimeSpan(long millisec, int digits) {
		if (millisec ==    0) { return "0 seconds"; } // Handle these cases this way rather than saying "0 milliseconds."
		if (millisec <  1000) { return comma(millisec) + " milliseconds"; } // Or just comment out these two lines?
		if (millisec > millisecInDay)    { return comma(millisec / millisecInDay)    + " days and "    + convertMillisecondsToTimeSpan(millisec % millisecInDay,    digits); }
		if (millisec > millisecInHour)   { return comma(millisec / millisecInHour)   + " hours and "   + convertMillisecondsToTimeSpan(millisec % millisecInHour,   digits); }
		if (millisec > millisecInMinute) { return comma(millisec / millisecInMinute) + " minutes and " + convertMillisecondsToTimeSpan(millisec % millisecInMinute, digits); }

		return truncate(millisec / 1000.0, digits) + " seconds";
	}

	public static String comma(int value) { // Always use separators (e.g., "100,000").
		return String.format("%,d", value);
	}
	public static String comma(long value) { // Always use separators (e.g., "100,000").
		return String.format("%,d", value);
	}
	public static String comma(double value) { // Always use separators (e.g., "100,000").
		return String.format("%,f", value);
	}
	public static String padLeft(String value, int width) {
		String spec = "%" + width + "s";
		return String.format(spec, value);
	}

	/**
	 * Format the given floating point number by truncating it to the specified
	 * number of decimal places.
	 *
	 * @param d
	 *            A number.
	 * @param decimals
	 *            How many decimal places the number should have when displayed.
	 * @return A string containing the given number formatted to the specified
	 *         number of decimal places.
	 */
	public static String truncate(double d, int decimals) {
		double abs = Math.abs(d);
		if (abs > 1e13)             {
			return String.format("%."  + (decimals + 4) + "g", d);
		} else if (abs > 0 && abs < Math.pow(10, -decimals))  {
			return String.format("%."  +  decimals      + "g", d);
		}
		return     String.format("%,." +  decimals      + "f", d);
	}

	/** Randomly permute vector in place.
	 *
	 * @param <T>  Type of vector to permute.
	 * @param vector Vector to permute in place.
	 */
	public static <T> void permute(Vector<T> vector) {
		if (vector != null) { // NOTE from JWS (2/2/12): not sure this is an unbiased permute; I prefer (1) assigning random number to each element, (2) sorting, (3) removing random numbers.
			// But also see "http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle" which justifies this.
			/*	To shuffle an array a of n elements (indices 0..n-1):
								for i from n - 1 downto 1 do
								j <- random integer with 0 <= j <= i
								exchange a[j] and a[i]
			 */

			for (int i = vector.size() - 1; i >= 1; i--) {  // Note from JWS (2/2/12): to match the above I reversed the FOR loop that Trevor wrote, though I don't think it matters.
				int j = random0toNminus1(i + 1);
				if (j != i) {
					T swap =    vector.get(i);
					vector.set(i, vector.get(j));
					vector.set(j, swap);
				}
			}
		}
	}

	public static Random randomInstance = new Random(638 * 838);  // Change the 638 * 838 to get a different sequence of random numbers.

	/**
	 * @return The next random double.
	 */
	public static double random() {
		return randomInstance.nextDouble();
	}

	/**
	 * @param lower
	 *            The lower end of the interval.
	 * @param upper
	 *            The upper end of the interval. It is not possible for the
	 *            returned random number to equal this number.
	 * @return Returns a random integer in the given interval [lower, upper).
	 */
	public static int randomInInterval(int lower, int upper) {
		return lower + (int) Math.floor(random() * (upper - lower));
	}


	/**
	 * @param upper
	 *            The upper bound on the interval.
	 * @return A random number in the interval [0, upper).
	 */
	public static int random0toNminus1(int upper) {
		return randomInInterval(0, upper);
	}

	///////////////////////////////////////////////////////////////////////////////////////////////  Write your own code below here.  Feel free to use or discard what is provided.

	private static int trainPerceptrons(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		Vector<Vector<Double>> perceptrons = new Vector<Vector<Double>>(Category.values().length);  // One perceptron per category.

		for (int i = 0; i < Category.values().length; i++) {
			Vector<Double> perceptron = new Vector<Double>(inputVectorSize);  // Note: inputVectorSize includes the OUTPUT CATEGORY as the LAST element.  That element in the perceptron will be the BIAS.
			perceptrons.add(perceptron);
			for (int indexWgt = 0; indexWgt < inputVectorSize; indexWgt++) perceptron.add(getRandomWeight(inputVectorSize, 1)); // Initialize weights.
		}

		if (fractionOfTrainingToUse < 1.0) {  // Randomize list, then get the first N of them.
			int numberToKeep = (int) (fractionOfTrainingToUse * trainFeatureVectors.size());
			Vector<Vector<Double>> trainFeatureVectors_temp = new Vector<Vector<Double>>(numberToKeep);

			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.
			for (int i = 0; i <numberToKeep; i++) {
				trainFeatureVectors_temp.add(trainFeatureVectors.get(i));
			}
			trainFeatureVectors = trainFeatureVectors_temp;
		}

		int trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
		long  overallStart = System.currentTimeMillis(), start = overallStart;


		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

			// CODE NEEDED HERE!

			println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
			reportPerceptronConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
			start = System.currentTimeMillis();
		}
		println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch)
		+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
		return testSetErrorsAtBestTune;
	}

	private static void reportPerceptronConfig() {
		println(  "***** PERCEPTRON: UseRGB = " + useRGB + ", imageSize = " + imageWidth + "x" + imageHeight + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2) + ", eta = " + truncate(eta, 2) + ", input_dropout rate = " + truncate(hiddenDropoutRate, 2) + ", hidden_dropout rate = " + truncate(hiddenDropoutRate, 2)	);
	}

	////////////////////////////////////////////////////////////////////////////////////////////////   ONE HIDDEN LAYER

	private static boolean debugOneLayer               = false;  // If set true, more things checked and/or printed (which does slow down the code).
	public static int    numberOfHiddenUnits          = 50;

	private static int trainOneHU(Vector<Vector<Double>> trainFeatureVectors, Vector<Vector<Double>> tuneFeatureVectors, Vector<Vector<Double>> testFeatureVectors) {
		long overallStart   = System.currentTimeMillis(), start = overallStart;
		int  trainSetErrors = Integer.MAX_VALUE, tuneSetErrors = Integer.MAX_VALUE, best_tuneSetErrors = Integer.MAX_VALUE, testSetErrors = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;

		OneLayer network = new OneLayer();

		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(trainFeatureVectors); // Note: this is an IN-PLACE permute, but that is OK.

			for (Vector<Double> example : trainFeatureVectors) {
				network.train(example);
			}

			tuneSetErrors = network.errors(tuneFeatureVectors);
			testSetErrors = network.errors(testFeatureVectors);
			println("Current tune set error: " + comma((double) tuneSetErrors / tuneFeatureVectors.size())
			+ ", test set error: " + comma((double) testSetErrors / testFeatureVectors.size()));

			if (tuneSetErrors < best_tuneSetErrors) {
				best_epoch = epoch;
				best_tuneSetErrors = tuneSetErrors;
				testSetErrorsAtBestTune = testSetErrors;
			}

			println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
			//reportOneLayerConfig(); // Print out some info after epoch, so you can see what experiment is running in a given console.
			start = System.currentTimeMillis();
		}

		println("\n***** Best tuneset errors = " + comma(best_tuneSetErrors) + " of " + comma(tuneFeatureVectors.size()) + " (" + truncate((100.0 *      best_tuneSetErrors) / tuneFeatureVectors.size(), 2) + "%) at epoch = " + comma(best_epoch)
		+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(testFeatureVectors.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / testFeatureVectors.size(), 2) + "%).\n");
		return testSetErrorsAtBestTune;
	}

	private static void reportOneLayerConfig() {
		println(  "***** ONE-LAYER: UseRGB = " + useRGB + ", imageSize = " + imageWidth + "x" + imageHeight + ", fraction of training examples used = " + truncate(fractionOfTrainingToUse, 2)
		+ ", eta = " + truncate(eta, 2)   + ", input dropout rate = "      + truncate(inputDropoutRate, 2) + ", hidden dropout rate = "      + truncate(hiddenDropoutRate, 2) + ", number HUs = " + numberOfHiddenUnits
		//	+ ", activationFunctionForHUs = " + activationFunctionForHUs + ", activationFunctionForOutputs = " + activationFunctionForOutputs
		//	+ ", # forward props = " + comma(forwardPropCounter)
				);
		//	for (Category cat : Category.values()) {  // Report the output unit biases.
		//		int catIndex = cat.ordinal();
		//
		//		System.out.print("  bias(" + cat + ") = " + truncate(weightsToOutputUnits[numberOfHiddenUnits][catIndex], 6));
		//	}   System.out.println();
	}

	// private static long forwardPropCounter = 0;  // Count the number of forward propagations performed.


	////////////////////////////////////////////////////////////////////////////////////////////////  DEEP ANN Code


	/**
	 * Trains a deep neural network based on the images provided.
	 * @param train a vector of Examples containing the training set.
	 * @param tune a vector of Examples containing the tuning set.
	 * @param test a vector of Examples containing the test set.
	 * @return A count of test set errors at the early stopping epoch.
	 */
	private static DeepNet loadNetwork(String path) {
		// load network in
		String filename;
		if(deepNet_burstDetection.useContext){	
			filename = "/Users/tammi/Desktop/burstDetection/java/fullNetwork_context.ser";
		}else{
			filename = "/Users/tammi/Desktop/burstDetection/java/fullNetwork_noContext.ser";
		}
		

		try {
			FileInputStream fis = new FileInputStream(filename);
			ObjectInputStream ois = new ObjectInputStream(fis);
			DeepNet network = (DeepNet) ois.readObject();
			ois.close();
			return network;
		} catch (IOException | ClassNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		return null;

	}


	private static int trainDeep(Vector<Example> train, Vector<Example> tune, Vector<Example> test) {
		DeepNet network = new DeepNet(dropOut, normalizeKernelOutputByKernelSum);
		int  trainSetError = Integer.MAX_VALUE, best_trainSetError = Integer.MAX_VALUE, tuneSetError = Integer.MAX_VALUE, best_tuneSetError = Integer.MAX_VALUE, testSetError = Integer.MAX_VALUE, best_epoch = -1, testSetErrorsAtBestTune = Integer.MAX_VALUE;
		long overallStart   = System.currentTimeMillis(), start = overallStart;

		int trainConfusion[][][] = new int[nTypes][2][2];
		int tuneConfusion[][][] = new int[nTypes][2][2]; // Two categories, so 2x2 confusion matricies.
		int testConfusion[][][] = new int[nTypes][2][2];
		int bestTestConfusion[][][] = new int[nTypes][2][2];
		double allTrainErrors[] = new double[maxEpochs];
		double allTuneErrors[] = new double[maxEpochs];
		double allTestErrors[] = new double[maxEpochs];

		double mapTestErrors[] = new double[nTypes];

		double[][][] features = deepNet_burstDetection.features;
		for (int epoch = 1; epoch <= maxEpochs /* && trainSetErrors > 0 */; epoch++) { // Might still want to train after trainset error = 0 since we want to get all predictions on the 'right side of zero' (whereas errors defined wrt HIGHEST output).
			permute(train); // Note: this is an IN-PLACE permute, but that is OK.
			for (Example example : train) {
				network.train(example.getFeatures(features), example.label);
			}
			trainSetError = deepErrors(train, network, trainConfusion, features);
			tuneSetError = deepErrors(tune, network, tuneConfusion, features);
			testSetError = deepErrors(test, network, testConfusion, features);
			println("Current tune set error: " + comma((double) tuneSetError / tune.size())
			+ ", test set error: " + comma((double) testSetError / test.size()));

			allTrainErrors[epoch-1] = (double)trainSetError / train.size();
			allTuneErrors[epoch-1] = (double)tuneSetError / tune.size();
			allTestErrors[epoch-1] = (double)testSetError / test.size();
			println ("[" + allTrainErrors[epoch-1] + "," + allTuneErrors[epoch-1] + "," + allTestErrors[epoch-1] + "]");

			if (confusionMatricies) {
				if(epoch % 10 == 0){
					println("Tune Confusion Matrix");
					printConfusion(tuneConfusion);
				}
			}
			if (tuneSetError < best_tuneSetError) {
				if (deepNet_burstDetection.useContext) {
					String filename = "/Users/tammi/Desktop/burstDetection/java/fullNetwork_context.ser";
					saveNetwork(network,filename);
				}else{
					String filename = "/Users/tammi/Desktop/burstDetection/java/fullNetwork_noContext.ser";
					saveNetwork(network,filename);
				}

				println("New best tuning set error!");
				println("Testset confusion matrix");
				printConfusion(testConfusion);
				best_epoch = epoch;
				best_tuneSetError = tuneSetError;
				testSetErrorsAtBestTune = testSetError;

				deepErrors2 (test, network, mapTestErrors, features);
				System.out.println("Best testset errors by categories");
				for (int i = 0;i < nTypes;i ++) {
					System.out.println ("'" + typesList.get (i) + "' errors = " + (mapTestErrors[i] * 100.0) + "%");
				}
				for (int i = 0;i < nTypes;i ++) {
					for (int j = 0;j < 2;j ++) {
						for (int k = 0;k < 2;k ++) {
							bestTestConfusion[i][j][k] = testConfusion[i][j][k];
						}
					}
				}
			}

			println("Done with Epoch # " + comma(epoch) + ".  Took " + convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " (" + convertMillisecondsToTimeSpan(System.currentTimeMillis() - overallStart) + " overall).");
			start = System.currentTimeMillis();
		}
		println("\n***** Best tuneset errors = " + comma(best_tuneSetError) + " of " + comma(tune.size()) + " (" + truncate((100.0 *      best_tuneSetError) / tune.size(), 2) + "%) at epoch = " + comma(best_epoch)
		+ " (testset errors = "    + comma(testSetErrorsAtBestTune) + " of " + comma(test.size()) + ", " + truncate((100.0 * testSetErrorsAtBestTune) / test.size(), 2) + "%).\n");

		System.out.println("Best testset errors by categories");
		for (int i = 0;i < nTypes;i ++) {
			System.out.println ("'" + typesList.get (i) + "' errors = " + (mapTestErrors[i] * 100.0) + "%");
		}
		printConfusion(bestTestConfusion);

		if(printEpochErrorPercentages){
			System.out.println("Printing epoch error percentages...");
			System.out.println("------");
			System.out.println("TRAIN");
			System.out.println(Arrays.toString(allTrainErrors));
			System.out.println("------");
			System.out.println("TUNE");
			System.out.println(Arrays.toString(allTuneErrors));
			System.out.println("------");
			System.out.println("TEST");
			System.out.println(Arrays.toString(allTestErrors));
		}

		return testSetErrorsAtBestTune;
	}

	private static void saveNetwork(DeepNet network, String filename) {

		try {
			FileOutputStream fout;
			fout = new FileOutputStream(filename);
			ObjectOutputStream oos;
			oos = new ObjectOutputStream(fout);
			oos.writeObject(network);
		} catch (IOException e1) {
			e1.printStackTrace();
		}		
	}


	/**
	 * Returns the number of errors made by a deep network across a dataset.
	 * @param dataset The dataset across which the network should predict classes.
	 * @param network DeepNet used to predict classes.
	 * @return The number of correctly classified examples.
	 */
	private static int deepErrors(Vector<Example> dataset, DeepNet network, int[][][] confusionMatrix, double[][][] features) {
		int errors = 0;
		for (int k = 0;k < nTypes;k ++) {
			for (int i = 0; i < confusionMatrix[k].length; i++) {
				Arrays.fill(confusionMatrix[k][i],0);
			}
		}
		for (Example example : dataset) {
			String name = getType (example.name);
			int k = typesOfImages.get (name);
			int label = network.getLabel(example.getFeatures (features));
			confusionMatrix[k][label][example.label]++;
			if (label != example.label) {
				errors++;
			}
		}
		return errors;
	}

	private static void deepErrors2(Vector<Example> dataset, DeepNet network, double[] mapLoss, double[][][] features) {
		int counts[] = new int[nTypes];
		int errors[] = new int[nTypes];
		for (Example example : dataset) {
			int label = network.getLabel(example.getFeatures (features));
			String name = getType (example.name);
			int type = typesOfImages.get (name);
			counts[type] ++;
			if (label != example.label) {
				errors[type] ++;
			}
		}
		for (int i = 0;i < nTypes;i ++) {
			mapLoss[i] = (1.0 * errors[i]) / counts[i];
		}
	}

	private static void printConfusion(int[][][] confusion) {
		for (int k = 0;k < nTypes;k ++) {
			println ("Type:" + typesList.get (k));
			for (int i = 0; i < confusion[k].length; i++) {
				for (int j = 0; j < confusion[k][i].length; j++) {
					print(String.format("%3d ", confusion[k][i][j]));
				}
				println();
			}
		}
	}

	/**
	 * Shavlik's "dribble file" code for Piazza @120
	 *
	 * Use createDribbleFile at the beginning of a train method, and closeDribbleFile at the end.
	 */

	private static PrintStream dribbleStream = null;
	public  static String    dribbleFileName = null;

	public static void createDribbleFile(String fileName) {
		if (dribbleStream != null) {
			dribbleStream.println("\n\n// Closed existing dribble file due to a createDribble call with file = " + fileName);
		}
		closeDribbleFile();
		try {
			ensureDirExists(fileName);
			FileOutputStream outStream = new FileOutputStream(fileName);
			dribbleStream = new PrintStream(outStream, false); // No auto-flush (can slow down code).
			dribbleFileName = fileName;
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			throw new Error("Unable to successfully open this file for writing:\n " + fileName + ".\nError message: " + e.getMessage());
		}
	}

	public static void closeDribbleFile() {
		dribbleFileName = null;
		if (dribbleStream == null) { return; }
		dribbleStream.close();
		dribbleStream = null;
	}

	private static int dribbleCharCount = 0;
	public static void print(String string) {
		// By having all printing go through here, we can ALSO write a copy to a file (called a "dribble" file in AI, by some people at least):
		System.out.print(string);
		if (dribbleStream != null) {
			dribbleCharCount += string.length();
			dribbleStream.print(string);
			if (dribbleCharCount > 10000000) {
				dribbleStream.print("\n\n// DRIBBLING TERMINATED SINCE THIS FILE HAS OVER TEN MILLION CHARACTERS IN IT.\n");
				closeDribbleFile();
			}
		}
	}

	public static void println() {
		print("\n");
	}
	public static void println(String string) {
		print(string);
		print("\n"); // Do two calls so no time wasting concatenating strings.
	}

	public static File ensureDirExists(String file) {
		if (file == null) { return null; }
		if (file.endsWith("/") || file.endsWith("\\")) { file += "dummy.txt"; } // A hack to deal with directories being passed in.
		File f = new File(file);

		String parentName = f.getParent();
		File   parentDir  = (parentName == null ? null : f.getParentFile());
		if (parentDir != null) {
			if (!parentDir.exists() && !parentDir.mkdirs()) { // Be careful to not make the file into a directory.
				// waitForEnter("Unable to create (sometimes these are intermittent; will try again)\n   file      = " + file + "\n   parentDir = " + parentDir);
				parentDir.mkdirs();
			}
		}
		return f;
	}

}

class Example implements Serializable {
	public int label;
	public String name,path;

	public Example(String path, int label, String name) {
		this.path = path;
		this.label = label;
		this.name = name;
	}

	public double[][][] getFeatures (double[][][] features) {
		//	long start = System.currentTimeMillis();
		try {

			FileInputStream fis = new FileInputStream(path);

			int content;
			int i = 0,j = 0;
			while ((content = fis.read()) != -1) {
				if (content == 9) {

				} else if (content == 13) {
					i ++;
					j = 0;
				} else {
					String value = "" + ((char) content);
					while ((content = fis.read()) != -1) {
						if (content == 9) {
							for (int k = 0;k < deepNet_burstDetection.unitsPerPixel;k ++) {
								features[i][j][k] = Double.parseDouble (value);
							}
							j ++;
							break;
						} else if (content == 13) {
							for (int k = 0;k < deepNet_burstDetection.unitsPerPixel;k ++) {
								features[i][j][k] = Double.parseDouble (value);
							}
							i ++;
							j = 0;
							break;
						}
						value += ((char) content);
					}
				}
			}
		} catch (Exception e) {}
		//	System.out.println("Took " + deepNet_burstDetection.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to read features from one file.");

		return features;
	}
}

class UnlabelledEx implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public int windowIndex;
	public double windowNumber;
	public String name,path;

	public UnlabelledEx(String path, String name, double windowNumber, int windowIndex) {
		this.path = path;
		this.windowNumber = windowNumber;
		this.windowIndex = windowIndex;
		this.name = name;
	}

	public String getPath() {
		return path;
	}

	public double[][][] getFeatures () {
		//	long start = System.currentTimeMillis();
		double[][][] features = deepNet_burstDetection.features;
		try {

			FileInputStream fis = new FileInputStream(path);

			int content;
			int i = 0,j = 0;
			while ((content = fis.read()) != -1) {
				if (content == 9) {

				} else if (content == 13) {
					i ++;
					j = 0;
				} else {
					String value = "" + ((char) content);
					while ((content = fis.read()) != -1) {
						if (content == 9) {
							for (int k = 0;k < deepNet_burstDetection.unitsPerPixel;k ++) {
								features[i][j][k] = Double.parseDouble (value);
							}
							j ++;
							break;
						} else if (content == 13) {
							for (int k = 0;k < deepNet_burstDetection.unitsPerPixel;k ++) {
								features[i][j][k] = Double.parseDouble (value);
							}
							i ++;
							j = 0;
							break;
						}
						value += ((char) content);
					}
				}
			}
		} catch (Exception e) {}
		//	System.out.println("Took " + deepNet_burstDetection.convertMillisecondsToTimeSpan(System.currentTimeMillis() - start) + " to read features from one file.");
//		features[17][]
		return features;
	}
}



// ALL ADDITIONAL CLASSES


class DeepNet implements Serializable{
	/**
	 * 
	 */
	private static final long serialVersionUID = -1337808277364839659L;
	public static double reluPrime(double sum) {
		return (sum > 0) ? 1 : 0;
	}

	public static double relu(double sum) {
		return (sum > 0) ? sum : 0;
	}

	public static double leakyReluPrime(double sum) {
		return (sum > 0) ? 1 : 0.01;
	}

	public static double leakyRelu(double sum) {
		return (sum > 0) ? sum : 0.01 * sum;
	}

	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static double sigmoidPrime(double x) {
		return DeepNet.sigmoid(x) * (1 - DeepNet.sigmoid(x));
	}

	ArrayList<Layer> layers = new ArrayList<>();
	Layer wideKernel;
	ArrayList<Layer> contextlayers = new ArrayList<>();
	static boolean doneTraining;
	public double contextrate = 0.1;
	public DeepNet(boolean dropOut, boolean normalizeKernelOutputByKernelSum) {
		// Create CNN architecture
		// ConvolutionLayer must always be followed by a pool layer

		// Less readable but may alter code later to handle variable poolSize
		//			int poolSize = 4;
		//			int filterWindow = 5;
		//			int numFilters = 8;
		//			int numFiltersSecConv = 16;
		//
		//			int imageSize = 32;
		//			// First boolean argument added to signify that layer should dropout (if dropout is on) input to layer
		//			this.layers.add(new ConvolutionLayer(imageSize, deepNet_burstDetection.unitsPerPixel, numFilters, filterWindow, dropOut,normalizeKernelOutputByKernelSum));
		//			this.layers.add(new PoolLayer((imageSize-filterWindow+1), numFilters, poolSize, dropOut));// ***may want to test out different numFilters
		//			this.layers.add(new ConvolutionLayer((imageSize-filterWindow+1)/poolSize, numFilters, numFiltersSecConv, filterWindow,dropOut,normalizeKernelOutputByKernelSum));
		//			this.layers.add(new PoolLayer(((imageSize-filterWindow+1)/poolSize)-filterWindow+1, numFiltersSecConv,poolSize,dropOut));
		//			this.layers.add(new FullyConnectedLayer(5, 5, 16, 128,dropOut,layers.get(3)));// **may want to test out varying #nodes in fully connected layer
		//			this.layers.add(new FullyConnectedLayer(128, 1, 1, 6,dropOut,layers.get(4)));
		//			this.doneTraining = false; // necessary for implementing dropout

		// First boolean argument added to signify that layer should dropout (if dropout is on) input to layer
		if (deepNet_burstDetection.useContext) {
			this.contextlayers.add(new ConvolutionLayer(19, 1750, deepNet_burstDetection.unitsPerPixel, 8, 2,51,dropOut,normalizeKernelOutputByKernelSum));
			this.contextlayers.add(new PoolLayer(18,1700, 8,2,dropOut));// ***may want to test out different numFilters
			this.contextlayers.add(new ConvolutionLayer(9,850, 8, 16, 2,51,dropOut,normalizeKernelOutputByKernelSum));
			this.contextlayers.add(new PoolLayer(8,800, 16,4,dropOut));
			this.contextlayers.add(new FullyConnectedLayer(2, 200, 16, 32,dropOut,contextlayers.get(3)));// **may want to test out varying #nodes in fully connected layer
			this.contextlayers.add(new FullyConnectedLayer(32, 1, 1, 2,dropOut,contextlayers.get(4)));
			if (deepNet_burstDetection.useWideKernel) {
				wideKernel = new FullyConnectedLayer(19, 1750, 1, deepNet_burstDetection.wideKernelSize,dropOut,null);
			}
		} else {
			this.layers.add(new ConvolutionLayer(19,351, deepNet_burstDetection.unitsPerPixel, 8, 4,dropOut,normalizeKernelOutputByKernelSum));
			this.layers.add(new PoolLayer(16,348,8,2,dropOut));// ***may want to test out different numFilters
			this.layers.add(new ConvolutionLayer(8,174, 8, 16, 3,dropOut,normalizeKernelOutputByKernelSum));
			this.layers.add(new PoolLayer(6,172, 16,2,dropOut));
			this.layers.add(new FullyConnectedLayer(3, 86, 16, 128,dropOut,layers.get(3)));// **may want to test out varying #nodes in fully connected layer
			this.layers.add(new FullyConnectedLayer(128, 1, 1, 2,dropOut,layers.get(4)));
			if (deepNet_burstDetection.useWideKernel) {
				wideKernel = new FullyConnectedLayer(19, 351, 1, deepNet_burstDetection.wideKernelSize,dropOut,null);
			}
		}
		this.doneTraining = false; // necessary for implementing dropout


		//			this.layers.add(new ConvolutionLayer(32, 4, 20, 5));
		//			this.layers.add(new PoolLayer(28, 20));
		//			this.layers.add(new FullyConnectedLayer(14, 14, 20, 6));

		// Same thing as OneLayer.java:
		//			this.layers.add(new FullyConnectedLayer(32, 32, 4, 50));
		//			this.layers.add(new FullyConnectedLayer(50, 1, 1, 6));
	}

	public void feedforward(double[][][] image) {
		//	        System.out.println(image.length+" "+image[1].length+" "+image[0][1].length);
		if (deepNet_burstDetection.useWideKernel) {
			wideKernel.feedforward (image);
		}
		if (deepNet_burstDetection.useContext) {
			Layer layer = contextlayers.get(0);
			layer.feedforward(image);
			for (int i = 1; i < contextlayers.size(); i++) {
				Layer nextLayer = contextlayers.get(i);
				if (deepNet_burstDetection.useWideKernel && i == 4) {
					nextLayer.feedforward2 (layer.getOutputs (), wideKernel.getOutputs ());
				} else {
					nextLayer.feedforward(layer.getOutputs());
				}
				if (nextLayer instanceof PoolLayer) {
					((PoolLayer) nextLayer).setSums(layer.getSums());
				}
				layer = nextLayer;
			}
		} else {
			Layer layer = layers.get(0);
			layer.feedforward(image);
			for (int i = 1; i < layers.size(); i++) {
				Layer nextLayer = layers.get(i);
				if (deepNet_burstDetection.useWideKernel && i == 4) {
					nextLayer.feedforward2 (layer.getOutputs (), wideKernel.getOutputs ());
				} else {
					nextLayer.feedforward(layer.getOutputs());
				}
				if (nextLayer instanceof PoolLayer) {
					((PoolLayer) nextLayer).setSums(layer.getSums());
				}
				layer = nextLayer;
			}
		}
	}

	public void train(double[][][] image, int label) {
		/*
		 * deepNet_burstDetection.java is written so that we train using flattened 1D feature vectors. This made it simple to
		 * connect the code to OneLayer.java, but is more difficult to reason about in a convolutional net.
		 * So for now this method assumes that we're given a 3D array and a label instead of Vector<Double>
		 */

		this.feedforward(image);

		// Calculate error
		// 6x1x1 array holding the activations of the last layer
		if (deepNet_burstDetection.useContext) {
			Layer contextlastLayer = this.contextlayers.get(this.contextlayers.size() - 1);
			double[][][] contextoutputs = contextlastLayer.getOutputs();
			double[][][] contexterrors = new double[2][1][1];
			double[][][] errors2 = null;
			for (int i = 0; i < 2; i++) {
				double expected = (label == i) ? 1 : 0;
				contexterrors[i][0][0] = contextoutputs[i][0][0] - expected;
			}

			// Backpropagate error, starting with output layer
			for (int i = contextlayers.size() - 1; i >= 0; i--) {
				Layer layer = contextlayers.get(i);
				if (i > 0) {
					Layer nextLayer = contextlayers.get(i - 1);
					if (deepNet_burstDetection.useWideKernel && i == 4) {
						errors2 = layer.backprop2 (contexterrors, wideKernel.getSums ());
					}
					contexterrors = layer.backprop(contexterrors, nextLayer.getSums());
				} else {
					contexterrors = layer.backprop(contexterrors, null);
				}
			}
			if (deepNet_burstDetection.useWideKernel && errors2 != null) {
				wideKernel.backprop (errors2, null);
			}
		} else {
			Layer lastLayer = this.layers.get(this.layers.size() - 1);
			double[][][] outputs = lastLayer.getOutputs();
			double[][][] errors = new double[2][1][1];
			double[][][] errors2 = null;
			for (int i = 0; i < 2; i++) {
				double expected = (label == i) ? 1 : 0;
				errors[i][0][0] = outputs[i][0][0] - expected;
			}

			// Backpropagate error, starting with output layer
			for (int i = layers.size() - 1; i >= 0; i--) {
				Layer layer = layers.get(i);
				if (i > 0) {
					Layer nextLayer = layers.get(i - 1);
					if (deepNet_burstDetection.useWideKernel && i == 4) {
						errors2 = layer.backprop2 (errors, wideKernel.getSums ());
					}
					errors = layer.backprop(errors, nextLayer.getSums());
				} else {
					errors = layer.backprop(errors, null);
				}
			}
			if (deepNet_burstDetection.useWideKernel && errors2 != null) {
				wideKernel.backprop (errors2, null);
			}
		}
	}

	public int getLabel(double[][][] image) {
		this.doneTraining = true;
		this.feedforward(image);

		int maxIndex = 0;
		if (deepNet_burstDetection.useContext) {
			Layer contextlastLayer = this.contextlayers.get(this.contextlayers.size() - 1);
			double[][][] contextoutputs = contextlastLayer.getOutputs();
			for (int i = 1; i < 2; i++) {
				if (contextoutputs[i][0][0] > contextoutputs[maxIndex][0][0]) {
					maxIndex = i;
				}
			}
		} else {
			Layer lastLayer = this.layers.get(this.layers.size() - 1);
			double[][][] outputs = lastLayer.getOutputs();

			for (int i = 1; i < 2; i++) {
				if (outputs[i][0][0] > outputs[maxIndex][0][0]) {
					maxIndex = i;
				}
			}
		}
		return maxIndex;
	}

}

interface Layer {
	public void feedforward(double[][][] input);
	public void setBiases(double[] convLayer1Biases);
	public void setWeights(double[][][][] convLayer1Weights);
	public double[][][][] getWeights();
	public double[] getBiases();
	public void feedforward2(double[][][] input, double[][][] input2);
	public double[][][] backprop(double[][][] error, double[][][] sums);
	public double[][][] backprop2(double[][][] error, double[][][] sums);
	public double[][][] getSums();
	public double[][][] getOutputs();
	public double[][][] getDropOutFlags();
}

class ConvolutionLayer implements Layer, Serializable{
	public double[][][] inputs;//
	// Filter weights. Expected order: width x height x depth x filter
	public double[][][][] weights;
	// Filter activations. Expected order: width x height x filter
	public double[][][] outputs;
	public double[][][] sums;
	public double[] biases;
	public double[] filterNormalizer; // sum of weights in kernel/filter; used to normalize outputs such that intensities of output remain consistent across images
	public boolean normalizeKernelOutputByKernelSum;// turn this flag on/off to turn normalization of kernel output on/off

	int inputSize, inputDepth, numFilters, filterSize, outputSize;
	int inputLength, inputWidth, outputLength, outputWidth, filterWidth, filterLength;
	public boolean dropOutLayer; // for dropout purposes
	public double [][][] inputDropOutFlags;

	public ConvolutionLayer(int inputSize, int inputDepth, int numFilters, int filterSize, boolean dropOutLayer, boolean normalizeKernelOutputByKernelSum) {
		this.inputSize = inputSize;
		this.inputDepth = inputDepth;
		this.numFilters = numFilters;
		this.filterSize = filterSize;
		this.dropOutLayer = dropOutLayer;
		this.normalizeKernelOutputByKernelSum = normalizeKernelOutputByKernelSum;

		if (filterSize % 2 != 1) {
			System.out.println("filterSize must be odd");
			System.exit(1);
		}

		this.weights = new double[filterSize][filterSize][inputDepth][numFilters];
		this.biases = new double[this.numFilters];
		this.outputSize = inputSize - filterSize + 1;
		this.outputs = new double[this.outputSize][this.outputSize][numFilters];
		this.sums = new double[this.outputSize][this.outputSize][numFilters];
		this.filterNormalizer = new double[numFilters];

		for (int x = 0; x < filterSize; x++) {
			for (int y = 0; y < filterSize; y++) {
				for (int z = 0; z < inputDepth; z++) {
					for (int filter = 0; filter < numFilters; filter++) {
						this.weights[x][y][z][filter] = deepNet_burstDetection.getRandomWeight(inputSize * inputSize * inputDepth, outputSize * outputSize * numFilters);
					}
				}
			}
		}
	}
	/**
	 * Additional constructor for Final Project to be able to process trial image(not square)
	 */
	public ConvolutionLayer(int inputLength, int inputWidth, int inputDepth, int numFilters, int filterSize, boolean dropOutLayer, boolean normalizeKernelOutputByKernelSum) {
		this.inputLength = inputLength;
		this.inputWidth = inputWidth;
		this.inputDepth = inputDepth;
		this.numFilters = numFilters;
		this.filterSize = filterSize;
		this.dropOutLayer = dropOutLayer;
		this.normalizeKernelOutputByKernelSum = normalizeKernelOutputByKernelSum;

		this.weights = new double[filterSize][filterSize][inputDepth][numFilters];
		this.biases = new double[this.numFilters];
		this.outputLength = inputLength - filterSize + 1;
		this.outputWidth = inputWidth - filterSize + 1;
		this.outputs = new double[this.outputLength][this.outputWidth][numFilters];
		this.sums = new double[this.outputLength][this.outputWidth][numFilters];
		this.filterNormalizer = new double[numFilters];

		for (int x = 0; x < filterSize; x++) {
			for (int y = 0; y < filterSize; y++) {
				for (int z = 0; z < inputDepth; z++) {
					for (int filter = 0; filter < numFilters; filter++) {
						this.weights[x][y][z][filter] = deepNet_burstDetection.getRandomWeight(inputLength * inputWidth * inputDepth, outputLength * outputWidth * numFilters);
					}
				}
			}
		}
	}
	public ConvolutionLayer(int inputLength, int inputWidth, int inputDepth, int numFilters, int filterLength, int filterWidth, boolean dropOutLayer, boolean normalizeKernelOutputByKernelSum) {
		this.inputLength = inputLength;
		this.inputWidth = inputWidth;
		this.inputDepth = inputDepth;
		this.numFilters = numFilters;
		this.filterLength = filterLength;
		this.filterWidth = filterWidth;
		this.dropOutLayer = dropOutLayer;
		this.normalizeKernelOutputByKernelSum = normalizeKernelOutputByKernelSum;



		this.weights = new double[filterLength][filterWidth][inputDepth][numFilters];
		this.biases = new double[this.numFilters];
		this.outputLength = inputLength - filterLength + 1;
		this.outputWidth = inputWidth - filterWidth + 1;
		this.outputs = new double[this.outputLength][this.outputWidth][numFilters];
		this.sums = new double[this.outputLength][this.outputWidth][numFilters];
		this.filterNormalizer = new double[numFilters];

		for (int x = 0; x < filterLength; x++) {
			for (int y = 0; y < filterWidth; y++) {
				for (int z = 0; z < inputDepth; z++) {
					for (int filter = 0; filter < numFilters; filter++) {
						this.weights[x][y][z][filter] = deepNet_burstDetection.getRandomWeight(inputLength * inputWidth * inputDepth, outputLength * outputWidth * numFilters);
					}
				}
			}
		}
	}
	// Expected input order: width x height x depth
	// Walks through the output array, calculating the value that should go into each position
	public void feedforward(double[][][] inputs) {
		// update dropout flags for each example
		// the code inside if statement is the previous code for lab 3
		if(inputSize != 0) {
			if(dropOutLayer){
				this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
				if(deepNet_burstDetection.inputDropoutRate <= 0.0 || DeepNet.doneTraining){
					// effectively turns off dropOut
					for(double rowCol[][]: inputDropOutFlags){
						for(double depth[]: rowCol){
							Arrays.fill(depth, 1.0);
						}
					}
				}else{
					// flag updates
					this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
					for(int i = 0; i < inputSize; i++) {
						for(int j = 0; j < inputSize; j++) {
							for(int k = 0; k < inputDepth; k++) {
								if (Math.random() > deepNet_burstDetection.inputDropoutRate) inputDropOutFlags[i][j][k] = 1.0;
							}
						}
					}
				}
			}
			this.inputs = inputs;
			for (int startX = 0; startX < this.outputSize; startX++) {
				for (int startY = 0; startY < this.outputSize; startY++) {
					for (int filter = 0; filter < this.numFilters; filter++) {
						this.sums[startX][startY][filter] = this.applyFilter(inputs, startX, startY, filter) + this.biases[filter];
						if(this.normalizeKernelOutputByKernelSum) this.sums[startX][startY][filter] /= this.filterNormalizer[filter];

						this.outputs[startX][startY][filter] = DeepNet.leakyRelu(this.sums[startX][startY][filter]);
					}
				}
			}
		} else {
			// this chunk of code is for processing trial image(context network)
			if(dropOutLayer){
				this.inputDropOutFlags = new double[inputLength][inputWidth][inputDepth];
				if(deepNet_burstDetection.inputDropoutRate <= 0.0 || DeepNet.doneTraining){
					// effectively turns off dropOut
					for(double rowCol[][]: inputDropOutFlags){
						for(double depth[]: rowCol){
							Arrays.fill(depth, 1.0);
						}
					}
				}else{
					// flag updates
					this.inputDropOutFlags = new double[inputLength][inputWidth][inputDepth];
					for(int i = 0; i < inputLength; i++) {
						for(int j = 0; j < inputWidth; j++) {
							for(int k = 0; k < inputDepth; k++) {
								if (Math.random() > deepNet_burstDetection.inputDropoutRate) inputDropOutFlags[i][j][k] = 1.0;
							}
						}
					}
				}
			}
			this.inputs = inputs;
			for (int startX = 0; startX < this.outputLength; startX++) {
				for (int startY = 0; startY < this.outputWidth; startY++) {
					for (int filter = 0; filter < this.numFilters; filter++) {
						this.sums[startX][startY][filter] = this.applyFilter(inputs, startX, startY, filter) + this.biases[filter];
						if(this.normalizeKernelOutputByKernelSum) this.sums[startX][startY][filter] /= this.filterNormalizer[filter];

						this.outputs[startX][startY][filter] = DeepNet.leakyRelu(this.sums[startX][startY][filter]);
					}
				}
			}
		}
	}

	public void feedforward2(double[][][] input, double[][][] input2) {}

	public double[][][] backprop(double[][][] errors, double[][][] sums) {
		if(inputSize != 0) {
			double[][][] nextErrors = new double[this.inputSize][this.inputSize][this.inputDepth];
			double [][][] dropOutFlags = this.getDropOutFlags();

			if (sums != null) {
				for (int x = 0; x < this.outputSize; x++) {
					for (int y = 0; y < this.outputSize; y++) {
						for (int filter = 0; filter < this.numFilters; filter++) {
							double error = errors[x][y][filter];
							if (error == 0) {
								continue;
							}
							for (int filterX = 0; filterX < this.filterSize; filterX++) {
								for (int filterY = 0; filterY < this.filterSize; filterY++) {
									for (int filterZ = 0; filterZ < this.inputDepth; filterZ++) {
										//											if(this.dropOutLayer == true){
										//												nextErrors[x + filterX][y + filterY][filterZ] += this.weights[filterX][filterY][filterZ][filter] * error * dropOutFlags[x + filterX][y + filterY][filterZ];
										//											}else{
										nextErrors[x + filterX][y + filterY][filterZ] += this.weights[filterX][filterY][filterZ][filter] * error;
										//											}
									}
								}
							}
						}
					}
				}

				for (int x = 0; x < this.inputSize; x++) {
					for (int y = 0; y < this.inputSize; y++) {
						for (int z = 0; z < this.inputDepth; z++) {
							if(this.dropOutLayer == true){
								nextErrors[x][y][z] *= DeepNet.leakyReluPrime(sums[x][y][z]) * dropOutFlags[x][y][z];
							}else{
								nextErrors[x][y][z] *= DeepNet.leakyReluPrime(sums[x][y][z]);
							}
						}
					}
				}
			}

			for (int x = 0; x < this.outputSize; x++) {
				for (int y = 0; y < this.outputSize; y++) {
					for (int filter = 0; filter < this.numFilters; filter++) {
						double error = errors[x][y][filter];
						this.biases[filter] -= deepNet_burstDetection.eta * error * (1.0 / (this.outputSize * this.outputSize)); // Update bias
						this.updateWeights(error * (1.0 / (this.filterSize * this.filterSize)), x, y, filter);
					}
				}
			}
			return nextErrors;
		} else {
			double[][][] nextErrors = new double[this.inputLength][this.inputWidth][this.inputDepth];

			double [][][] dropOutFlags = this.getDropOutFlags();

			if (sums != null) {
				for (int x = 0; x < this.outputLength; x++) {
					for (int y = 0; y < this.outputWidth; y++) {
						for (int filter = 0; filter < this.numFilters; filter++) {
							double error = errors[x][y][filter];
							if (error == 0) {
								continue;
							}
							for (int filterX = 0; filterX < this.filterSize; filterX++) {
								for (int filterY = 0; filterY < this.filterSize; filterY++) {
									for (int filterZ = 0; filterZ < this.inputDepth; filterZ++) {
										//											if(this.dropOutLayer == true){
										//												nextErrors[x + filterX][y + filterY][filterZ] += this.weights[filterX][filterY][filterZ][filter] * error * dropOutFlags[x + filterX][y + filterY][filterZ];
										//											}else{
										nextErrors[x + filterX][y + filterY][filterZ] += this.weights[filterX][filterY][filterZ][filter] * error;
										//											}
									}
								}
							}
						}
					}
				}

				for (int x = 0; x < this.inputLength; x++) {
					for (int y = 0; y < this.inputWidth; y++) {
						for (int z = 0; z < this.inputDepth; z++) {
							if(this.dropOutLayer == true){
								nextErrors[x][y][z] *= DeepNet.leakyReluPrime(sums[x][y][z]) * dropOutFlags[x][y][z];
							}else{
								nextErrors[x][y][z] *= DeepNet.leakyReluPrime(sums[x][y][z]);
							}
						}
					}
				}
			}

			for (int x = 0; x < this.outputLength; x++) {
				for (int y = 0; y < this.outputWidth; y++) {
					for (int filter = 0; filter < this.numFilters; filter++) {
						double error = errors[x][y][filter];
						this.biases[filter] -= deepNet_burstDetection.eta * error * (1.0 / (this.outputLength * this.outputWidth)); // Update bias
						this.updateWeights(error * (1.0 / (this.filterSize * this.filterSize)), x, y, filter);
					}
				}
			}
			return nextErrors;
		}

	}

	public double[][][] backprop2(double[][][] error, double[][][] sums) {
		return error;
	}

	public void updateWeights(double error, int startX, int startY, int filter) {
		for (int x = 0; x < this.filterSize; x++) {
			for (int y = 0; y < this.filterSize; y++) {
				for (int z = 0; z < this.inputDepth; z++) {
					double input = this.inputs[startX + x][startY + y][z];
					this.weights[x][y][z][filter] -= deepNet_burstDetection.eta * input * error;
				}
			}
		}
	}

	double applyFilter(double[][][] inputs, int startX, int startY, int filter) {
		if(!DeepNet.doneTraining){
			this.filterNormalizer[filter] = 0;
			for (int x = 0; x < this.filterSize; x++) {
				for (int y = 0; y < this.filterSize; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						this.filterNormalizer[filter] += this.weights[x][y][z][filter];
					}
				}
			}
		}

		double sum = 0;
		for (int x = 0; x < this.filterSize; x++) {
			for (int y = 0; y < this.filterSize; y++) {
				for (int z = 0; z < this.inputDepth; z++) {
					if(!DeepNet.doneTraining){
						if(dropOutLayer){
							sum += inputs[startX + x][startY + y][z] * this.weights[x][y][z][filter] * inputDropOutFlags[startX + x][startY + y][z];
						}else{
							sum += inputs[startX + x][startY + y][z] * this.weights[x][y][z][filter];
						}
					}else{
						sum += inputs[startX + x][startY + y][z] * this.weights[x][y][z][filter] * (1-deepNet_burstDetection.inputDropoutRate);
					}
				}
			}
		}
		return sum;


	}

	public double[][][] getSums() {
		return this.sums;
	}

	public double[][][] getOutputs() {
		return this.outputs;
	}

	public double[][][] getDropOutFlags() {
		return inputDropOutFlags;
	}
	@Override
	public double[][][][] getWeights() {
		return this.weights;
	}
	public double[] getBiases() {
		return this.biases;
	}
	@Override
	public void setWeights(double[][][][] weights) {
		this.weights = weights;
	}
	@Override
	public void setBiases(double[] biases) {
		this.biases = biases;
	}
}

class PoolLayer implements Layer, Serializable {
	double[][][] sums;
	double[][][] outputs;
	double[][][] dropOutFlags; // decided to not implement for pooling layers

	int inputSize, inputDepth, poolSize, outputSize;
	int inputLength, inputWidth, outputLength, outputWidth;
	int[][][] sources;

	public boolean dropOutLayer; // for dropout purposes
	public double [][][] inputDropOutFlags;

	public PoolLayer(int inputSize, int inputDepth, int poolSize, boolean dropOutLayer) { // int window 2; stride = 2 // for overlap, change to stride = 1
		this.inputSize = inputSize;
		this.inputDepth = inputDepth;
		this.poolSize = poolSize;

		if (inputSize % poolSize != 0) {
			System.out.println("inputSize (" + inputSize + ") not divisible by poolSize (" + poolSize + ")");
			System.exit(1);
		}

		this.outputSize = inputSize / poolSize;
		this.outputs = new double[outputSize][outputSize][inputDepth];
		this.sources = new int[inputSize][inputSize][inputDepth];
	}
	public PoolLayer(int inputLength, int inputWidth, int inputDepth, int poolSize, boolean dropOutLayer) {
		this.inputLength = inputLength;
		this.inputWidth = inputWidth;
		this.inputDepth = inputDepth;
		this.poolSize = poolSize;

		if (inputLength % poolSize != 0) {
			System.out.println("inputLength (" + inputLength + ") not divisible by poolSize (" + poolSize + ")");
			System.exit(1);
		}
		if (inputWidth % poolSize != 0) {
			System.out.println("inputWidth (" + inputWidth + ") not divisible by poolSize (" + poolSize + ")");
			System.exit(1);
		}

		this.outputLength = inputLength / poolSize;
		this.outputWidth = inputWidth / poolSize;
		this.outputs = new double[outputLength][outputWidth][inputDepth];
		this.sources = new int[inputLength][inputWidth][inputDepth];
	}

	// Walks through the output array, calculating the value that should go into each position
	public void feedforward(double[][][] inputs) {
		// update dropout flags for each example
		if(inputSize != 0) {
			if(dropOutLayer){
				this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
				if(deepNet_burstDetection.inputDropoutRate <= 0.0 || DeepNet.doneTraining){
					// effectively turns off dropOut
					for(double rowCol[][]: inputDropOutFlags){
						for(double depth[]: rowCol){
							Arrays.fill(depth, 1.0);
						}
					}
				}else{
					// flag updates
					this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
					for(int i = 0; i < inputSize; i++) {
						for(int j = 0; j < inputSize; j++) {
							for(int k = 0; k < inputDepth; k++) {
								if (Math.random() > deepNet_burstDetection.inputDropoutRate) inputDropOutFlags[i][j][k] = 1.0;
							}
						}
					}
				}
			}
			for (int outputX = 0; outputX < this.outputSize; outputX++) {
				for (int outputY = 0; outputY < this.outputSize; outputY++) {
					for (int z = 0; z < this.inputDepth; z++) {
						this.outputs[outputX][outputY][z] = this.calculateMax(inputs, outputX, outputY, z);
					}
				}
			}
		} else {
			if(dropOutLayer){
				this.inputDropOutFlags =  new double[inputLength][inputWidth][inputDepth];
				if(deepNet_burstDetection.inputDropoutRate <= 0.0 || DeepNet.doneTraining){
					// effectively turns off dropOut
					for(double rowCol[][]: inputDropOutFlags){
						for(double depth[]: rowCol){
							Arrays.fill(depth, 1.0);
						}
					}
				}else{
					// flag updates
					this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
					for(int i = 0; i < inputLength; i++) {
						for(int j = 0; j < inputWidth; j++) {
							for(int k = 0; k < inputDepth; k++) {
								if (Math.random() > deepNet_burstDetection.inputDropoutRate) inputDropOutFlags[i][j][k] = 1.0;
							}
						}
					}
				}
			}
			for (int outputX = 0; outputX < this.outputLength; outputX++) {
				for (int outputY = 0; outputY < this.outputWidth; outputY++) {
					for (int z = 0; z < this.inputDepth; z++) {
						this.outputs[outputX][outputY][z] = this.calculateMax(inputs, outputX, outputY, z);
					}
				}
			}
		}
	}

	public void feedforward2(double[][][] input, double[][][] input2) {}

	public double[][][] backprop(double[][][] errors, double[][][] sums) {
		// Returns a sparse matrix that copies the errors to each source
		// For example [[ 1, 2 ], [ 3, 4 ]] would give us [4] in the forward pass, and a sources matrix that looks like:
		// [[ 0, 0 ], [ 0, 1 ]]. If the errors matrix is [d], this method would then return [[ 0, 0 ], [ 0, d ]]
		double [][][] dropOutFlags = this.getDropOutFlags();
		if(inputSize != 0) {
			double[][][] nextError = new double[inputSize][inputSize][inputDepth];
			for (int x = 0; x < this.inputSize; x++) {
				for (int y = 0; y < this.inputSize; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						if (this.sources[x][y][z] == 1) {
							if(this.dropOutLayer == true){
								nextError[x][y][z] = errors[x / this.poolSize][y / this.poolSize][z] * dropOutFlags[x][y][z];
							}else{
								nextError[x][y][z] = errors[x / this.poolSize][y / this.poolSize][z];
							}
						} else {
							nextError[x][y][z] = 0;
						}
					}
				}
			}

			return nextError;
		} else {
			double[][][] nextError = new double[inputLength][inputWidth][inputDepth];
			for (int x = 0; x < this.inputLength; x++) {
				for (int y = 0; y < this.inputWidth; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						if (this.sources[x][y][z] == 1) {
							if(this.dropOutLayer == true){
								nextError[x][y][z] = errors[x / this.poolSize][y / this.poolSize][z] * dropOutFlags[x][y][z];
							}else{
								nextError[x][y][z] = errors[x / this.poolSize][y / this.poolSize][z];
							}
						} else {
							nextError[x][y][z] = 0;
						}
					}
				}
			}

			return nextError;
		}

	}

	public double[][][] backprop2(double[][][] error, double[][][] sums) {
		return error;
	}

	double calculateMax(double[][][] inputs, int outputX, int outputY, int z) {
		int startX = outputX * this.poolSize;
		int startY = outputY * this.poolSize;
		double max = Double.NEGATIVE_INFINITY;
		int xIndex = 0;
		int yIndex = 0;
		for (int x = startX; x < startX + this.poolSize; x++) {
			for (int y = startY; y < startY + this.poolSize; y++) {
				if (max < inputs[x][y][z]) {
					max = inputs[x][y][z];
					xIndex = x;
					yIndex = y;
				}
			}
		}
		this.sources[xIndex][yIndex][z] = 1;
		return max;
	}

	public double[][][] setSums(double[][][] lastSums) {
		if(inputSize != 0) {
			this.sums = new double[outputSize][outputSize][inputDepth];

			for (int x = 0; x < this.inputSize; x++) {
				for (int y = 0; y < this.inputSize; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						if (this.sources[x][y][z] == 1) {
							this.sums[x / this.poolSize][y / this.poolSize][z] = lastSums[x][y][z];
						}
					}
				}
			}
			return this.sums;
		} else {
			this.sums = new double[outputLength][outputWidth][inputDepth];

			for (int x = 0; x < this.inputLength; x++) {
				for (int y = 0; y < this.inputWidth; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						if (this.sources[x][y][z] == 1) {
							this.sums[x / this.poolSize][y / this.poolSize][z] = lastSums[x][y][z];
						}
					}
				}
			}
			return this.sums;
		}

	}

	public double[][][] getSums() {
		return this.sums;
	}

	public double[][][] getOutputs() {
		return this.outputs;
	}

	public double[][][] getDropOutFlags() {
		return dropOutFlags;
	}
	@Override
	public double[][][][] getWeights() {
		// Pool layer has no weights, so no need to get weights here
		return null;
	}
	@Override
	public double[] getBiases() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public void setWeights(double[][][][] convLayer1Weights) {
		// TODO Auto-generated method stub

	}
	@Override
	public void setBiases(double[] convLayer1Biases) {
		// TODO Auto-generated method stub

	}
}

class FullyConnectedLayer implements Layer, Serializable {
	int inputWidth, inputHeight, inputDepth, numOutputs;
	double[][][][] weights;
	int inputWidth2, inputHeight2, inputDepth2;
	double[][][][] weights2;
	double[] biases;
	double[][][] sums;
	double[][][] inputs;
	double[][][] inputs2;
	double[][][] outputs;

	// drop out vars
	boolean dropOutLayer;
	double[][][] fullyConnectedLayerDropOutFlags;
	public Layer prevLayer;
	public double[] combine(double[] a, double b[]){
		double[] ret;
		return null;
	}
	public FullyConnectedLayer combine(FullyConnectedLayer newlayer){
		inputWidth = this.inputWidth + newlayer.inputWidth;
		FullyConnectedLayer combinelayer = new FullyConnectedLayer(inputWidth, this.inputHeight, this.inputDepth, this.numOutputs,this.dropOutLayer , this.prevLayer);
		newlayer.sums = this.sums;
		return null;
	}
	public FullyConnectedLayer(int inputWidth, int inputHeight, int inputDepth, int numOutputs, boolean dropOutLayer, Layer prevLayer) {
		this.inputWidth = inputWidth;
		this.inputHeight = inputHeight;
		this.inputDepth = inputDepth;
		this.numOutputs = numOutputs;
		this.dropOutLayer = dropOutLayer;
		this.prevLayer = prevLayer;

		this.weights = new double[inputWidth][inputHeight][inputDepth][numOutputs];
		for (int x = 0; x < inputWidth; x++) {
			for (int y = 0; y < inputHeight; y++) {
				for (int z = 0; z < inputDepth; z++) {
					for (int output = 0; output < numOutputs; output++) {
						this.weights[x][y][z][output] = deepNet_burstDetection.getRandomWeight(inputWidth * inputHeight * inputDepth, numOutputs);
					}
				}
			}
		}

		this.inputWidth2 = deepNet_burstDetection.wideKernelSize;
		this.inputHeight2 = 1;
		this.inputDepth2 = 1;
		this.weights2 = new double[inputWidth2][inputHeight2][inputDepth2][numOutputs];
		for (int x = 0; x < inputWidth2; x++) {
			for (int y = 0; y < inputHeight2; y++) {
				for (int z = 0; z < inputDepth2; z++) {
					for (int output = 0; output < numOutputs; output++) {
						this.weights2[x][y][z][output] = deepNet_burstDetection.getRandomWeight(inputWidth * inputHeight * inputDepth, numOutputs);
					}
				}
			}
		}
		// For typing reasons, we need this array to be 3D
		this.outputs = new double[numOutputs][1][1];
		this.biases = new double[numOutputs];
		this.sums = new double[numOutputs][1][1];
	}

	public void feedforward(double[][][] inputs) {
		this.inputs = inputs;
		if(dropOutLayer){
			// Update dropOut flags for each example
			fullyConnectedLayerDropOutFlags = new double[inputWidth][inputHeight][inputDepth];
			if(deepNet_burstDetection.hiddenDropoutRate <= 0.0 || DeepNet.doneTraining){
				// effectively turns off dropOut
				for(double rowCol[][]: fullyConnectedLayerDropOutFlags){
					for(double depth[]: rowCol){
						Arrays.fill(depth, 1.0);
					}
				}
			}else{
				// flag updates
				for(int i = 0; i < inputWidth; i++) {
					for(int j = 0; j < inputHeight; j++) {
						for(int k = 0; k < inputDepth; k++) {
							if (Math.random() > deepNet_burstDetection.hiddenDropoutRate) fullyConnectedLayerDropOutFlags[i][j][k] = 1.0;
						}
					}
				}
			}
		}
		for (int output = 0; output < this.numOutputs; output++) {
			this.sums[output][0][0] = this.biases[output];
			for (int x = 0; x < this.inputWidth; x++) {
				for (int y = 0; y < this.inputHeight; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						if(!DeepNet.doneTraining){
							if(dropOutLayer){
								this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output] * fullyConnectedLayerDropOutFlags[x][y][z];
							}else{
								this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output];
							}
						}else{
							this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output] * (1 - deepNet_burstDetection.hiddenDropoutRate);
						}
					}
				}
			}
		}

		for (int output = 0; output < this.numOutputs; output++) {
			this.outputs[output][0][0] = DeepNet.sigmoid(this.sums[output][0][0]);
		}
	}

	public void feedforward2 (double[][][] inputs, double[][][] inputs2) {
		this.inputs = inputs;
		this.inputs2 = inputs2;
		if(dropOutLayer){
			// Update dropOut flags for each example
			fullyConnectedLayerDropOutFlags = new double[inputWidth][inputHeight][inputDepth];
			if(deepNet_burstDetection.hiddenDropoutRate <= 0.0 || DeepNet.doneTraining){
				// effectively turns off dropOut
				for(double rowCol[][]: fullyConnectedLayerDropOutFlags){
					for(double depth[]: rowCol){
						Arrays.fill(depth, 1.0);
					}
				}
			}else{
				// flag updates
				for(int i = 0; i < inputWidth; i++) {
					for(int j = 0; j < inputHeight; j++) {
						for(int k = 0; k < inputDepth; k++) {
							if (Math.random() > deepNet_burstDetection.hiddenDropoutRate) fullyConnectedLayerDropOutFlags[i][j][k] = 1.0;
						}
					}
				}
			}
		}
		for (int output = 0; output < this.numOutputs; output++) {
			this.sums[output][0][0] = this.biases[output];
			for (int x = 0; x < this.inputWidth; x++) {
				for (int y = 0; y < this.inputHeight; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						if(!DeepNet.doneTraining){
							if(dropOutLayer){
								this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output] * fullyConnectedLayerDropOutFlags[x][y][z];
							}else{
								this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output];
							}
						}else{
							this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output] * (1 - deepNet_burstDetection.hiddenDropoutRate);
						}
					}
				}
			}
			for (int x = 0; x < this.inputWidth2; x++) {
				for (int y = 0; y < this.inputHeight2; y++) {
					for (int z = 0; z < this.inputDepth2; z++) {
						if(!DeepNet.doneTraining){
							if(dropOutLayer){
								this.sums[output][0][0] += inputs2[x][y][z] * this.weights2[x][y][z][output] * fullyConnectedLayerDropOutFlags[x][y][z];
							}else{
								this.sums[output][0][0] += inputs2[x][y][z] * this.weights2[x][y][z][output];
							}
						}else{
							this.sums[output][0][0] += inputs2[x][y][z] * this.weights2[x][y][z][output] * (1 - deepNet_burstDetection.hiddenDropoutRate);
						}
					}
				}
			}
		}

		for (int output = 0; output < this.numOutputs; output++) {
			this.outputs[output][0][0] = DeepNet.sigmoid(this.sums[output][0][0]);
		}
	}

	public double[][][] backprop(double[][][] errors, double[][][] sums) {
		double[][][] nextErrors = new double[this.inputWidth][this.inputHeight][this.inputDepth];

		// sums == null when this is the first layer in the network, and there is nothing to pass errors back to
		if (sums != null) {

			double [][][] dropOutFlags = this.getDropOutFlags();

			// Calculate the errors to be passed into the next layer
			for (int x = 0; x < this.inputWidth; x++) {
				for (int y = 0; y < this.inputHeight; y++) {
					for (int z = 0; z < this.inputDepth; z++) {
						nextErrors[x][y][z] = 0;
						for (int output = 0; output < this.numOutputs; output++) {
							if(this.dropOutLayer == true){
								// - Signifies you're looking at inputs from layer that had dropout (i.e. second fullyConnectedLayer)
								// - Multiply nextErrors by dropOut flags used during forward prop
								nextErrors[x][y][z] += this.weights[x][y][z][output] * errors[output][0][0] * dropOutFlags[x][y][z];

							}else{
								nextErrors[x][y][z] += this.weights[x][y][z][output] * errors[output][0][0];
							}
						}
						nextErrors[x][y][z] *= DeepNet.sigmoidPrime(sums[x][y][z]);
					}
				}
			}
		}

		// Perform bias updates
		for (int output = 0; output < this.numOutputs; output++) {
			this.biases[output] -= deepNet_burstDetection.eta * errors[output][0][0];
		}

		// Perform weight updates
		for (int x = 0; x < this.inputWidth; x++) {
			for (int y = 0; y < this.inputHeight; y++) {
				for (int z = 0; z < this.inputDepth; z++) {
					for (int output = 0; output < this.numOutputs; output++) {
						this.weights[x][y][z][output] -= deepNet_burstDetection.eta * this.inputs[x][y][z] * errors[output][0][0];
					}
				}
			}
		}

		return nextErrors;
	}

	public double[][][] backprop2(double[][][] errors, double[][][] sums) {
		double[][][] nextErrors = new double[this.inputWidth2][this.inputHeight2][this.inputDepth2];

		// sums == null when this is the first layer in the network, and there is nothing to pass errors back to
		if (sums != null) {

			double [][][] dropOutFlags = this.getDropOutFlags();

			// Calculate the errors to be passed into the next layer
			for (int x = 0; x < this.inputWidth2; x++) {
				for (int y = 0; y < this.inputHeight2; y++) {
					for (int z = 0; z < this.inputDepth2; z++) {
						nextErrors[x][y][z] = 0;
						for (int output = 0; output < this.numOutputs; output++) {
							if(this.dropOutLayer == true){
								// - Signifies you're looking at inputs from layer that had dropout (i.e. second fullyConnectedLayer)
								// - Multiply nextErrors by dropOut flags used during forward prop
								nextErrors[x][y][z] += this.weights2[x][y][z][output] * errors[output][0][0] * dropOutFlags[x][y][z];

							}else{
								nextErrors[x][y][z] += this.weights2[x][y][z][output] * errors[output][0][0];
							}
						}
						nextErrors[x][y][z] *= DeepNet.sigmoidPrime(sums[x][y][z]);
					}
				}
			}
		}

		// Perform weight updates
		for (int x = 0; x < this.inputWidth2; x++) {
			for (int y = 0; y < this.inputHeight2; y++) {
				for (int z = 0; z < this.inputDepth2; z++) {
					for (int output = 0; output < this.numOutputs; output++) {
						this.weights2[x][y][z][output] -= deepNet_burstDetection.eta * this.inputs2[x][y][z] * errors[output][0][0];
					}
				}
			}
		}

		return nextErrors;
	}

	public double[][][] getSums() {
		return this.sums;
	}

	public double[][][] getOutputs() {
		return this.outputs;
	}
	public double[] getBiases() {
		return this.biases;
	}

	public double[][][] getDropOutFlags() {
		return fullyConnectedLayerDropOutFlags;
	}
	@Override
	public double[][][][] getWeights() {
		return this.weights;
	}
	@Override
	public void setWeights(double[][][][] weights) {
		this.weights = weights;
	}
	@Override
	public void setBiases(double[] biases) {
		this.biases = biases;
	}
}
class OneLayer {
	private final int numOutputClasses = 6;// just in case we want reduce output number for experimenting
	int numFeatures = deepNet_burstDetection.inputVectorSize - 1;
	double[][] hiddenWeights = new double[deepNet_burstDetection.numberOfHiddenUnits][numFeatures];
	double[][] outputWeights = new double[numOutputClasses][deepNet_burstDetection.numberOfHiddenUnits];
	double[][] hiddenWeightMomentums = new double[deepNet_burstDetection.numberOfHiddenUnits][numFeatures];
	double[][] outputWeightMomentums = new double[numOutputClasses][deepNet_burstDetection.numberOfHiddenUnits];
	double[] hiddenBiases = new double[deepNet_burstDetection.numberOfHiddenUnits];
	double[] outputBiases = new double[numOutputClasses];
	double[] hiddenBiasMomentums = new double[deepNet_burstDetection.numberOfHiddenUnits];
	double[] outputBiasMomentums = new double[numOutputClasses];

	double[] outputActivations = new double[numOutputClasses];
	double[] outputSums = new double[numOutputClasses];
	double[] hiddenActivations = new double[deepNet_burstDetection.numberOfHiddenUnits];
	double[] hiddenSums = new double[deepNet_burstDetection.numberOfHiddenUnits];

	private double[] outputError;
	private double[] hiddenError;

	private double[] hiddenDropOutFlags;
	private double[] inputDropOutFlags;

	private boolean finishedTraining = false;// boolean for enabling dropout during training

	public OneLayer() {
		for (int i = 0; i < deepNet_burstDetection.numberOfHiddenUnits; i++) {
			for (int j = 0; j < numFeatures; j++) {
				hiddenWeights[i][j] = deepNet_burstDetection.getRandomWeight(numFeatures, deepNet_burstDetection.numberOfHiddenUnits);
			}
		}

		for (int i = 0; i < numOutputClasses; i++) {
			for (int j = 0; j < deepNet_burstDetection.numberOfHiddenUnits; j++) {
				outputWeights[i][j] = deepNet_burstDetection.getRandomWeight(deepNet_burstDetection.numberOfHiddenUnits, numOutputClasses);
			}
		}
	}

	public int errors(Vector<Vector<Double>> examples) {
		// need to incorporate
		int incorrect = 0;
		for (Vector<Double> example : examples) {
			feedforward(example);
			int prediction = 0;
			for (int i = 1; i < numOutputClasses; i++) {
				if (outputActivations[i] > outputActivations[prediction]) {
					prediction = i;
				}
			}
			if (prediction != example.lastElement()) {
				incorrect++;
			}
		}

		return incorrect;
	}

	public double error(Vector<Vector<Double>> examples) {
		double error = 0;
		for (Vector<Double> example : examples) {
			feedforward(example);
			for (int i = 0; i < numOutputClasses; i++) {
				if (example.lastElement() == i) {
					error += Math.pow(1 - outputActivations[i], 2);
				} else {
					error += Math.pow(-outputActivations[i], 2);
				}
			}
		}
		return Math.sqrt(error / examples.size());
	}

	public void train(Vector<Double> example) {
		feedforward(example);
		backprop(example);
		finishedTraining = true; //necessary boolean for dropout
	}

	public void feedforward(Vector<Double> example) {
		// A. PER EXAMPLE, with dopoutRate probability, set dropOut flags to true
		// 1. Reset previous example flags
		inputDropOutFlags =  new double[numFeatures];
		hiddenDropOutFlags = new double[deepNet_burstDetection.numberOfHiddenUnits];
		if(deepNet_burstDetection.inputDropoutRate <= 0.0){
			// effectively turns off dropOut
			Arrays.fill(inputDropOutFlags, 1.0);
		}
		if(deepNet_burstDetection.hiddenDropoutRate <= 0.0){
			// effectively turns off dropOut
			Arrays.fill(hiddenDropOutFlags, 1.0);
		}

		// 2. InputLayer flag updates
		for (int i = 0; i < numFeatures; i++) {
			if (Math.random() < .5) inputDropOutFlags[i] = 1.0;
		}
		// 3. HULayer flag updates
		for (int i = 0; i < deepNet_burstDetection.numberOfHiddenUnits; i++) {
			if (Math.random() < .5) hiddenDropOutFlags[i] = 1.0;
		}


		// B. Hidden Unit Activations
		for (int i = 0; i < deepNet_burstDetection.numberOfHiddenUnits; i++) {
			hiddenSums[i] = hiddenBiases[i];
			// drop out during training
			if(!finishedTraining){
				for (int j = 0; j < numFeatures; j++) {
					hiddenSums[i] += example.get(j) * hiddenWeights[i][j] * inputDropOutFlags[j];
				}
			}else{
				for (int j = 0; j < numFeatures; j++) {
					hiddenSums[i] += example.get(j) * hiddenWeights[i][j] * (1-deepNet_burstDetection.inputDropoutRate);
				}
			}
			hiddenActivations[i] = OneLayer.sigmoid(hiddenSums[i]);

		}

		// C. Output Activations
		for (int i = 0; i < numOutputClasses; i++) {
			outputSums[i] = -outputBiases[i];
			// drop out during training
			if(!finishedTraining){
				for (int j = 0; j < deepNet_burstDetection.numberOfHiddenUnits; j++) {
					outputSums[i] += hiddenActivations[j] * outputWeights[i][j] * hiddenDropOutFlags[j];
				}
			}else{
				for (int j = 0; j < deepNet_burstDetection.numberOfHiddenUnits; j++) {
					outputSums[i] += hiddenActivations[j] * outputWeights[i][j] * (1-deepNet_burstDetection.hiddenDropoutRate);//hiddenDropOutFlags[j];
				}
			}
			outputActivations[i] = OneLayer.sigmoid(outputSums[i]);

		}
	}

	public void backprop(Vector<Double> example) {

		// Calc and store output error in outputError vector
		calcOutputError(example);

		// Calc and store hidden unit error in hiddenLayerError vector - ***will likely change to 2D array with convolution network
		calcHiddenLayerError(example);

		// Update bias --> output weights
		updateOutputLayerInputWeights(example);

		// Update bias --> hidden weights
		updateHiddenLayerInputWeights(example); // again, will likely adapt this method to handle 2D array

	}

	private void calcOutputError(Vector<Double> example) {
		this.outputError = new double[numOutputClasses];
		for (int i = 0; i < numOutputClasses; i++) {
			if (example.lastElement() == i) {
				outputError[i] = outputActivations[i] - 1;
			} else {
				outputError[i] = outputActivations[i];
			}
			outputError[i] *= sigmoid_prime(outputSums[i]);
		}
	}

	private void calcHiddenLayerError(Vector<Double> example) {
		this.hiddenError = new double[deepNet_burstDetection.numberOfHiddenUnits];
		for (int i = 0; i < deepNet_burstDetection.numberOfHiddenUnits; i++) {
			hiddenError[i] = 0;
			for (int j = 0; j < numOutputClasses; j++) {
				hiddenError[i] += outputError[j] * outputWeights[j][i];
			}
			// CME comment -- pretty sure this is how you would account for dropout in backprop
			hiddenError[i] *= sigmoid_prime(hiddenSums[i]) * hiddenDropOutFlags[i];
		}
	}

	public void updateOutputLayerInputWeights(Vector<Double> example) {
		for (int i = 0; i < numOutputClasses; i++) {
			double biasUpdate = deepNet_burstDetection.eta * outputError[i];

			/*
            if (Lab2.WEIGHT_DECAY) {
                biasUpdate -= deepNet_burstDetection.LEARNING_RATE * Lab2.DECAY_RATE * outputBiases[i];
            }
			 */
			outputBiases[i] -= biasUpdate;

			/*
            if (Lab2.MOMENTUM) {
                outputBiases[i] -= outputBiasMomentums[i] * 0.9;
            }
			 */

			// Store previous bias --> output w_update for momentum term
			outputBiasMomentums[i] = biasUpdate;

			// Update h --> o weights
			for (int j = 0; j < deepNet_burstDetection.numberOfHiddenUnits; j++) {
				double weightUpdate = deepNet_burstDetection.eta * hiddenActivations[j] * outputError[i];
				/*
                if (Lab2.WEIGHT_DECAY) {
                    weightUpdate -= Lab2.LEARNING_RATE * Lab2.DECAY_RATE * outputWeights[i][j];
                }
				 */

				outputWeights[i][j] -= weightUpdate;
				/*
                if (Lab2.MOMENTUM) {
                    outputWeights[i][j] -= outputWeightMomentums[i][j] * 0.9;
                }
				 */
				// Store previous h --> o w_update for momentum term
				outputWeightMomentums[i][j] = weightUpdate;
			}
		}
	}

	private void updateHiddenLayerInputWeights(Vector<Double> example) {
		for (int i = 0; i < deepNet_burstDetection.numberOfHiddenUnits; i++) {
			double biasUpdate = deepNet_burstDetection.eta * hiddenError[i];
			/*
            if (Lab2.WEIGHT_DECAY) {
                biasUpdate -= Lab2.LEARNING_RATE * Lab2.DECAY_RATE * hiddenBiases[i];
            }
			 */
			hiddenBiases[i] -= biasUpdate;
			/*
            if (Lab2.MOMENTUM) {
                hiddenBiases[i] -= hiddenBiasMomentums[i] * 0.9;
            }
			 */
			hiddenBiasMomentums[i] = biasUpdate;

			// Update input --> hidden unit weights
			for (int j = 0; j < numFeatures; j++) {
				double weightUpdate = deepNet_burstDetection.eta * example.get(j) * hiddenError[i];
				/*
                if (Lab2.WEIGHT_DECAY) {
                    weightUpdate -= Lab2.LEARNING_RATE * Lab2.DECAY_RATE * hiddenWeights[i][j];
                }
				 */
				hiddenWeights[i][j] -= weightUpdate;
				/*
                if (Lab2.MOMENTUM) {
                    hiddenWeights[i][j] -= hiddenWeightMomentums[i][j] * 0.9;
                }
				 */
				hiddenWeightMomentums[i][j] = weightUpdate;
			}
		}
	}

	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public static double sigmoid_prime(double x) {
		return sigmoid(x) * (1 - sigmoid(x));
	}

	public static double relu_prime(double x) {
		return (x > 0) ? 1 : 0;
	}
}
