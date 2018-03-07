/*
 * TODO:
 * Verify that the max pooling backprop works
 * Verify that the weight and bias updates work for ConvolutionLayer
 * Create the nextErrors matrix in the ConvolutionLayer's backprop method (until this is done, only the first layer can be convolutional)
 * (I think this is done with a full convolution, so that the nextErrors matrix has the same dimensions as the input matrix)
 * Figure out how to initialize the weights in the ConvolutionLayer
 * Add dropout
 * Test it
 */

import java.util.ArrayList;
import java.util.Arrays;

public class DeepNet {
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
	static boolean doneTraining;

	public DeepNet(boolean dropOut, boolean normalizeKernelOutputByKernelSum) {
		// Create CNN architecture
		// ConvolutionLayer must always be followed by a pool layer
		
		// Less readable but may alter code later to handle variable poolSize
//		int poolSize = 4;
//		int filterWindow = 5;
//		int numFilters = 8;
//		int numFiltersSecConv = 16;
//
//		int imageSize = 32;
//		// First boolean argument added to signify that layer should dropout (if dropout is on) input to layer
//		this.layers.add(new ConvolutionLayer(imageSize, Lab3.unitsPerPixel, numFilters, filterWindow, dropOut,normalizeKernelOutputByKernelSum));
//		this.layers.add(new PoolLayer((imageSize-filterWindow+1), numFilters, poolSize, dropOut));// ***may want to test out different numFilters
//		this.layers.add(new ConvolutionLayer((imageSize-filterWindow+1)/poolSize, numFilters, numFiltersSecConv, filterWindow,dropOut,normalizeKernelOutputByKernelSum));
//		this.layers.add(new PoolLayer(((imageSize-filterWindow+1)/poolSize)-filterWindow+1, numFiltersSecConv,poolSize,dropOut));
//		this.layers.add(new FullyConnectedLayer(5, 5, 16, 128,dropOut,layers.get(3)));// **may want to test out varying #nodes in fully connected layer
//		this.layers.add(new FullyConnectedLayer(128, 1, 1, 6,dropOut,layers.get(4)));
//		this.doneTraining = false; // necessary for implementing dropout
		
		// First boolean argument added to signify that layer should dropout (if dropout is on) input to layer
		this.layers.add(new ConvolutionLayer(32, Lab3.unitsPerPixel, 8, 5,dropOut,normalizeKernelOutputByKernelSum));
		this.layers.add(new PoolLayer(28, 8,2,dropOut));// ***may want to test out different numFilters
		this.layers.add(new ConvolutionLayer(14, 8, 16, 5,dropOut,normalizeKernelOutputByKernelSum));
		this.layers.add(new PoolLayer(10, 16,2,dropOut));
		this.layers.add(new FullyConnectedLayer(5, 5, 16, 128,dropOut,layers.get(3)));// **may want to test out varying #nodes in fully connected layer
		this.layers.add(new FullyConnectedLayer(128, 1, 1, 6,dropOut,layers.get(4)));
		this.doneTraining = false; // necessary for implementing dropout


//		this.layers.add(new ConvolutionLayer(32, 4, 20, 5));
//		this.layers.add(new PoolLayer(28, 20));
//		this.layers.add(new FullyConnectedLayer(14, 14, 20, 6));

		// Same thing as OneLayer.java:
//		this.layers.add(new FullyConnectedLayer(32, 32, 4, 50));
//		this.layers.add(new FullyConnectedLayer(50, 1, 1, 6));
	}

	public void feedforward(double[][][] image) {
//        System.out.println(image.length+" "+image[1].length+" "+image[0][1].length);
		Layer layer = layers.get(0);
		layer.feedforward(image);
		for (int i = 1; i < layers.size(); i++) {
			Layer nextLayer = layers.get(i);
			nextLayer.feedforward(layer.getOutputs());
			if (nextLayer instanceof PoolLayer) {
				((PoolLayer) nextLayer).setSums(layer.getSums());
			}
			layer = nextLayer;
		}
	}

	public void train(double[][][] image, int label) {
		/*
		 * Lab3.java is written so that we train using flattened 1D feature vectors. This made it simple to
		 * connect the code to OneLayer.java, but is more difficult to reason about in a convolutional net.
		 * So for now this method assumes that we're given a 3D array and a label instead of Vector<Double>
		 */

		this.feedforward(image);
		
		// Calculate error
		// 6x1x1 array holding the activations of the last layer
		Layer lastLayer = this.layers.get(this.layers.size() - 1);
		double[][][] outputs = lastLayer.getOutputs();
		double[][][] errors = new double[6][1][1];
		for (int i = 0; i < 6; i++) {
			double expected = (label == i) ? 1 : 0;
			errors[i][0][0] = outputs[i][0][0] - expected;
		}

		
		// Backpropagate error, starting with output layer
		for (int i = layers.size() - 1; i >= 0; i--) {
			Layer layer = layers.get(i);
			if (i > 0) {
				Layer nextLayer = layers.get(i - 1);
				errors = layer.backprop(errors, nextLayer.getSums());
			} else {
				errors = layer.backprop(errors, null);
			}
		}
	}

	public int getLabel(double[][][] image) {
		this.doneTraining = true;
		this.feedforward(image);

		Layer lastLayer = this.layers.get(this.layers.size() - 1);

		// Calculate error
		// 6x1x1 array holding the activations of the last layer
		double[][][] outputs = lastLayer.getOutputs();
		int maxIndex = 0;
		for (int i = 1; i < 6; i++) {
			if (outputs[i][0][0] > outputs[maxIndex][0][0]) {
				maxIndex = i;
			}
		}

		return maxIndex;
	}
}

interface Layer {
	public void feedforward(double[][][] input);
	public double[][][] backprop(double[][][] error, double[][][] sums);
	public double[][][] getSums();
	public double[][][] getOutputs();
	public double[][][] getDropOutFlags();
}

class ConvolutionLayer implements Layer {
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
						this.weights[x][y][z][filter] = Lab3.getRandomWeight(inputSize * inputSize * inputDepth, outputSize * outputSize * numFilters);
					}
				}
			}
		}
	}

	// Expected input order: width x height x depth
	// Walks through the output array, calculating the value that should go into each position
	public void feedforward(double[][][] inputs) {
		// update dropout flags for each example 
		if(dropOutLayer){
			this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
			if(Lab3.inputDropoutRate <= 0.0 || DeepNet.doneTraining){
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
							if (Math.random() > Lab3.inputDropoutRate) inputDropOutFlags[i][j][k] = 1.0;
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
	}

	public double[][][] backprop(double[][][] errors, double[][][] sums) {
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
//									if(this.dropOutLayer == true){
//										nextErrors[x + filterX][y + filterY][filterZ] += this.weights[filterX][filterY][filterZ][filter] * error * dropOutFlags[x + filterX][y + filterY][filterZ];
//									}else{
									nextErrors[x + filterX][y + filterY][filterZ] += this.weights[filterX][filterY][filterZ][filter] * error;
//									}
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
					this.biases[filter] -= Lab3.eta * error * (1.0 / (this.outputSize * this.outputSize)); // Update bias
					this.updateWeights(error * (1.0 / (this.filterSize * this.filterSize)), x, y, filter);
				}
			}
		}

		return nextErrors;
	}

	public void updateWeights(double error, int startX, int startY, int filter) {
		for (int x = 0; x < this.filterSize; x++) {
			for (int y = 0; y < this.filterSize; y++) {
				for (int z = 0; z < this.inputDepth; z++) {
					double input = this.inputs[startX + x][startY + y][z];
					this.weights[x][y][z][filter] -= Lab3.eta * input * error;
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
						sum += inputs[startX + x][startY + y][z] * this.weights[x][y][z][filter] * (1-Lab3.inputDropoutRate);
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
}

class PoolLayer implements Layer {
	double[][][] sums;
	double[][][] outputs;
	double[][][] dropOutFlags; // decided to not implement for pooling layers

	int inputSize, inputDepth, poolSize, outputSize;
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

	// Walks through the output array, calculating the value that should go into each position
	public void feedforward(double[][][] inputs) {
		// update dropout flags for each example 
		if(dropOutLayer){
			this.inputDropOutFlags =  new double[inputSize][inputSize][inputDepth];
			if(Lab3.inputDropoutRate <= 0.0 || DeepNet.doneTraining){
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
							if (Math.random() > Lab3.inputDropoutRate) inputDropOutFlags[i][j][k] = 1.0;
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
	}

	public double[][][] backprop(double[][][] errors, double[][][] sums) {
		// Returns a sparse matrix that copies the errors to each source
		// For example [[ 1, 2 ], [ 3, 4 ]] would give us [4] in the forward pass, and a sources matrix that looks like:
		// [[ 0, 0 ], [ 0, 1 ]]. If the errors matrix is [d], this method would then return [[ 0, 0 ], [ 0, d ]]
		double [][][] dropOutFlags = this.getDropOutFlags();
		
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
}

class FullyConnectedLayer implements Layer {
	int inputWidth, inputHeight, inputDepth, numOutputs;
	double[][][][] weights;
	double[] biases;
	double[][][] sums;
	double[][][] inputs;
	double[][][] outputs;
	
	// drop out vars
	boolean dropOutLayer;
	double[][][] fullyConnectedLayerDropOutFlags;
	public Layer prevLayer;


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
						this.weights[x][y][z][output] = Lab3.getRandomWeight(inputWidth * inputHeight * inputDepth, numOutputs);
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
			if(Lab3.hiddenDropoutRate <= 0.0 || DeepNet.doneTraining){
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
							if (Math.random() > Lab3.hiddenDropoutRate) fullyConnectedLayerDropOutFlags[i][j][k] = 1.0;
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
							this.sums[output][0][0] += inputs[x][y][z] * this.weights[x][y][z][output] * (1 - Lab3.hiddenDropoutRate);
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
			this.biases[output] -= Lab3.eta * errors[output][0][0];
		}

		// Perform weight updates
		for (int x = 0; x < this.inputWidth; x++) {
			for (int y = 0; y < this.inputHeight; y++) {
				for (int z = 0; z < this.inputDepth; z++) {
					for (int output = 0; output < this.numOutputs; output++) {
						this.weights[x][y][z][output] -= Lab3.eta * this.inputs[x][y][z] * errors[output][0][0];
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
	
	public double[][][] getDropOutFlags() {
		return fullyConnectedLayerDropOutFlags;
	}
}