import java.io.File;
import java.util.*;

public class OneLayer {
	private final int numOutputClasses = 6;// just in case we want reduce output number for experimenting
	int numFeatures = Lab3.inputVectorSize - 1;
	double[][] hiddenWeights = new double[Lab3.numberOfHiddenUnits][numFeatures];
	double[][] outputWeights = new double[numOutputClasses][Lab3.numberOfHiddenUnits];
	double[][] hiddenWeightMomentums = new double[Lab3.numberOfHiddenUnits][numFeatures];
	double[][] outputWeightMomentums = new double[numOutputClasses][Lab3.numberOfHiddenUnits];
	double[] hiddenBiases = new double[Lab3.numberOfHiddenUnits];
	double[] outputBiases = new double[numOutputClasses];
	double[] hiddenBiasMomentums = new double[Lab3.numberOfHiddenUnits];
	double[] outputBiasMomentums = new double[numOutputClasses];

	double[] outputActivations = new double[numOutputClasses];
	double[] outputSums = new double[numOutputClasses];
	double[] hiddenActivations = new double[Lab3.numberOfHiddenUnits];
	double[] hiddenSums = new double[Lab3.numberOfHiddenUnits];

	private double[] outputError;
	private double[] hiddenError;

	private double[] hiddenDropOutFlags;
	private double[] inputDropOutFlags;

	private boolean finishedTraining = false;// boolean for enabling dropout during training

	public OneLayer() {
		for (int i = 0; i < Lab3.numberOfHiddenUnits; i++) {
			for (int j = 0; j < numFeatures; j++) {
				hiddenWeights[i][j] = Lab3.getRandomWeight(numFeatures, Lab3.numberOfHiddenUnits);
			}
		}

		for (int i = 0; i < numOutputClasses; i++) {
			for (int j = 0; j < Lab3.numberOfHiddenUnits; j++) {
				outputWeights[i][j] = Lab3.getRandomWeight(Lab3.numberOfHiddenUnits, numOutputClasses);
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
		hiddenDropOutFlags = new double[Lab3.numberOfHiddenUnits];
		if(Lab3.inputDropoutRate <= 0.0){
			// effectively turns off dropOut
			Arrays.fill(inputDropOutFlags, 1.0);
		}
		if(Lab3.hiddenDropoutRate <= 0.0){
			// effectively turns off dropOut
			Arrays.fill(hiddenDropOutFlags, 1.0); 
		}

		// 2. InputLayer flag updates
		for (int i = 0; i < numFeatures; i++) {
			if (Math.random() < .5) inputDropOutFlags[i] = 1.0;
		}
		// 3. HULayer flag updates
		for (int i = 0; i < Lab3.numberOfHiddenUnits; i++) {
			if (Math.random() < .5) hiddenDropOutFlags[i] = 1.0;
		}


		// B. Hidden Unit Activations
		for (int i = 0; i < Lab3.numberOfHiddenUnits; i++) {
			hiddenSums[i] = hiddenBiases[i];
			// drop out during training
			if(!finishedTraining){
				for (int j = 0; j < numFeatures; j++) {
					hiddenSums[i] += example.get(j) * hiddenWeights[i][j] * inputDropOutFlags[j];
				}
			}else{
				for (int j = 0; j < numFeatures; j++) {
					hiddenSums[i] += example.get(j) * hiddenWeights[i][j] * (1-Lab3.inputDropoutRate);
				}
			}
			hiddenActivations[i] = OneLayer.sigmoid(hiddenSums[i]);

		}

		// C. Output Activations
		for (int i = 0; i < numOutputClasses; i++) {
            outputSums[i] = -outputBiases[i];
			// drop out during training
			if(!finishedTraining){
	            for (int j = 0; j < Lab3.numberOfHiddenUnits; j++) {
	                outputSums[i] += hiddenActivations[j] * outputWeights[i][j] * hiddenDropOutFlags[j];
	            }
			}else{
	            for (int j = 0; j < Lab3.numberOfHiddenUnits; j++) {
	                outputSums[i] += hiddenActivations[j] * outputWeights[i][j] * (1-Lab3.hiddenDropoutRate);//hiddenDropOutFlags[j];
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
		this.hiddenError = new double[Lab3.numberOfHiddenUnits];
		for (int i = 0; i < Lab3.numberOfHiddenUnits; i++) {
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
			double biasUpdate = Lab3.eta * outputError[i];

			/*
            if (Lab2.WEIGHT_DECAY) {
                biasUpdate -= Lab3.LEARNING_RATE * Lab2.DECAY_RATE * outputBiases[i];
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
			for (int j = 0; j < Lab3.numberOfHiddenUnits; j++) {
				double weightUpdate = Lab3.eta * hiddenActivations[j] * outputError[i];
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
		for (int i = 0; i < Lab3.numberOfHiddenUnits; i++) {
			double biasUpdate = Lab3.eta * hiddenError[i];
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
				double weightUpdate = Lab3.eta * example.get(j) * hiddenError[i];
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
