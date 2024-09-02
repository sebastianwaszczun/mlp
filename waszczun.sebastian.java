import java.util.Random;

public class Main {
    public static void main(String[] args) {
        int numNeurons = 5;
        double learningRate = 0.01;
        int maxEpochs = 1000;

        DataSetGenerator dataSetGenerator = new DataSetGenerator();
        double[][] trainingData = dataSetGenerator.generateDataSet(1000);
        double[][] X_train = new double[trainingData.length][2];
        double[][] y_train = new double[trainingData.length][1];

        for (int i = 0; i < trainingData.length; i++) {
            X_train[i][0] = trainingData[i][0];
            X_train[i][1] = trainingData[i][1];
            y_train[i][0] = trainingData[i][2];
        }

        MLP model = new MLP(numNeurons, learningRate, maxEpochs);

        model.fit(X_train, y_train);

        testModel(model, dataSetGenerator);

        testMaxEpochs(X_train, y_train);

        testNumNeurons(X_train, y_train);

        testLearningRate(X_train, y_train);
    }

    static void testModel(MLP model, DataSetGenerator dataSetGenerator) {
        double[][] X_test = dataSetGenerator.generateDataSet(10000);

        double sumSquaredError = 0.0;

        for (int i = 0; i < X_test.length; i++) {
            double[] predictedOutput = model.predict(X_test[i]);
            double trueOutput = X_test[i][2];
            sumSquaredError += Math.pow(predictedOutput[0] - trueOutput, 2);
        }

        double meanSquaredError = sumSquaredError / X_test.length;
        System.out.println("Średni błąd kwadratowy na zbiorze testowym: " + meanSquaredError);
    }

    static void testMaxEpochs(double[][] X_train, double[][] y_train) {
        int[] maxEpochsArray = {100, 500, 1000, 2000};

        for (int maxEpochs : maxEpochsArray) {
            MLP model = new MLP(5, 0.01, maxEpochs);
            model.fit(X_train, y_train);
            System.out.println("Maksymalna liczba kroków uczenia: " + maxEpochs);
            testModel(model, new DataSetGenerator());
            System.out.println("--------------------");
        }
    }

    static void testNumNeurons(double[][] X_train, double[][] y_train) {
        int[] numNeuronsArray = {2, 5, 10, 20, 50, 100};

        for (int numNeurons : numNeuronsArray) {
            MLP model = new MLP(numNeurons, 0.01, 1000);
            model.fit(X_train, y_train);
            System.out.println("Liczba neuronów w warstwie ukrytej: " + numNeurons);
            testModel(model, new DataSetGenerator());
            System.out.println("--------------------");
        }
    }

    static void testLearningRate(double[][] X_train, double[][] y_train) {
        double[] learningRateArray = {0.001, 0.01, 0.1};

        for (double learningRate : learningRateArray) {
            MLP model = new MLP(5, learningRate, 1000);
            model.fit(X_train, y_train);
            System.out.println("Współczynnik uczenia: " + learningRate);
            testModel(model, new DataSetGenerator());
            System.out.println("--------------------");
        }
    }
}

class DataSetGenerator {
    private static final double PI = Math.PI;

    public double[][] generateDataSet(int m) {
        double[][] dataSet = new double[m][3];
        Random random = new Random();

        for (int i = 0; i < m; i++) {
            double x1 = random.nextDouble() * PI;
            double x2 = random.nextDouble() * PI;
            double y = Math.cos(x1 * x2) * Math.cos(2 * x1);
            dataSet[i][0] = x1;
            dataSet[i][1] = x2;
            dataSet[i][2] = y;
        }

        return dataSet;
    }
}

class MLP {
    private int numNeurons;
    private double learningRate;
    private int maxEpochs;
    private double[][] V;
    private double[] W;

    public MLP(int numNeurons, double learningRate, int maxEpochs) {
        this.numNeurons = numNeurons;
        this.learningRate = learningRate;
        this.maxEpochs = maxEpochs;

        Random random = new Random();
        V = new double[2][numNeurons];
        W = new double[numNeurons + 1];

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < numNeurons; j++) {
                V[i][j] = random.nextGaussian() * 0.1;
            }
        }

        for (int i = 0; i < numNeurons + 1; i++) {
            W[i] = random.nextGaussian() * 0.1;
        }
    }

    public void fit(double[][] X_train, double[][] y_train) {
        for (int epoch = 0; epoch < maxEpochs; epoch++) {
            for (int i = 0; i < X_train.length; i++) {
                double[] hiddenLayerOutput = _forward(X_train[i]);
                double output = hiddenLayerOutput[numNeurons];
                double[] errors = new double[numNeurons];

                for (int j = 0; j < numNeurons; j++) {
                    errors[j] = y_train[i][0] - output;
                }

                for (int j = 0; j < numNeurons; j++) {
                    for (int k = 0; k < 2; k++) {
                        V[k][j] += learningRate * errors[j] * W[j] * hiddenLayerOutput[j] * (1 - hiddenLayerOutput[j]) * X_train[i][k];
                    }
                    W[j] += learningRate * errors[j] * hiddenLayerOutput[j];
                }
            }
        }
    }

    private double[] _forward(double[] input) {
        double[] hiddenLayerOutput = new double[numNeurons + 1];

        for (int i = 0; i < numNeurons; i++) {
            double s = V[0][i] * input[0] + V[1][i] * input[1];
            hiddenLayerOutput[i] = sigmoid(s);
        }

        hiddenLayerOutput[numNeurons] = 1.0;

        double output = 0.0;
        for (int i = 0; i < numNeurons + 1; i++) {
            output += W[i] * hiddenLayerOutput[i];
        }

        hiddenLayerOutput[numNeurons] = output;
        return hiddenLayerOutput;
    }

    private double sigmoid(double s) {
        return 1.0 / (1.0 + Math.exp(-s));
    }

    public double[][] getHiddenLayerWeights() {
        return V;
    }

    public double[] getOutputLayerWeights() {
        return W;
    }

    public double[] predict(double[] input) {
        return _forward(input);
    }
}
