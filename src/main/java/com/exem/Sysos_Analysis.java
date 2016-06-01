package com.exem;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;

/**
 * Created by khkim on 2016-06-01.
 */
public class Sysos_Analysis {

    private final static int FEATURES = 20;
    private static Logger log = LoggerFactory.getLogger(Sysos_Analysis.class);

    public static void main(String[] args) throws Exception {
        // 1. get the dataset
        RecordReader recordReader1 = new CSVRecordReader(1, ",");
        recordReader1.initialize(new FileSplit(new ClassPathResource("sysos.csv").getFile()));
        DataSetIterator inputIter = new RecordReaderDataSetIterator(recordReader1, 99999);

        RecordReader recordReader2 = new CSVRecordReader(1, ",");
        recordReader2.initialize(new FileSplit(new ClassPathResource("sysos_lable.csv").getFile()));
        DataSetIterator targetIter = new RecordReaderDataSetIterator(recordReader2, 99999);

        ArrayList inputList = new ArrayList();
        while (inputIter.hasNext()) {
            Iterator it = inputIter.next().iterator();
            while (it.hasNext()) {
                DataSet d = (DataSet) it.next();
                inputList.add(d.getFeatures());
            }
        }

        INDArray input = Nd4j.zeros(1, 1, inputList.size());

        for (int idx = 0; idx < inputList.size(); idx++) {
            input.putScalar(new int[]{0, 0, idx}, Double.parseDouble(inputList.get(idx).toString()));
        }
        System.out.println("input :: " + input);


        ArrayList targetList = new ArrayList();
        while (targetIter.hasNext()) {
            Iterator it = targetIter.next().iterator();
            while (it.hasNext()) {
                DataSet d = (DataSet) it.next();
                targetList.add(d.getFeatures());
            }
        }

        INDArray target = Nd4j.zeros(1, 1, targetList.size());

        for (int idx = 0; idx < targetList.size(); idx++) {
            target.putScalar(new int[]{0, 0, idx}, Double.parseDouble(targetList.get(idx).toString()));
        }
        System.out.println("target :: " + target);


        DataSet trainingData = new DataSet(input, target);
        System.out.println("TrainingDataSet");
        System.out.println(trainingData.toString());


        // 1-1. Create training data

        log.info("Build model....");
        // 2. setup the model configuration
        final int numInputs = 1;
        int outputNum = 100;
        int iterations = 1;
        long seed = 20160527;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(iterations)
                .seed(seed)
                .learningRate(0.01)
                .momentum(0.5)
                .regularization(true).l2(0.01)
                .list(2)
                .layer(0, new GravesLSTM.Builder().nIn(numInputs).nOut(outputNum)
                        .updater(Updater.RMSPROP)
                        .activation("tanh")
//                .weightInit(WeightInit.UNIFORM)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT).activation("identity")
                        .updater(Updater.RMSPROP)
                        .nIn(outputNum).nOut(1)
//                .weightInit(WeightInit.UNIFORM)
                        .build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTBackwardLength(FEATURES)
                .tBPTTForwardLength(FEATURES)
                .pretrain(false)
                .backprop(true)
                .build();

        // 3. instantiate the configured network
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));


        // 4. Prediction



        for (int epoch = 0; epoch < 200; epoch++) {

            System.out.println("Epoch : " + epoch);

            // 4-1. train the data
            net.fit(trainingData);
            // 4-2. clear current stance from the last example
            net.rnnClearPreviousState();
            // 4-3. put the first number into the rnn as initialization
            INDArray testInit = Nd4j.zeros(1);
            testInit.putScalar(new int[]{0, 0}, 2.507);

            INDArray output = net.rnnTimeStep(testInit);

            System.out.println("outputvalue : " + output);

            for (int j = 0; j < 28000; j++) {
                System.out.print(Math.round(output.getFloat(0)*100f)/100f + ",");

                INDArray nextInput = Nd4j.zeros(1);
                nextInput = nextInput.putScalar(new int[]{0, 0}, output.getFloat(0));
                output = net.rnnTimeStep(nextInput);
            }

            System.out.println("\n");

        }
    }


}
