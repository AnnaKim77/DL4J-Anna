package examples.DBN;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;


/**
 * Using Deep Belief Algorithm (Upgrade version of Restricted Boltzman Machine)
 * DBN : can serve as multinomial classifiers.
 * Created by khkim on 2016-06-14.
 */
public class DBNIrisExample {

    private static Logger log = LoggerFactory.getLogger(DBNIrisExample.class);

    public static void main(String[] args) throws Exception {

        final int numRows = 4;
        final int numColumns = 1;
        int outputNum = 3;
        int numSamples = 150;
        int batchSize = 150;
        int iterations = 5;
        int splitTrainNum = (int) (batchSize * .8);
        int seed = 123;

        log.info("Load data....");
        BaseDatasetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
        DataSet next = iter.next();
        next.shuffle();
        next.normalizeZeroMeanZeroUnitVariance();

        log.info("Split data....");
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .learningRate(1e-6f)
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .l1(1e-1).regularization(true).l2(2e-4)
                .useDropConnect(true)
                .list() //
                .layer(0, new RBM.Builder(RBM.HiddenUnit.RECTIFIED, RBM.VisibleUnit.GAUSSIAN)
                        //적용된 Visible, Hidden unit의 조합이 가장 우수한것으로 알려짐
                        .nIn(numRows * numColumns)
                        .nOut(3)
                        .weightInit(WeightInit.XAVIER)
                        .k(1)
                        .activation("relu")
                        .lossFunction(LossFunctions.LossFunction.RMSE_XENT)
                        .updater(Updater.ADAGRAD)
                        .dropOut(0.5)
                        .build()
                )
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(3)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build()
                )
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new HistogramIterationListener(1));

        log.info("Train model....");
        model.fit(train);

        log.info("Evaluate weights....");
        for (org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation();
        INDArray output = model.output(test.getFeatureMatrix());


        for (int i = 0; i < output.rows(); i++) {
            String actual = train.getLabels().getRow(i).toString().trim();
            String predicted = output.getRow(i).toString().trim();
            log.info("actual " + actual + " vs predicted " + predicted);
        }
        eval.eval(test.getLabels(), output);
        log.info(eval.stats());


        log.info("****************Example finished********************");


    }

}
