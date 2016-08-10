package examples;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import static javafx.scene.input.KeyCode.M;
import static javafx.scene.input.KeyCode.O;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MCXENT;
import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MSE;

/**
 * Layer check를 위해서 임의로 만든 class
 *
 * Created by khkim on 2016-06-13.
 */
public class checkLayers {


    public static void main(String[] args) {
        int iterations= 1;
        int seed =1 ;
        int numInputs=1;
        int outputNum=1;
        int tbpttLength=1;


        MultiLayerConfiguration.Builder builder
//                .layer(3, new GravesLSTM.Builder(MCXENT).forgetGateBiasInit(0.1).activation("relu"))

                ;
        builder = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipL2PerLayer)
                .iterations(iterations)
                .seed(seed)
                .learningRate(0.2)
                .momentum(0.5)
                .regularization(true).l2(0.001).dropOut(0.5)

                //check below


                .list(2)
                .layer(0, new GravesLSTM.Builder().nIn(numInputs).nOut(outputNum)
                        .updater(Updater.NESTEROVS)
                        .activation("tanh")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new ConvolutionLayer.Builder().stride(1)

                        .build())
                .layer(2, new RnnOutputLayer.Builder(MSE).activation("identity")
                        .updater(Updater.NESTEROVS)
                        .nIn(outputNum).nOut(1)
                        .weightInit(WeightInit.XAVIER)
                        .build());


        // 3. INSTANTIATE THE CONFIGURED NETWORK
        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();


}






}
