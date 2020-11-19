package info.skyblond.dl4j.gan.mnist

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.graph.StackVertex
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Sgd
import org.nd4j.linalg.lossfunctions.LossFunctions


class GANConfiguration(val learningRate: Double, val noiseDimension: Int) {

    val computationGraphConfiguration: ComputationGraphConfiguration
        get() = NeuralNetConfiguration.Builder()
            .updater(Sgd(learningRate))
            .weightInit(WeightInit.XAVIER)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .graphBuilder()
            .addInputs("noiseInput", "picInput", "labelInput")
            // Generator
            .addVertex("G-Input", MergeVertex(), "labelInput", "noiseInput")
            .addLayer(
                "G-L1",
                DenseLayer.Builder()
                    .nIn(noiseDimension + Companion.conditionDimension)
                    .nOut(128)
                    .activation(Activation.LEAKYRELU)
                    .build(),
                "G-Input"
            )
            .addLayer(
                "G-L2",
                DenseLayer.Builder()
                    .nIn(128)
                    .nOut(512)
                    .activation(Activation.LEAKYRELU)
                    .build(),
                "G-L1"
            )
            .addLayer(
                "G-Output",
                DenseLayer.Builder()
                    .nIn(512)
                    .nOut(Companion.picHeight * Companion.picWidth)
                    .activation(Activation.SIGMOID)
                    .build(),
                "G-L2"
            )

            // Discriminator
            .addVertex("D-InputFromG", MergeVertex(), "labelInput", "G-Output")
            .addVertex("D-InputFromTruth", MergeVertex(), "labelInput", "picInput")
            // combined ground real pic and generated fake pic
            .addVertex("D-Input", StackVertex(), "D-InputFromTruth", "D-InputFromG")
            .addLayer(
                "D-L1",
                DenseLayer.Builder()
                    .nIn(Companion.picHeight * Companion.picWidth + Companion.conditionDimension)
                    .nOut(512)
                    .activation(Activation.LEAKYRELU)
                    .build(),
                "D-Input"
            )
            .addLayer(
                "D-L2",
                DenseLayer.Builder()
                    .nIn(512)
                    .nOut(128)
                    .activation(Activation.LEAKYRELU)
                    .build(),
                "D-L1"
            )
            .addLayer(
                "D-Output",
                OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                    .nIn(128)
                    .nOut(1)
                    .activation(Activation.SIGMOID)
                    .build(),
                "D-L2"
            )
            .setOutputs("D-Output")
            .build()

    companion object {
        const val conditionDimension = 10
        const val picHeight = 28
        const val picWidth = 28
        const val generatorOutputLayerName = "G-Output"
        val generatorLayersName = listOf(
            "G-L1", "G-L2", "G-Output"
        )
        val discriminatorLayersName = listOf(
            "D-L1", "D-L2", "D-Output"
        )
    }
}