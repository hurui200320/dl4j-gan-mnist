package info.skyblond.dl4j.gan.mnist

import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.activations.impl.ActivationLReLU
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

fun getModel(hyperParameter: HyperParameter): CGAN {
    return CGAN.Companion.Builder()
        .apply {
            generatorUpdater = Adam(
                hyperParameter.learningRate,
                hyperParameter.adamBeta1,
                Adam.DEFAULT_ADAM_BETA2_VAR_DECAY,
                Adam.DEFAULT_ADAM_EPSILON
            )
            discriminatorUpdater = Adam(
                hyperParameter.learningRate,
                hyperParameter.adamBeta1,
                Adam.DEFAULT_ADAM_BETA2_VAR_DECAY,
                Adam.DEFAULT_ADAM_EPSILON
            )

//            generatorGradientNormalization = GradientNormalization.RenormalizeL2PerParamType
//            generatorNormalizationThreshold = 50.0
//            discriminatorGradientNormalization = GradientNormalization.RenormalizeL2PerParamType
//            discriminatorNormalizationThreshold = 50.0

            this.seed = hyperParameter.seed
        }
        .apply {
            addGeneratorVertex(
                "Input",
                MergeVertex(),
                CGAN.GENERATOR_NOISE_INPUT_NAME,
                CGAN.GENERATOR_LABEL_INPUT_NAME
            )
            addGeneratorLayer(
                "L1",
                DenseLayer.Builder()
                    .nIn(noiseDim + labelDim)
                    .nOut(256)
                    .activation(ActivationLReLU(0.2))
                    .build(),
                "Input"
            )
            addGeneratorLayer(
                "L2",
                DenseLayer.Builder()
                    .nIn(256)
                    .nOut(512)
                    .activation(ActivationLReLU(0.2))
                    .build(),
                "L1"
            )
            addGeneratorLayer(
                "L3",
                DenseLayer.Builder()
                    .nIn(512)
                    .nOut(1024)
                    .activation(ActivationLReLU(0.2))
                    .build(),
                "L2"
            )
            addGeneratorLayer(
                CGAN.GENERATOR_OUTPUT_NAME,
                DenseLayer.Builder()
                    .nIn(1024)
                    .nOut(picHeight * picWidth)
                    .activation(Activation.SIGMOID)
                    .build(),
                "L3"
            )
        }
        .apply {
            addDiscriminatorVertex(
                "MergedInput",
                MergeVertex(),
                CGAN.DISCRIMINATOR_PIC_INPUT_NAME,
                CGAN.DISCRIMINATOR_LABEL_INPUT_NAME
            )
            addDiscriminatorLayer(
                "L1",
                DenseLayer.Builder()
                    .nIn(picHeight * picWidth + labelDim)
                    .nOut(1024)
                    .dropOut(0.3)
                    .activation(ActivationLReLU(0.2))
                    .build(),
                "MergedInput"
            )
            addDiscriminatorLayer(
                "L2",
                DenseLayer.Builder()
                    .nIn(1024)
                    .nOut(512)
                    .dropOut(0.3)
                    .activation(ActivationLReLU(0.2))
                    .build(),
                "L1"
            )
            addDiscriminatorLayer(
                "L3",
                DenseLayer.Builder()
                    .nIn(512)
                    .nOut(256)
                    .dropOut(0.3)
                    .activation(ActivationLReLU(0.2))
                    .build(),
                "L2"
            )
            addDiscriminatorLayer(
                CGAN.DISCRIMINATOR_OUTPUT_NAME,
                OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                    .nIn(256)
                    .nOut(1)
                    .activation(Activation.SIGMOID)
                    .build(),
                "L3"
            )
        }
        .build()
}