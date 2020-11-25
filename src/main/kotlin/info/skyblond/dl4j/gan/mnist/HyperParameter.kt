package info.skyblond.dl4j.gan.mnist

import org.nd4j.linalg.learning.config.Adam

/**
 * A class describing hyper parameters.
 * */
data class HyperParameter(
    /** Label amplitude */
    val labelAmp: Double,
    /** Learning rate of updater */
    val learningRate: Double,
    /** Beta1 of [Adam] updater */
    val adamBeta1: Double,
    /** seed */
    val seed: Long
)