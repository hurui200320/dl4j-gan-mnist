package info.skyblond.dl4j.gan.mnist

import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.schedule.ISchedule

/**
 * A class describing hyper parameters.
 * */
data class HyperParameter(
    /** What the minimum confidence has D to give for real pic in training */
    val realConfidence: Double,
    /** What the maximum confidence has D to give for fake pic in training*/
    val fakeIdentitySchedule: ISchedule,
    /** Learning rate of updater */
    val learningRate: Double,
    /** Beta1 of [Adam] updater */
    val adamBeta1: Double,
    /** seed */
    val seed: Long
)