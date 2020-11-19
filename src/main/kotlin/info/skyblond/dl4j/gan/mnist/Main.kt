package info.skyblond.dl4j.gan.mnist

import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution
import org.slf4j.LoggerFactory
import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors

// TODO 避免盲目查找带来的运算过大，首先在Amp=5的时候寻找最佳的trainD值(Amp=1时没有可观的效果)
//      然后在该D值下查找最佳Amp值，找到后再以新的Amp值查找最佳trainD，如此往复直到二者变化不大为之
// TODO 训练时先训练D直到判别正确率大于0.5，再训练G使得D的正确率小于0.5，循环往复。问题在于如何在训练的同时判断获得其正确率？
fun main() {
    val logger = LoggerFactory.getLogger("Application")

    // Need proxy to download pytorch engine library
    System.setProperty("http.proxyHost", "localhost")
    System.setProperty("https.proxyHost", "localhost")
    System.setProperty("http.proxyPort", "1081")
    System.setProperty("https.proxyPort", "1081")

    // train models at same time, GPU memory shares across models(MNIST dataset)
    val nThreads = 10

    val labelAmplifierList = listOf(5.0, 7.0, 9.0)

    val trainDTimeList = listOf(7, 9, 11, 13, 15)

    val seed = 1189998819991197253L
    val batchSize = 32
    val iterations = 10_000
    val executor = Executors.newFixedThreadPool(nThreads)

    val latch = CountDownLatch(trainDTimeList.size * labelAmplifierList.size)

    // loop around
    for (currentTrainDTime in trainDTimeList) {
        for (currentLabelAmplifier in labelAmplifierList) {
            // submit threads
            executor.submit {
                val ganConfiguration = GANConfiguration(0.1, 100)
                val noiseDistribution = UniformDistribution(0.0, 1.0)
                val trainer = GANTrainer(currentLabelAmplifier, currentTrainDTime)
                trainer.train(batchSize, iterations, seed, noiseDistribution, ganConfiguration)
                latch.countDown()
            }
            logger.info("Current done: labelAmplifier $currentLabelAmplifier trainDTime $currentTrainDTime")
        }
    }

    logger.info("Submit thread finished. Waiting for those threads...")

    latch.await()

    executor.shutdownNow()
    logger.info("All thread finished. Done!")
}
