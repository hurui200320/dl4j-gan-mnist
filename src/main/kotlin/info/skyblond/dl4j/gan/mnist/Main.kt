package info.skyblond.dl4j.gan.mnist

import org.deeplearning4j.core.storage.StatsStorage
import org.deeplearning4j.datasets.fetchers.MnistDataFetcher
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.ui.VertxUIServer
import org.deeplearning4j.ui.model.stats.StatsListener
import org.deeplearning4j.ui.model.storage.FileStatsStorage
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.rng.distribution.impl.UniformDistribution
import org.nd4j.linalg.dataset.MultiDataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.schedule.ExponentialSchedule
import org.nd4j.linalg.schedule.ScheduleType
import org.slf4j.LoggerFactory
import java.awt.image.BufferedImage
import java.io.File
import java.nio.charset.StandardCharsets
import javax.imageio.ImageIO
import kotlin.math.abs
import kotlin.random.Random

// Global parameter
const val noiseDim = 100
const val labelDim = 10
const val picHeight = 28
const val picWidth = 28

fun main() {
    val logger = LoggerFactory.getLogger("Application")

    val hyperParameter = HyperParameter(
        realConfidence = 0.9,
        fakeIdentitySchedule = ExponentialSchedule(ScheduleType.ITERATION, 0.6, 0.9965400695800781),
        learningRate = 0.0002,
        adamBeta1 = 0.5,
        seed = System.currentTimeMillis() // 1189998819991197253L
    )

    // fixed parameter
    val batchSize = 32
    val iterations = 5000
    val noiseDistribution = UniformDistribution(0.0, 1.0)

    val cgan = getModel(hyperParameter)

    val workingFolder = File("./trainingProcess").prepareFolder(true)
    File(workingFolder, "info.json").clear().writeText(toBeautifiedJson(hyperParameter), StandardCharsets.UTF_8)

    val ganStatsStorage: StatsStorage = FileStatsStorage(File(workingFolder, "gan.stats").clear())
    val ganStatsListener = StatsListener(ganStatsStorage)
    ganStatsListener.sessionID = "GAN"

    val discriminatorStatsStorage: StatsStorage = FileStatsStorage(File(workingFolder, "discriminator.stats").clear())
    val discriminatorStatsListener = StatsListener(discriminatorStatsStorage)
    discriminatorStatsListener.sessionID = "Discriminator"

    val uiServer = VertxUIServer.getInstance(9000, true, null)//) { statsStorageProvider[it] }
    uiServer.attach(ganStatsStorage)
    uiServer.attach(discriminatorStatsStorage)

    cgan.gan.setListeners(ganStatsListener)
    cgan.discriminator.setListeners(discriminatorStatsListener)

    val rawMnistDataSet = MnistDataSetIterator(
        batchSize,
        MnistDataFetcher.NUM_EXAMPLES + MnistDataFetcher.NUM_EXAMPLES_TEST,
        true, true, true, hyperParameter.seed
    )

    // training loop
    for (i in 0 until iterations) {
        var loopCounter = 0
        val fakeIdentity = hyperParameter.fakeIdentitySchedule.valueAt(i, -1)
        logger.info("Current fakeIdentity at iter. $i: $fakeIdentity, G's goal: ${hyperParameter.realConfidence - fakeIdentity}")

        cgan.updateParameters()


        // TODO reset epoch count as training loop iteration number, then use epoch to schedule lr
//        cgan.gan.configuration.epochCount = i

        // train D until D can distinguish this batch
        var realPicScoreSum = 0.0
        var fakePicScoreSum = 0.0
        var picCount = 0
        var lastFakePicAvg = 2.0
        while (true) {
            // sample real pic
            if (!rawMnistDataSet.hasNext()) {
                rawMnistDataSet.reset()
            }
            val dataset = rawMnistDataSet.next()
            val realPic: INDArray = dataset.features
            val label: INDArray = dataset.labels
            val noise = Nd4j.rand(
                noiseDistribution,
                batchSize.toLong(),
                noiseDim.toLong()
            )

            // sample G
            val fakePic = cgan.generator.feedForward(arrayOf(noise, label), false)[CGAN.GENERATOR_OUTPUT_NAME]
                ?: error("No layer named ${CGAN.GENERATOR_OUTPUT_NAME}")

            // make up the dataset for D
            val featurePicD = Nd4j.vstack(realPic, fakePic)
            val featureLabelD = Nd4j.vstack(label, label)

            // When training D, real pic inputs stack on top of fake pic(G output)
            // so first batchSize is 1(i.e. real pic), then followed by batchSize 0(i.e. fake pic)
            val labelD = Nd4j.vstack(Nd4j.ones(batchSize, 1), Nd4j.zeros(batchSize, 1))

            val dataSetD = MultiDataSet(arrayOf(featurePicD, featureLabelD), arrayOf(labelD))

            cgan.discriminator.fit(dataSetD)

            val output: Map<String, INDArray> =
                cgan.discriminator.feedForward(arrayOf(featurePicD, featureLabelD), false)
            val infer =
                output[CGAN.DISCRIMINATOR_OUTPUT_NAME] ?: error("No layer named ${CGAN.DISCRIMINATOR_OUTPUT_NAME}")
            val realPicInfer = infer.get(NDArrayIndex.interval(0, batchSize))
            val fakePicInfer = infer.get(NDArrayIndex.interval(batchSize, 2 * batchSize))

            realPicScoreSum += realPicInfer.sum(0).getDouble(0)
            fakePicScoreSum += fakePicInfer.sum(0).getDouble(0)
            picCount += realPicInfer.shape()[0].toInt()

            loopCounter++

            val realPicAvg = realPicScoreSum / picCount
            val fakePicAvg = fakePicScoreSum / picCount

            if (loopCounter % 200 == 0) {
                logger.info("Training D $loopCounter times at iter. $i. Current realPicAvg: $realPicAvg, fakePicAvg: $fakePicAvg")

                // update last
                val deltaFakeAvg = abs(lastFakePicAvg - fakePicAvg)
                lastFakePicAvg = fakePicAvg
                // reset statistic
                realPicScoreSum = 0.0
                fakePicScoreSum = 0.0
                picCount = 0

                // If discriminator can identify real pic and not confused with fake pic
                // then stop training D
                // TODO D的停止条件为能正确识别真实的 + 生成的识别精度不再提高
                if (realPicAvg > hyperParameter.realConfidence && deltaFakeAvg <= 5e-4) {
                    logger.info("D done at iter. $i with realPicAvg: $realPicAvg, fakePicAvg: $fakePicAvg")
                    break
                }
            }
        }

        cgan.updateParameters()

        // When training G, only later batchSize samples(generated by G) could effect G's parameters
        // To train the G, we have to falsely label them as real (1)
        // since we want G generate pic like the real one
        val labelG = Nd4j.ones(batchSize, 1)

        // train G
        loopCounter = 0
        while (true) {
            val noise = Nd4j.rand(noiseDistribution, batchSize.toLong(), noiseDim.toLong())
            val label = Nd4j.zeros(batchSize.toLong(), labelDim.toLong())
            for (index in 0 until batchSize) {
                val randomLabel = Random.nextInt(0, 10)
                label.putScalar(intArrayOf(index, randomLabel), 1.0)
            }
            val dataSetG = MultiDataSet(arrayOf(noise, label), arrayOf(labelG))

            cgan.gan.fit(dataSetG)

            loopCounter++
            val output: Map<String, INDArray> =
                cgan.gan.feedForward(arrayOf(noise, label), false)
            val fakePicInfer =
                output[CGAN.DISCRIMINATOR_OUTPUT_NAME] ?: error("No layer named ${CGAN.DISCRIMINATOR_OUTPUT_NAME}")
            val fakePicAvg = fakePicInfer.sum(0).div(batchSize).getDouble(0)

            if (loopCounter % 200 == 0) {
                logger.info("Training G $loopCounter times at iter. $i. Current fakePicAvg: $fakePicAvg")
            }
            // If discriminator cannot identify fake pic(i.e. though fake pic have some chance of being real)
            // then stop training G
            if (fakePicAvg >= hyperParameter.realConfidence - fakeIdentity) {
                logger.info("G done at iter. $i with fakePicAvg: $fakePicAvg")
                break
            }
        }

        if (i % 1 == 0) {
            logger.info("Sampling number picture at iter.$i")
            cgan.updateParameters()
            // for infer, real pic are all zeros
            val inferNoise = Nd4j.rand(noiseDistribution, 10L, noiseDim.toLong())
            val inferLabel: INDArray = Nd4j.zeros(10, 10)
            for (index in 0..9) {
                // set label from 0 to 9, we want get each number's picture
                inferLabel.putScalar(intArrayOf(index, index), 1.0)
            }

            val generatorOutput = cgan.generator.feedForward(
                arrayOf(inferNoise, inferLabel), false
            )[CGAN.GENERATOR_OUTPUT_NAME] ?: error("No layer named ${CGAN.GENERATOR_OUTPUT_NAME}")

            val bufferedImage = BufferedImage(5 * 28, 2 * 28, BufferedImage.TYPE_BYTE_GRAY)
            for (j in 0..9) {
                // process each digits
                val digitINDArray = generatorOutput.getRow(j.toLong())
                // 0 1 2 3 4
                // 5 6 7 8 9
                for (index in 0..783) {
                    bufferedImage.raster.setSample(
                        (j % 5) * 28 + index % 28,
                        (j / 5) * 28 + index / 28,
                        0,          // channel
                        255 * digitINDArray.getDouble(index.toLong())
                    )
                }
            }
            ImageIO.write(
                bufferedImage,
                "png",
                File(workingFolder, "iter${i}.png")
            )
        }
    }

    cgan.save(File(workingFolder, "save"))

    uiServer.stop()
    logger.info("Done!")
}
