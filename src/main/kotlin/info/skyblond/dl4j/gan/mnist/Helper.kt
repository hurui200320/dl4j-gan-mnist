package info.skyblond.dl4j.gan.mnist

import com.google.gson.GsonBuilder
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import java.awt.image.BufferedImage
import java.io.File

fun File.prepareFolder(delete: Boolean = false): File {
    if (delete)
        this.deleteRecursively()
    this.mkdirs()
    require(this.isDirectory) { "$this is not a directory" }
    return this
}

fun File.clear(): File {
    this.deleteRecursively()
    require(!this.exists()) { "cannot delete $this" }
    return this
}

fun toBeautifiedJson(obj: Any?): String {
    val gson = GsonBuilder().setPrettyPrinting().create()
    return gson.toJson(obj)
}

fun ndArrayToImg(array: INDArray): BufferedImage {
    val bufferedImage = BufferedImage(5 * 28, 2 * 28, BufferedImage.TYPE_BYTE_GRAY)
    for (j in 0..9) {
        // process each digits
        val digitINDArray = array.getRow(j.toLong())
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
    return bufferedImage
}

/**
 * @return Pair(realPicAvg, fakePicAvg)
 * */
fun evaluateCGAN(
    cgan: CGAN,
    testDataSetIterator: DataSetIterator,
    hyperParameter: HyperParameter
): Pair<Double, Double> {
    if (!testDataSetIterator.hasNext())
        testDataSetIterator.reset()

    // sample real pic
    val dataset = testDataSetIterator.next()
    val realPic: INDArray = dataset.features
    val label: INDArray = dataset.labels.mul(hyperParameter.labelAmp)

    // sample G
    val noise = Nd4j.rand(
        noiseDistribution,
        label.shape()[0],
        noiseDim.toLong()
    )

    cgan.updateParameters()

    val fakePic = cgan.gan.feedForward(arrayOf(noise, label), false)[CGAN.GENERATOR_OUTPUT_NAME]
        ?: error("No layer named ${CGAN.GENERATOR_OUTPUT_NAME}")

    // make up the dataset for D
    val featurePicD = Nd4j.vstack(realPic, fakePic)
    val featureLabelD = Nd4j.vstack(label, label)

    val discriminatorOutput: Map<String, INDArray> =
        cgan.discriminator.feedForward(arrayOf(featurePicD, featureLabelD), false)

    val infer = discriminatorOutput[CGAN.DISCRIMINATOR_OUTPUT_NAME]
        ?: error("No layer named ${CGAN.DISCRIMINATOR_OUTPUT_NAME}")
    val realPicInfer = infer.get(NDArrayIndex.interval(0, realPic.shape()[0]))
    val fakePicInfer = infer.get(NDArrayIndex.interval(realPic.shape()[0], infer.shape()[0]))

    val realPicAvg = realPicInfer.sum(0).div(realPicInfer.shape()[0].toInt()).getDouble(0)
    val fakePicAvg = fakePicInfer.sum(0).div(fakePicInfer.shape()[0].toInt()).getDouble(0)

    return Pair(realPicAvg, fakePicAvg)
}