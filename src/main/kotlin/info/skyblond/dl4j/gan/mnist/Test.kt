package info.skyblond.dl4j.gan.mnist

import com.madgag.gif.fmsware.AnimatedGifEncoder
import java.io.File
import java.io.FileOutputStream
import javax.imageio.ImageIO


fun main() {
    val lastInterval = 5000 // 最后一张图持续时间，ms
    val internalInterval = 10 // 每张图展示时间，ms
    val scaleRate = 3.0 // 放大比例
    val folder = File("tp2")

    val gifEncoder = AnimatedGifEncoder()
    // 设置循环模式，0为无限循环 这里没有使用源文件的播放次数
    gifEncoder.setRepeat(0)
    gifEncoder.start(FileOutputStream(File("result.gif")))

    val imageList = folder.listFiles()!!
        .toList()
        .filter { it.extension == "png" }
        .sortedBy { it.name.removePrefix("iter").removeSuffix(".png").toInt() }
        .map { if (scaleRate == 1.0) ImageIO.read(it) else ImageUtils.scalePic(ImageIO.read(it), scaleRate) }

    imageList
        .fold(gifEncoder) { encoder, image ->
            encoder.setDelay(internalInterval)
            encoder.addFrame(image)
            encoder
        }
    // 延长最后一张图
    gifEncoder.setDelay(lastInterval - internalInterval)
    gifEncoder.addFrame(imageList.last())
    gifEncoder.finish()
}