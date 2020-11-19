package info.skyblond.dl4j.gan.mnist

import java.awt.Image
import java.awt.image.BufferedImage


object ImageUtils {
    fun scalePic(img: BufferedImage, rate: Double): BufferedImage {
        val newWidth = (img.width.toDouble() * rate).toInt()
        val newHeight = (img.height.toDouble() * rate).toInt()
        val imgCompressed = BufferedImage(newWidth, newHeight, img.type)
        imgCompressed.graphics.let {
            it.drawImage(
                img.getScaledInstance(newWidth, newHeight, Image.SCALE_REPLICATE), 0, 0, null
            )
            it.dispose()
        }
        return imgCompressed
    }
}