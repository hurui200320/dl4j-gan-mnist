package info.skyblond.dl4j.gan.mnist

import kotlin.math.pow

fun main() {
    // 搜索半衰期为指定iter的gamma值
    val halfLife = 200 // 每隔多少iter使得fakeIdentity减半
    val totalIter = 1000 // 总迭代次数
    val initValue = 0.6 // 初始fakeIdentity
    // 衰减gamma应该在[0,1]之间
    var gammaLeft = 0.0
    var gammaRight = 1.0
    // 二分搜索
    while (gammaRight - gammaLeft >= 1e-5) {
        val newGamma = (gammaLeft + gammaRight) / 2.0
        val y = newGamma.pow(halfLife)
        when {
            y > 0.5 -> {
                gammaRight = newGamma
            }
            y < 0.5 -> {
                gammaLeft = newGamma
            }
            else -> {
                gammaLeft = newGamma
                gammaRight = newGamma
            }
        }
    }
    val newGamma = (gammaLeft + gammaRight) / 2.0
    var y = newGamma.pow(halfLife)
    println("$newGamma^$halfLife = $y")
    y = newGamma.pow(totalIter).times(initValue)
    println("$initValue*$newGamma^$totalIter = $y")

}