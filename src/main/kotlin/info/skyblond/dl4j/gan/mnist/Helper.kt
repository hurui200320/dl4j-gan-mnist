package info.skyblond.dl4j.gan.mnist

import com.google.gson.GsonBuilder
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