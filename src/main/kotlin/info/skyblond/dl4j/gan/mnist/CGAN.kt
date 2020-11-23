@file:Suppress("MemberVisibilityCanBePrivate", "unused")

package info.skyblond.dl4j.gan.mnist

import info.skyblond.dl4j.gan.mnist.CGAN.Companion.DISCRIMINATOR_LABEL_INPUT_NAME
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.DISCRIMINATOR_NODE_PREFIX
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.DISCRIMINATOR_OUTPUT_NAME
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.DISCRIMINATOR_PIC_INPUT_NAME
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.GENERATOR_LABEL_INPUT_NAME
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.GENERATOR_NODE_PREFIX
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.GENERATOR_NOISE_INPUT_NAME
import info.skyblond.dl4j.gan.mnist.CGAN.Companion.GENERATOR_OUTPUT_NAME
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration.GraphBuilder
import org.deeplearning4j.nn.conf.GradientNormalization
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.GraphVertex
import org.deeplearning4j.nn.conf.layers.Layer
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.IUpdater
import org.nd4j.linalg.learning.config.Sgd
import java.io.File

/**
 * CGAN model, contains 3 sub model: Generator(G for short), Discriminator(D for short), GAN(G + D).
 * Each model is a [ComputationGraph], and both generator and discriminator's nodes arerequired toinitialize them.
 *
 * With specific designed name rules, by supplying the generator nodes(layers, vertexes and input pre-processors)
 * and discriminator nodes, G, D and GAN can be automatically initialized:
 * + All generator layers and vertexes' name should started with [GENERATOR_NODE_PREFIX]
 * + All discriminator layers and vertexes' name should started with [DISCRIMINATOR_NODE_PREFIX]
 * + Generator and GAN model's input are named as [GENERATOR_NOISE_INPUT_NAME] and [GENERATOR_LABEL_INPUT_NAME]
 * + Discriminator's input are named as [DISCRIMINATOR_PIC_INPUT_NAME] and [DISCRIMINATOR_LABEL_INPUT_NAME]
 * + Generator's output name is [GENERATOR_OUTPUT_NAME], which is identical to [DISCRIMINATOR_PIC_INPUT_NAME]
 * + Discriminator and GAN model's output name is [DISCRIMINATOR_OUTPUT_NAME]
 *
 * Nodes supplied to initialize G and D should handle how to concat/use label and input data.
 *
 * TODO 是否要让程序处理命名前缀的问题？
 */
class CGAN(
    // one time used values. No public access offered.
    private val generatorUpdater: IUpdater,
    private val generatorOptimizer: OptimizationAlgorithm,
    private val generatorGradientNormalization: GradientNormalization,
    private val generatorNormalizationThreshold: Double,
    private val generatorLayers: List<Triple<String, Layer, List<String>>>,
    private val generatorVertexes: List<Triple<String, GraphVertex, List<String>>>,
    private val generatorInputPreProcessors: List<Pair<String, InputPreProcessor>>,

    private val discriminatorUpdater: IUpdater,
    private val discriminatorOptimizer: OptimizationAlgorithm,
    private val discriminatorGradientNormalization: GradientNormalization,
    private val discriminatorNormalizationThreshold: Double,
    private val discriminatorLayers: List<Triple<String, Layer, List<String>>> = listOf(),
    private val discriminatorVertexes: List<Triple<String, GraphVertex, List<String>>>,
    private val discriminatorInputPreProcessors: List<Pair<String, InputPreProcessor>>,

    private val seed: Long
) {
    companion object {
        const val GENERATOR_NOISE_INPUT_NAME = "noise"
        const val GENERATOR_LABEL_INPUT_NAME = "label"
        const val GENERATOR_OUTPUT_NAME = "pic"
        const val GENERATOR_NODE_PREFIX = "G-"

        const val DISCRIMINATOR_PIC_INPUT_NAME = GENERATOR_OUTPUT_NAME
        const val DISCRIMINATOR_LABEL_INPUT_NAME = GENERATOR_LABEL_INPUT_NAME
        const val DISCRIMINATOR_OUTPUT_NAME = "output"
        const val DISCRIMINATOR_NODE_PREFIX = "D-"

        /**
         * Though most of parameters can be set be [apply],
         * this class still want to offer some apis just like the original
         * [GraphBuilder], i.e.:
         * + [GraphBuilder.addLayer]
         * + [GraphBuilder.addVertex]
         * + [GraphBuilder.inputPreProcessor].
         *
         * Builder also concat prefixs to layer name. So user won't bother that.
         *
         * Example code:
         * ```kotlin
         * CGAN.Companion.Builder()
         *     .apply {
         *         generatorUpdater = Nesterovs()
         *         generatorGradientNormalization = RenormalizeL2PerLayer
         *         generatorNormalizationThreshold = 10.0
         *         discriminatorUpdater = Sgd()
         *         noiseDim = 90
         *         labelDim = 10
         *         seed = System.currentTimeMillis()
         *     }
         *     .apply {
         *         addGeneratorVertex("Input", MergeVertex(), "noise", "label")
         *         addGeneratorLayer("L1", DenseLayer.Builder().build(), "Input")
         *         addGeneratorLayer(...)
         *     }
         *     .apply {
         *         addDiscriminatorVertex("Input", MergeVertex(), "pic", "label")
         *         addDiscriminatorLayer("L1", DenseLayer.Builder().build(), "Input")
         *         addDiscriminatorLayer(...)
         *     }
         *     .build()
         * ```
         * */
        class Builder(
            var generatorUpdater: IUpdater = Sgd(),
            var generatorOptimizer: OptimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
            var generatorGradientNormalization: GradientNormalization = GradientNormalization.None,
            var generatorNormalizationThreshold: Double = 10.0,

            var discriminatorUpdater: IUpdater = Sgd(),
            var discriminatorOptimizer: OptimizationAlgorithm = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT,
            var discriminatorGradientNormalization: GradientNormalization = GradientNormalization.None,
            var discriminatorNormalizationThreshold: Double = 10.0,

            var seed: Long = 1189998819991197253L,

            private val generatorLayers: MutableList<Triple<String, Layer, List<String>>> = mutableListOf(),
            private val generatorVertexes: MutableList<Triple<String, GraphVertex, List<String>>> = mutableListOf(),
            private val generatorInputPreProcessors: MutableList<Pair<String, InputPreProcessor>> = mutableListOf(),

            private val discriminatorLayers: MutableList<Triple<String, Layer, List<String>>> = mutableListOf(),
            private val discriminatorVertexes: MutableList<Triple<String, GraphVertex, List<String>>> = mutableListOf(),
            private val discriminatorInputPreProcessors: MutableList<Pair<String, InputPreProcessor>> = mutableListOf()
        ) {
            /**
             * Add prefix if input is noe the
             * */
            private fun prepareName(prefix: String, input: String): String {
                return if (input !in listOf(
                        GENERATOR_NOISE_INPUT_NAME,
                        GENERATOR_LABEL_INPUT_NAME,
                        GENERATOR_OUTPUT_NAME,
                        DISCRIMINATOR_PIC_INPUT_NAME,
                        DISCRIMINATOR_LABEL_INPUT_NAME,
                        DISCRIMINATOR_OUTPUT_NAME
                    )
                ) {
                    prefix + input
                } else {
                    input
                }
            }

            fun addGeneratorLayer(
                layerName: String,
                layer: Layer,
                vararg layerInputs: String
            ): Builder {
                generatorLayers.add(
                    Triple(
                        prepareName(GENERATOR_NODE_PREFIX, layerName),
                        layer,
                        layerInputs.map { prepareName(GENERATOR_NODE_PREFIX, it) }.toList()
                    )
                )
                return this
            }

            fun addGeneratorVertex(
                vertexName: String,
                vertex: GraphVertex,
                vararg vertexInputs: String
            ): Builder {
                generatorVertexes.add(
                    Triple(
                        prepareName(GENERATOR_NODE_PREFIX, vertexName),
                        vertex,
                        vertexInputs.map { prepareName(GENERATOR_NODE_PREFIX, it) }.toList()
                    )
                )
                return this
            }

            fun generatorInputPreProcessor(
                layer: String,
                processor: InputPreProcessor
            ): Builder {
                generatorInputPreProcessors.add(Pair(prepareName(GENERATOR_NODE_PREFIX, layer), processor))
                return this
            }

            fun addDiscriminatorLayer(
                layerName: String,
                layer: Layer,
                vararg layerInputs: String
            ): Builder {
                discriminatorLayers.add(
                    Triple(
                        prepareName(DISCRIMINATOR_NODE_PREFIX, layerName),
                        layer,
                        layerInputs.map { prepareName(DISCRIMINATOR_NODE_PREFIX, it) }.toList()
                    )
                )
                return this
            }

            fun addDiscriminatorVertex(
                vertexName: String,
                vertex: GraphVertex,
                vararg vertexInputs: String
            ): Builder {
                discriminatorVertexes.add(
                    Triple(
                        prepareName(DISCRIMINATOR_NODE_PREFIX, vertexName),
                        vertex,
                        vertexInputs.map { prepareName(DISCRIMINATOR_NODE_PREFIX, it) }.toList()
                    )
                )
                return this
            }

            fun discriminatorInputPreProcessor(
                layer: String,
                processor: InputPreProcessor
            ): Builder {
                discriminatorInputPreProcessors.add(Pair(prepareName(DISCRIMINATOR_NODE_PREFIX, layer), processor))
                return this
            }

            fun build(): CGAN {
                // to prevent future change break the code, using named arguments
                return CGAN(
                    generatorUpdater = generatorUpdater,
                    generatorOptimizer = generatorOptimizer,
                    generatorGradientNormalization = generatorGradientNormalization,
                    generatorNormalizationThreshold = generatorNormalizationThreshold,
                    generatorLayers = generatorLayers,
                    generatorVertexes = generatorVertexes,
                    generatorInputPreProcessors = generatorInputPreProcessors,
                    discriminatorUpdater = discriminatorUpdater,
                    discriminatorOptimizer = discriminatorOptimizer,
                    discriminatorGradientNormalization = discriminatorGradientNormalization,
                    discriminatorNormalizationThreshold = discriminatorNormalizationThreshold,
                    discriminatorLayers = discriminatorLayers,
                    discriminatorVertexes = discriminatorVertexes,
                    discriminatorInputPreProcessors = discriminatorInputPreProcessors,
                    seed = seed
                )
            }
        }
    }

    // G在GAN中训练，因此GAN的训练参数与G相同
    // D单独训练，数据集来自G生成的数据和真实数据
    // 初始化GAN、D后训练流程如下：
    // 1. G生成数据，与真实数据一起训练D
    // 2. 把D的参数拷贝进GAN
    // 3. 训练GAN（此时D层为FrozenLayerWithBackprop），只更新G的参数
    // 4. 把GAN中G的部分拷贝到G里面
    // 5. 回到1，训练D
    // TODO 根据D和G的表现决定何时结束本次Batch训练

    val generator: ComputationGraph
    val discriminator: ComputationGraph
    val gan: ComputationGraph

    init {
        generator = ComputationGraph(
            NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(generatorUpdater)
                .optimizationAlgo(generatorOptimizer)
                .gradientNormalization(generatorGradientNormalization)
                .gradientNormalizationThreshold(generatorNormalizationThreshold)
                .weightInit(WeightInit.XAVIER) // Default value, can be overridden by layer config
                .activation(Activation.IDENTITY) // Default value, can be overridden by layer config
                .graphBuilder()
                .addInputs(GENERATOR_NOISE_INPUT_NAME, GENERATOR_LABEL_INPUT_NAME)
                .apply {
                    // add layers
                    generatorLayers.forEach {
                        require(it.first.startsWith(GENERATOR_NODE_PREFIX) || it.first == GENERATOR_OUTPUT_NAME) {
                            "Generator layers' name should start with '$GENERATOR_NODE_PREFIX', except output layer, which should named '$GENERATOR_OUTPUT_NAME'"
                        }
                        addLayer(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add vertexes
                    generatorVertexes.forEach {
                        require(it.first.startsWith(GENERATOR_NODE_PREFIX) || it.first == GENERATOR_OUTPUT_NAME) {
                            "Generator vertexes' name should start with '$GENERATOR_NODE_PREFIX', except output vertex, which should named '$GENERATOR_OUTPUT_NAME'"
                        }
                        addVertex(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add input pre processors
                    generatorInputPreProcessors.forEach {
                        inputPreProcessor(it.first, it.second)
                    }
                }
                .setOutputs(GENERATOR_OUTPUT_NAME)
                .build()
        )

        discriminator = ComputationGraph(
            NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(discriminatorUpdater)
                .optimizationAlgo(discriminatorOptimizer)
                .gradientNormalization(discriminatorGradientNormalization)
                .gradientNormalizationThreshold(discriminatorNormalizationThreshold)
                .weightInit(WeightInit.XAVIER) // Default value, can be overridden by layer config
                .activation(Activation.IDENTITY) // Default value, can be overridden by layer config
                .graphBuilder()
                .addInputs(DISCRIMINATOR_PIC_INPUT_NAME, DISCRIMINATOR_LABEL_INPUT_NAME)
                .apply {
                    // add layers
                    discriminatorLayers.forEach {
                        require(it.first.startsWith(DISCRIMINATOR_NODE_PREFIX) || it.first == DISCRIMINATOR_OUTPUT_NAME) {
                            "Discriminator layers' name should start with '$DISCRIMINATOR_NODE_PREFIX', except output layer, which should named '$DISCRIMINATOR_OUTPUT_NAME'"
                        }
                        addLayer(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add vertexes
                    discriminatorVertexes.forEach {
                        require(it.first.startsWith(DISCRIMINATOR_NODE_PREFIX) || it.first == DISCRIMINATOR_OUTPUT_NAME) {
                            "Discriminator vertexes' name should start with '$DISCRIMINATOR_NODE_PREFIX', except output vertex, which should named '$DISCRIMINATOR_OUTPUT_NAME'"
                        }
                        addVertex(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add input pre processors
                    discriminatorInputPreProcessors.forEach {
                        inputPreProcessor(it.first, it.second)
                    }
                }
                .setOutputs(DISCRIMINATOR_OUTPUT_NAME)
                .build()
        )

        // G is trained by GAN, thus GAN's parameters are identical to G
        gan = ComputationGraph(
            NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(generatorUpdater)
                .optimizationAlgo(generatorOptimizer)
                .gradientNormalization(generatorGradientNormalization)
                .gradientNormalizationThreshold(generatorNormalizationThreshold)
                .weightInit(WeightInit.XAVIER) // Default value, can be overridden by layer config
                .activation(Activation.IDENTITY) // Default value, can be overridden by layer config
                .graphBuilder()
                .addInputs(GENERATOR_NOISE_INPUT_NAME, GENERATOR_LABEL_INPUT_NAME)
                .apply {
                    // add generator layers
                    generatorLayers.forEach {
                        addLayer(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add generator vertexes
                    generatorVertexes.forEach {
                        addVertex(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add generator input pre processors
                    generatorInputPreProcessors.forEach {
                        inputPreProcessor(it.first, it.second)
                    }

                    // add discriminator layers
                    discriminatorLayers.forEach {
                        // wrap d layers into FrozenLayer
                        // keep d layers from updating parameter when training G
                        addLayer(it.first, FrozenLayerWithBackprop(it.second), *it.third.toTypedArray())
                    }

                    // add discriminator vertexes
                    discriminatorVertexes.forEach {
                        addVertex(it.first, it.second, *it.third.toTypedArray())
                    }

                    // add discriminator input pre processors
                    discriminatorInputPreProcessors.forEach {
                        inputPreProcessor(it.first, it.second)
                    }
                }
                .setOutputs(DISCRIMINATOR_OUTPUT_NAME)
                .build()
        )

        // initialize model
        generator.init()
        discriminator.init()
        gan.init()
    }

    /**
     * Copy parameter. D into GAN, From GAN copy into G.
     * */
    fun updateParameters() {
        // update D into GAN
        discriminatorLayers.forEach {
            gan.getLayer(it.first).setParams(discriminator.getLayer(it.first).params())
        }
        // then copy GAN's G part to G
        generatorLayers.forEach {
            generator.getLayer(it.first).setParams(gan.getLayer(it.first).params())
        }
    }

    /**
     * Save the model
     * TODO Load from file
     * */
    fun save(path: File) {
        path.prepareFolder(true)
        // save GAN and D with updater
        gan.save(File(path, "model_gan.zip"), true)
        discriminator.save(File(path, "model_discriminator.zip"), true)
        // G comes from GAN, no need to save updater
        generator.save(File(path, "model_generator.zip"))
    }
}