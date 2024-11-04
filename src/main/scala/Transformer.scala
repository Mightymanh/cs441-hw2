import org.slf4j.{Logger, LoggerFactory}
import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType}
import com.typesafe.config.{Config, ConfigFactory}
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{ActivationLayer, DenseLayer, GlobalPoolingLayer, OutputLayer, PoolingType, SelfAttentionLayer}
import org.deeplearning4j.nn.conf.preprocessor.{FeedForwardToRnnPreProcessor, RnnToFeedForwardPreProcessor}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.{BackpropType, MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.InvocationType
import org.deeplearning4j.optimize.listeners.{EvaluativeListener, ScoreIterationListener}
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File
import scala.annotation.tailrec
import scala.io.Source

object Transformer {

  val logger: Logger = LoggerFactory.getLogger(this.getClass)
  val appConf: Config = ConfigFactory.load().resolve()

  // encoding
  val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  val enc: Encoding = registry.getEncoding(EncodingType.R50K_BASE) // gpt2, numTokkens: 50,257 tokens

  // configuration
  val seed: Int = appConf.getInt("seed")
  val windowSize: Int = appConf.getInt("windowSize")
  val embeddingSize: Int = appConf.getInt("embeddingSize")
  val vocabSize = appConf.getInt("vocabSize") // gpt2 vocab size
  val batchSize = appConf.getInt("batchSize")
  val nEpochs = appConf.getInt("nEpochs")
  val learningRate = appConf.getInt("learningRate")

  def createModel(): MultiLayerNetwork = {
    val conf: MultiLayerConfiguration = new NeuralNetConfiguration.Builder()
      .seed(seed)
      .l2(1e-4)
      .updater(new Adam(learningRate))
      .list()
      .layer(0, new DenseLayer.Builder()
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .nIn(embeddingSize * windowSize)
        .nOut(3000)
        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX)
        .nIn(3000)
        .nOut(vocabSize)
        .build())
      .build()
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model
  }

  def generateNextWord(inputStr: String, model: MultiLayerNetwork, embeddingMatrix: INDArray, positionalEmbedding: INDArray): String = {
    val tokens = enc.encode(inputStr).toArray.toList
    val context = TokenHelper.getContext(tokens, windowSize)
    val inputEmbedding = TokenHelper.tokensToEmbeddingPosition(context, embeddingMatrix, positionalEmbedding).reshape(1, embeddingMatrix.size(1) * windowSize)
    val output = model.output(inputEmbedding)
    val outputToken = Nd4j.argMax(output, 1).getInt(0)
    val word = TokenHelper.tokenToWord(outputToken, enc)
    word
  }

  @tailrec
  def generateSentence(inputStr: String, maxWords: Int, model: MultiLayerNetwork, embeddingMatrix: INDArray, positionalEmbedding: INDArray): String = {
    if (maxWords <= 0) inputStr
    else {
      val word = generateNextWord(inputStr, model, embeddingMatrix, positionalEmbedding)
      if (word == "." || word == "END") inputStr
      else generateSentence(s"$inputStr$word", maxWords - 1, model, embeddingMatrix, positionalEmbedding)
    }
  }

  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      logger.error("need inputPath and outputPath")
      return
    }
    val inputPath = args(0)
    val outputPath = args(1)

    // create embedding matrix
    val embeddingMatrix = TokenHelper.createEmbeddingMatrix(vocabSize, embeddingSize)
    println(s"embeddingMatrix shape: ${embeddingMatrix.shape().mkString("Array(", ", ", ")")}")

    // create positional embedding
    val positionalEmbedding = TokenHelper.createPositionalEmbedding(embeddingSize)

    // read file and encode it
    val data = DataSetHelper.readFileFS(inputPath)
    val encoded = enc.encode(data).toArray.toList
    println(s"encoded ${encoded.length}")

    // create a dataset & datasetIterator
    val dataset = DataSetHelper.createSlidingWindows(encoded, windowSize, embeddingMatrix, positionalEmbedding)
    logger.info(s"Dataset size / Number of sliding windows: ${dataset.length}")
    val dataSetIterator = new ListDataSetIterator[DataSet](dataset, batchSize)

    // create a model
    val model = createModel()
    model.init()
    model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(dataSetIterator, 1, InvocationType.EPOCH_END))

    // test a model
    val testStr = "The man starts with"
    logger.info(generateSentence(testStr, 20, model, embeddingMatrix, positionalEmbedding))

    // train model
    logger.info(s"Training model with ${nEpochs} epochs and with batchSize = ${batchSize}")
    (0 until nEpochs).foreach(epoch => {
      logger.info(s"epoch ${epoch}")
      val startTime = System.currentTimeMillis
      dataSetIterator.reset()
      model.fit(dataSetIterator, nEpochs)
      val endTime = System.currentTimeMillis
      val t = (endTime - startTime) / 1000.0
      logger.info(s"Epoch $epoch execute $t seconds")
      println("Current Learning Rate: " + model.getLearningRate(0))
    })

    // save a model
//    val outputFile = new File("/Users/mightymanh/Desktop/myCode/cs441/hw2/src/main/resources/output")
//    model.save(outputFile)

    // test a model
    val sentence = generateSentence(testStr, 20, model, embeddingMatrix, positionalEmbedding)
    logger.info(sentence)

    // write sentence to file
    DataSetHelper.writeFileFS(outputPath, sentence)
  }
}
