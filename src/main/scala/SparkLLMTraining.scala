import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType}
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.spark.SparkConf
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.{EvaluativeListener, ScoreIterationListener}
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster
import org.nd4j.linalg.dataset.DataSet
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{MultiLayerConfiguration, NeuralNetConfiguration}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.InvocationType
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.{Logger, LoggerFactory}

import scala.annotation.tailrec

object SparkLLMTraining {

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

  def createSparkContext(): JavaSparkContext = {
    val sparkConf: SparkConf = new SparkConf()
      .setAppName("Spark LLM Training")
    val sc: JavaSparkContext = new JavaSparkContext(sparkConf)
    sc
  }

  def createRDDFromData(data: List[DataSet], sc: JavaSparkContext): JavaRDD[DataSet] = {
    val rddData: JavaRDD[DataSet] = sc.parallelize(data)
    rddData
  }

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

    // initialize spark context
    logger.info("initialize spark context")
    val sc: JavaSparkContext = createSparkContext()

    // create embedding matrix
    val embeddingMatrix = TokenHelper.createEmbeddingMatrix(vocabSize, embeddingSize)

    // create positional embedding
    val positionalEmbedding = TokenHelper.createPositionalEmbedding(embeddingSize)

    // prepare data
    logger.info("Prepare training data")
    val data = DataSetHelper.readFileFS(inputPath)
    val encoded = enc.encode(data).toArray.toList
    logger.info(s"encoded ${encoded.length}")

    // create a dataset & datasetIterator
    val dataset = DataSetHelper.createSlidingWindows(encoded, windowSize, embeddingMatrix, positionalEmbedding)
    logger.info(s"Dataset size / Number of sliding windows: ${dataset.length}")
    val dataSetIterator = new ListDataSetIterator[DataSet](dataset, batchSize)
    val rddData = createRDDFromData(dataset, sc)

    // create LLM model
    logger.info("initialize model")
    val model: MultiLayerNetwork = createModel()
    model.init()

    // set up training master
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(5)
      .workerPrefetchNumBatches(2)
      .build()

    // wrap the model with spark context and training master
    val sparkModel: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Set listeners to monitor the training progress: loss score, evaluation score
    model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(dataSetIterator, 10, InvocationType.EPOCH_END))

    // train the model on distributed RDD dataset
    logger.info("Start Training")
    (0 until nEpochs).foreach(i => {
      logger.info(s"Epoch: ${i}")
      val startTime = System.currentTimeMillis
      sparkModel.fit(rddData)
      val endTime = System.currentTimeMillis
      val t = (endTime - startTime) / 1000.0
      logger.info(s"Epoch $i execute $t seconds") // time per epoch
      println("Current Learning Rate: " + model.getLearningRate(0)) // learning rate
    })

    //
    logger.info("Total executors: " + sc.getExecutorMemoryStatus.size)



    // stop the context after training
    logger.info("Finish training. Stop the context")
    sc.stop()

    // test the model
    val testStr = "The man starts with"
    logger.info(s"Test str ${testStr}")
    val sentence = generateSentence(testStr, 20, model, embeddingMatrix, positionalEmbedding)
    logger.info(sentence)

    // write output sentence to file
    DataSetHelper.writeFileFS(outputPath, sentence)
  }
}
