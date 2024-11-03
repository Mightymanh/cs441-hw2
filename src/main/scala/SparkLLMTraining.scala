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

//      .set("spark.executor.memoryOverhead", "4g")
//      .set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator")

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
      //      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(learningRate))
      .list()
      //      .inputPreProcessor(0, new FeedForwardToRnnPreProcessor())
      //      .inputPreProcessor(1, new RnnToFeedForwardPreProcessor())
      //      .layer(0, new SelfAttentionLayer.Builder()
      //        .nIn(embeddingSize)
      //        .nOut(embeddingSize)
      //        .nHeads(1)
      //        .weightInit(WeightInit.XAVIER)
      //        .dropOut(0.1)
      //        .build())
      .layer(0, new DenseLayer.Builder()
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.RELU)
        .nIn(embeddingSize * windowSize)
        .nOut(3000)
        .build())
      //      .layer(2, new ActivationLayer.Builder()
      //        .activation(Activation.SOFTMAX)
      //        .build())
      .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .weightInit(WeightInit.XAVIER)
        .activation(Activation.SOFTMAX)
        .nIn(3000)
        .nOut(vocabSize)
        .build())
      .build()
    //    println(conf.toJson)
    val model: MultiLayerNetwork = new MultiLayerNetwork(conf)
    model
  }

  def generateNextWord(inputStr: String, model: MultiLayerNetwork, embeddingMatrix: INDArray, positionalEmbedding: INDArray): String = {
    val tokens = enc.encode(inputStr).toArray.toList
    val context = TokenHelper.getContext(tokens, windowSize)
    val inputEmbedding = TokenHelper.tokensToEmbeddingPosition(context, embeddingMatrix, positionalEmbedding).reshape(1, embeddingMatrix.size(1) * windowSize)
    val output = model.output(inputEmbedding)
    val lastRow = output //output.getRow(context.length - 1, true)
    val outputToken = Nd4j.argMax(lastRow, 1).getInt(0)
    val word = TokenHelper.tokenToWord(outputToken, enc)
    //    println(s"next word: ${word}")
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
      logger.error("need inputpath and outputpath")
      return
    }
    val inputPath = args(0)
    val outputPath = args(1)

    logger.info("initialize spark context")
    val sc: JavaSparkContext = createSparkContext()
    // create embedding matrix
    val embeddingMatrix = TokenHelper.createEmbeddingMatrix(vocabSize, embeddingSize)
//    println(s"embeddingMatrix shape: ${embeddingMatrix.shape()}")

    // create positional embedding
    val positionalEmbedding = TokenHelper.createPositionalEmbedding(embeddingSize)

    // prepare data
    // read file and encode it
//    val data = DataSetHelper.readFile(inputFile)
    logger.info("Prepare training data")
    val data = DataSetHelper.readFileFS(args(0))
    val encoded = enc.encode(data).toArray.toList
    logger.info(s"encoded ${encoded.length}")

    // create a dataset & datasetIterator
    val dataset = DataSetHelper.createSlidingWindows(encoded, windowSize, embeddingMatrix, positionalEmbedding)
    logger.info(s"Dataset size / Number of sliding windows: ${dataset.length}")
    val dataSetIterator = new ListDataSetIterator[DataSet](dataset, batchSize)
    val rddData = createRDDFromData(dataset, sc)

    // model
    logger.info("initialize model")
    val model: MultiLayerNetwork = createModel()
    model.init()

    // set up training master
    val trainingMaster = new ParameterAveragingTrainingMaster.Builder(32)
      .batchSizePerWorker(5)
      .workerPrefetchNumBatches(2)
      .build()

    val sparkModel: SparkDl4jMultiLayer = new SparkDl4jMultiLayer(sc, model, trainingMaster)

    // Set listeners to monitor the training progress
    model.setListeners(new ScoreIterationListener(10), new EvaluativeListener(dataSetIterator, 10, InvocationType.EPOCH_END))

    // train the model on distributed RDD dataset
    logger.info("Start Training")
    (0 until nEpochs).foreach(i => {
      logger.info(s"Epoch: ${i}")
      val startTime = System.currentTimeMillis
      sparkModel.fit(rddData)
      val endTime = System.currentTimeMillis
      val t = (endTime - startTime) / 1000.0
      logger.info(s"Epoch $i execute $t seconds")
      println("Current Learning Rate: " + model.getLearningRate(0))
    })

    logger.info("Total executors: " + sc.getExecutorMemoryStatus.size)

    // stop the context after training
    logger.info("Finish training. Stop the context")
    sc.stop()

    // test string
    val testStr = "The man starts with"
    logger.info(s"Test str ${testStr}")
    val sentence = generateSentence(testStr, 20, model, embeddingMatrix, positionalEmbedding)
    logger.info(sentence)

    // write sentence to file
    DataSetHelper.writeFileFS(outputPath, sentence)
  }
}
