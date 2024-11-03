import Transformer.{batchSize, logger, vocabSize, windowSize}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FSDataOutputStream, FileSystem, Path}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.{Logger, LoggerFactory}

import java.io.{BufferedReader, File, InputStreamReader}
import java.util
import scala.io.Source
import scala.jdk.CollectionConverters._

object DataSetHelper {

  val logger: Logger = LoggerFactory.getLogger(this.getClass)

  def readFile(filename: String): String = {
    val file = new File(filename)
    val buffer = Source.fromFile(file)
    val content = buffer.mkString
    buffer.close()
    content
  }

  def readFileFS(inputFilePath: String): String = {
    val path = new Path(inputFilePath)
    val inputStream: FSDataInputStream = path.getFileSystem(new Configuration()).open(path)
    val buffer = new BufferedReader(new InputStreamReader(inputStream))
    val content = buffer.lines().iterator().asScala.toList.mkString(" ")
    buffer.close()
    inputStream.close()
    content
  }

  def writeFileFS(outputFilePath: String, content: String): Unit = {
    val conf = new Configuration()
    val path = new Path(outputFilePath)
    val fs: FileSystem = path.getFileSystem(conf)
    val ostream: FSDataOutputStream = fs.create(path)
    ostream.writeBytes(content)
    ostream.close()
  }

  def createSlidingWindows(tokenIds: List[Int], windowSize: Int, embeddingMatrix: INDArray, positionalEmbedding: INDArray): List[DataSet] = {
    val length = tokenIds.length
    if (length == 1) {
      logger.error("Length of tokens = 1. Not enough to create sliding windows: need at least 2 to create input window and target")
      return List.empty
    }
    val contextLength = if (length > windowSize) windowSize else {
      logger.warn(s"Too few tokenIds (length = ${length}) for specified windowSize = ${windowSize}. Default windowSize to ${length - 1}")
      length-1
    }
    val numWindows = length - contextLength

//    println(s"input: ${tokenIds.mkString(", ")}")
    val windows = (0 until numWindows).map((i: Int) => {
      // prepare context and target
      val context = tokenIds.slice(i, i + contextLength)
//      val target = tokenIds.slice(i + 1, i + contextLength + 1)
      val target = tokenIds(i + contextLength)
//      println(s"${i}: ${context.mkString(", ")}")
//      println(s"target: ${target}")

      // convert context to its embedding with position
      val positionalInputEmbeddings = TokenHelper.tokensToEmbeddingPosition(context, embeddingMatrix, positionalEmbedding).reshape(1, embeddingMatrix.size(1) * windowSize)

      val targetOutput = Nd4j.zeros(1, embeddingMatrix.size(0).toInt)
      targetOutput.putScalar(Array(0L, target % embeddingMatrix.size(0)), 1)
//      val targetOutput = TokenHelper.createTargetOutput(target, embeddingMatrix.size(0).toInt)
//      println(s"positionalInputEmbeddings ${positionalInputEmbeddings.shape().mkString(",")}: ${positionalInputEmbeddings}")
//      println(s"targetOutput ${targetOutput.shape().mkString(",")}: ${targetOutput}")
      val ds = new DataSet(positionalInputEmbeddings, targetOutput)
      ds
    }).toList

    windows
  }

}
