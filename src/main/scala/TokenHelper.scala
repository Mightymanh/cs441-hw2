import com.knuddels.jtokkit.api.{Encoding, IntArrayList}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

object TokenHelper {
  // convert tokens to embedding vector
  def tokensToEmbedding(tokens: List[Int], embeddingMatrix: INDArray): INDArray = {
    val embedding = Nd4j.zeros(tokens.length, embeddingMatrix.size(1))
    tokens.indices.map(i => {
      val rowId = tokens(i) % embeddingMatrix.size(0)
      embedding.putRow(i, embeddingMatrix.getRow(rowId))
    })
    embedding
  }

  // convert tokenId to word
  def tokensToEmbeddingPosition(tokens: List[Int], embeddingMatrix: INDArray, positionalEmbedding: INDArray): INDArray = {
    val embedding = tokensToEmbedding(tokens, embeddingMatrix)
    val embeddingWithPosition = embedding.addRowVector(positionalEmbedding)
    embeddingWithPosition
  }

  // convert token to word
  def tokenToWord(token: Int, enc: Encoding): String = {
    val intArr = new IntArrayList(1)
    intArr.add(token)
    val word = enc.decode(intArr)
    word
  }

  // get the last context of input tokens
  def getContext(tokens: List[Int], windowSize: Int): List[Int] = {
    val length = tokens.length
    if (length < windowSize) {
      val empty = List.fill(windowSize - length)(0)
      List.concat(empty, tokens)
    }
    else {
      tokens.slice(length-windowSize, length)
    }
  }

  // create positional embedding
  def createPositionalEmbedding(embeddingSize: Int): INDArray = {
//    val positionalEmbedding = Nd4j.create(List.range(0, embeddingSize).map(_.toFloat), )
    val positionalEmbedding = Nd4j.zeros(embeddingSize)
    positionalEmbedding
  }

  // create embedding matrix
  def createEmbeddingMatrix(vocabSize: Int, embeddingSize: Int): INDArray = {
    Nd4j.rand(vocabSize, embeddingSize)
  }

  // create target output
  def createTargetOutput(targetTokens: List[Int], vocabSize: Int): INDArray = {
    val targetOutput = Nd4j.zeros(targetTokens.length, vocabSize)
    targetTokens.indices.map(i => {
      targetOutput.putScalar(Array(i.toLong, targetTokens(i).toLong), 1)
    })
    targetOutput
  }
}
