import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


class Test extends munit.FunSuite {

  // compare row a and row b
  def compareRow(a: INDArray, b: INDArray): Boolean = {
    if (a.length() != b.length()) false
    else {
      val length = a.length()
      (0 until length.toInt).map(i => {
        a.getScalar(i.toLong) == b.getScalar(i.toLong)
      }).reduce((x, y) => {x && y})
    }
  }

  test("readFile") {
    val inputPath = "src/test/resources/testFile.txt"
    val content = DataSetHelper.readFileFS(inputPath)
    val expectedContent = "I am Manh"
    assertEquals(content, expectedContent, "readFile failed: content does not match expected content")
  }

  test("getContext: # tokens < windowSize") {
    val windowSize = 10
    val tokens = List(1,2,3,4,5)
    val context = TokenHelper.getContext(tokens, windowSize)
    val expectedContext = List(0,0,0,0,0,1,2,3,4,5)
    assert(context.equals(expectedContext), "getContext failed: input tokens are given less than window size, and should be padded with 0 to fit window size")
  }

  test("getContext: # tokens >= windowSize") {
    val windowSize = 10
    val tokens = List(1,2,3,4,5,6,7,8,9,10,11,12)
    val context = TokenHelper.getContext(tokens, windowSize)
    val expectedContext = List(3,4,5,6,7,8,9,10,11,12)
    assert(context.equals(expectedContext), "getContext failed: should give latest context of length windowSize")
  }

  test("tokenToEmbedding") {
    val embeddingMatrix = Nd4j.rand(6, 4)
    val tokens = List(1,4)
    val embeddings = TokenHelper.tokensToEmbedding(tokens, embeddingMatrix)
    val row1 = embeddings.getRow(0)
    val expectedRow1 = embeddingMatrix.getRow(1)
    val row2 = embeddings.getRow(1)
    val expectedRow2 = embeddingMatrix.getRow(4)
    assert(compareRow(row1, expectedRow1), "tokenToEmbedding failed: first row of output embeddings is wrong")
    assert(compareRow(row2, expectedRow2), "tokenToEmbedding failed: second row of output embeddings is wrong")
  }

  test("createSlidingWindows") {
    val embeddingSize = 4
    val vocabSize = 10
    val embeddingMatrix = Nd4j.create(Array.range(0, vocabSize * embeddingSize).map(_.toFloat)).reshape(vocabSize, embeddingSize)
    val positionEmbedding = Nd4j.zeros()
    val windowSize = 3
    val tokens = List(1,2,3,4,5,6)
    val dataset = DataSetHelper.createSlidingWindows(tokens, windowSize, embeddingMatrix, positionEmbedding)
    assert(dataset.length == 3, "createSlidingWindows: number of window dataset is wrong")
  }
}
