import org.nd4j.linalg.factory.Nd4j

import java.io.File
import scala.io.Source

object Experiment2 {

  def readFile(filename: String): String = {
    val file = new File(filename)
    val buffer = Source.fromFile(file)
    val content = buffer.mkString
    buffer.close()
    content
  }


  def main(args: Array[String]): Unit = {
    val target = Nd4j.zeros(1,4)
    target.putScalar(Array(0L, 2L), 1)
    val id = Nd4j.argMax(target, 1).getInt(0)
    val sum = Nd4j.sum(target, 1)
    println(sum)
    println(target)
    println(id)
  }
}
