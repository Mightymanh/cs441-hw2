import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.DataSetPreProcessor
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import java.util

class ListDataSetIterator[T <: DataSet](coll: List[T], batch: Int) extends DataSetIterator {
  private var curr = 0
  private val list = coll
  private var preProcessor: DataSetPreProcessor = null

  override def hasNext: Boolean = curr < list.length

  override def next: DataSet = next(batch)

  override def remove(): Unit = {
    throw new UnsupportedOperationException
  }

  override def inputColumns: Int = list.head.getFeatures.columns

  override def totalOutcomes: Int = list.head.getLabels.columns

  override def resetSupported = true

  override def asyncSupported: Boolean = {
    //Already in memory -> doesn't make sense to prefetch
    false
  }

  override def reset(): Unit = {
    curr = 0
  }

  override def batch: Int = batch

  override def getPreProcessor: DataSetPreProcessor = preProcessor

  override def setPreProcessor(preProcessor: DataSetPreProcessor): Unit = {
    this.preProcessor = preProcessor
  }

  override def getLabels: util.List[String] = null

  override def next(num: Int): DataSet = {
    var end = curr + num
    val r = new util.ArrayList[DataSet]
    if (end >= list.length) end = list.length

    while (curr < end) {
//      println(curr)
      r.add(list(curr))
      curr += 1
    }
    val d = DataSet.merge(r)
    if (preProcessor != null) if (!d.isPreProcessed) {
      preProcessor.preProcess(d)
      d.markAsPreProcessed()
    }
    d
  }
}