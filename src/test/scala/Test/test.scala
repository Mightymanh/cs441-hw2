package Test

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.{Encoding, EncodingRegistry, EncodingType}
import org.apache.commons.io.FileUtils

import java.io.File

class test extends munit.FunSuite {

  val outputTestPath = "src/test/resources/testOutput/"

  // encoding
  val registry: EncodingRegistry = Encodings.newDefaultEncodingRegistry()
  val enc: Encoding = registry.getEncoding(EncodingType.R50K_BASE)

  def cleanUp() = {
    val dir: File = new File(outputTestPath)
    if (dir.exists()) {
      FileUtils.deleteDirectory(dir)
    }
  }

  override def beforeEach(context: BeforeEach): Unit = {
    println(s"setting up test ${context.test.name}")
    cleanUp()
  }
  override def afterEach(context: AfterEach): Unit = {
    cleanUp()
    println(s"closing up test ${context.test.name}")
  }

  test("readFile") {

  }

}
