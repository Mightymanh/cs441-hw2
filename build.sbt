ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.12"//"2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "hw2"
  )

// https://mvnrepository.com/artifact/com.knuddels/jtokkit
libraryDependencies += "com.knuddels" % "jtokkit" % "1.0.0"

// https://mvnrepository.com/artifact/ch.qos.logback/logback-classic
libraryDependencies += "ch.qos.logback" % "logback-classic" % "1.2.13"

// https://mvnrepository.com/artifact/com.typesafe/config
libraryDependencies += "com.typesafe" % "config" % "1.4.3"

// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-core
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1"
// https://mvnrepository.com/artifact/org.deeplearning4j/deeplearning4j-nlp
libraryDependencies += "org.deeplearning4j" % "deeplearning4j-nlp" % "1.0.0-M2.1"
// https://mvnrepository.com/artifact/org.nd4j/nd4j-native-platform
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"

excludeDependencies += "org.apache.logging.log4j" % "log4j-slf4j-impl"

// https://mvnrepository.com/artifact/org.scalameta/munit
libraryDependencies += "org.scalameta" %% "munit" % "1.0.2" % Test

//libraryDependencies += "org.deeplearning4j" % "dl4j-spark-ml" % "0.4-rc3.8"
//libraryDependencies += "org.deeplearning4j" % "dl4j-spark" % "0.4-rc3.8"
//libraryDependencies += "org.deeplearning4j" % "dl4j-spark-nlp_2.11" % "1.0.0-M1.1"
libraryDependencies += "org.deeplearning4j" % "dl4j-spark-parameterserver_2.12" % "1.0.0-M2.1"
//libraryDependencies += "org.deeplearning4j" % "dl4j-spark3_2.12" % "1.0.0-M2"

// META-INF discarding
assembly / assemblyMergeStrategy := {
  case PathList("META-INF", xs @ _*) =>
    xs match {
      case "MANIFEST.MF" :: Nil => MergeStrategy.discard
      case "services" :: _ => MergeStrategy.concat
      case _ => MergeStrategy.discard
    }
  case "reference.conf" => MergeStrategy.concat
  case x if x.endsWith(".proto") => MergeStrategy.rename
  case x if x.contains("hadoop") => MergeStrategy.first
  case _ => MergeStrategy.first
}
