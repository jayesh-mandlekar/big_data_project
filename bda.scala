import java.io.File

import org.apache.spark.sql.{Row, SaveMode, SparkSession}

case class Record(key: Int, value: String)

val warehouseLocation = new File("spark-warehouse").getAbsolutePath

val spark = SparkSession
  .builder()
  .appName("Spark Hive Example")
  .config("spark.sql.warehouse.dir", warehouseLocation)
  .enableHiveSupport()
  .getOrCreate()

import spark.implicits._
import spark.sql
import org.apache.spark.ml.stat.{Correlation,ChiSquareTest}
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer, PCA}

// ----------------------------------------------------------------------------------------------------------------- //

// IMPORT DATA FROM HIVE //

val prez_df = sql("SELECT * FROM project.prez")
val prez_df2 = prez_df.where("state_po <> 'state_po'")
// val prez_df3 = prez_df2.drop("state_po","state_fips","state_cen", "state_ic", "office", "party_simplified", "version", "notes","candidatevotes","totalvotes")
// println(prez_df3.show())
// prez_df.printSchema()

// ------------------------------------------------------------------------------------------------------------------ //

// STRING TO NUMERICAL INDEX //

val year_indexer = new StringIndexer().setInputCol("year").setOutputCol("year_index")
val state_indexer = new StringIndexer().setInputCol("state").setOutputCol("state_index")
val candidate_indexer = new StringIndexer().setInputCol("candidate").setOutputCol("candidate_index")
val writein_indexer = new StringIndexer().setInputCol("writein").setOutputCol("writein_index")
val label_indexer = new StringIndexer().setInputCol("party_detailed").setOutputCol("label")

// ------------------------------------------------------------------------------------------------------------------ //

// CHI SQUARE TEST //

// val assembler = new VectorAssembler().setInputCols(Array("year_index","state_index","candidate_index","writein_index")).setOutputCol("features")

// val pipeline = new Pipeline().setStages(Array(year_indexer,state_indexer,candidate_indexer,writein_indexer,label_indexer,assembler))

// val model = pipeline.fit(prez_df3)
// val prez_df4 = model.transform(prez_df3)

// val chi =  ChiSquareTest.test(prez_df4, "features", "label")
// chi.show(truncate=false)

// val chi_test = chi.head
// print(s"pValues: ${chi_test.getAs[Vector](0)}\n")
// print(s"degreesOfFreedom: ${chi_test.getSeq[Int](1).mkString("[",",","]")}\n")
// print(s"statistics: ${chi_test.getAs[Vector](2)}\n")

// ------------------------------------------------------------------------------------------------------------------ //

// CORRELATION //
val prez_df5 = prez_df2.drop("state_po","state_fips","state_cen","state_ic","office","party_simplified","version","notes")
val assembler2 = new VectorAssembler().setInputCols(Array("year_index","state_index","candidate_index","writein_index","label")).setOutputCol("features")
val pipeline2 = new Pipeline().setStages(Array(year_indexer,state_indexer,candidate_indexer,writein_indexer,label_indexer,assembler2))
val model2 = pipeline2.fit(prez_df5)
val prez_df6 = model2.transform(prez_df5)
// prez_df5.show()

// val pearson_coeff = Correlation.corr(prez_df6,"features")
// val spearman_coeff = Correlation.corr(prez_df6,"features","spearman")

// val Row(pearson_coeff_matrix: Matrix) = pearson_coeff.head
// val matrix_rdd = spark.sparkContext.parallelize(pearson_coeff_matrix.rowIter.toSeq)
// println(prez_df5.columns)
// matrix_rdd.take(pearson_coeff_matrix.numRows).foreach(println)

// println("\n")

// val Row(spearman_coeff_matrix: Matrix) = spearman_coeff.head
// println(s"Spearman correlation matrix:\n $spearman_coeff_matrix\n")
// val matrix_rdd2 = spark.sparkContext.parallelize(spearman_coeff_matrix.rowIter.toSeq)
// println(prez_df5.columns)
// matrix_rdd2.take(spearman_coeff_matrix.numRows).foreach(println)

// ---------------------------------------------------------------------------------------------------------------- //

// PCA - PRINCIPLE COMPONENT ANALYSIS //

val pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(4).fit(prez_df6)

prez_df6.select("features").show(false)

val result = pca.transform(prez_df6).select("pcaFeatures")
result.show(false)

// --------------------------------------------------------------------------------------------------------------- //

// val house_df = sql("SELECT * FROM project.house")
// house_df.show()
// house_df.printSchema()

// val senate_df = sql("SELECT * FROM project.senate")
// senate_df.show()
// senate_df.printSchema()

// println(prez_df.count())
// println(prez_df.first())

// println(house_df.count())
// println(house_df.first())

// println(senate_df.count())
// println(senate_df.first())

System.exit(0);
