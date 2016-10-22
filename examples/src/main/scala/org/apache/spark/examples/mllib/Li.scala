// scalastyle:off

package org.apache.spark.examples.mllib

import Math.abs
import breeze.linalg.{max, min, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Matrix, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Li {
  val rowPerPart = 20
  val colPerPart = 30
  val numPartitions = 30

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("SparkGrep").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    testSVD(sc)
    testComb(sc)
    println("--------------------------------" +
      "--------------------------------")
    println("All tests passed")

    sc.stop()
    System.exit(0)
  }

  def testSVD(sc: SparkContext): Unit = {
    val numRows = Seq(100, 200, 500, 1000)
    val numCols = Seq(50, 100, 200, 500)
    val k = Seq(10, 20, 30, 50)
    val caseNumS = Seq(1, 4, 5, 9)
    val isGram = Seq(true, true, false, false)
    val ifTwice = Seq(true, false, true, false)
    val computeU = true
    val iterPower = 1
    val iterSpectralNorm = 20
    val isRandom = true
    for (i <- 0 to 3) {
      println("--------------------------------" +
        "--------------------------------")
      println("Setting: m = " + numRows(i) + ", n = " + numCols(i) +
        ", k = " + k(i))
      println("Setting: caseNumS = " + caseNumS(i))
      println("Setting: isGram = " + isGram(i))
      println("Setting: ifTwice = " + ifTwice(i))
      println("--------------------------------" +
        "--------------------------------")
      val A = generateMatrix(numRows(i),
        numCols(i), k(i), caseNumS(i), sc)
      // test partialSVD
      println("Test partialSVD")
      val (ratio1, maxU1, maxV1) = partialSVDSuite(A._1, A._2, k(i), sc,
        computeU, isGram(i), iterPower, iterSpectralNorm, isRandom)
      // test tallSkinnySVD
      println("--------------------------------" +
        "--------------------------------")
      println("Test tallSkinnySVD")
      val (ratio2, maxU2, maxV2) = tallSkinnySVDSuite(A._1, A._2, k(i), sc,
        computeU, isGram(i), ifTwice(i), iterSpectralNorm)
      // test computeSVD
      println("--------------------------------" +
      "--------------------------------")
      println("Test computeSVD")
      val (ratio3, maxU3, maxV3) = computeSVDSuite(A._1, A._2, k(i), sc,
        computeU, iterSpectralNorm)
      println("--------------------------------" +
        "--------------------------------")
      println("Result: ratio of spectral norm between diff and input")
      println("partialSVD:    " + ratio1)
      println("tallSkinnySVD: " + ratio2)
      println("computeSVD: " + ratio3)

      println("Result: max entries of non-diagonal entries of the Gram")
      println("matrix of left singular vectors and right singular vectors")
      println("partialSVD:    " + maxU1 + ", " + maxV1)
      println("tallSkinnySVD: " + maxU2 + ", " + maxV2)
      println("computeSVD: " + maxU2 + ", " + maxV2)

      val gramTol = if (isGram(i)) 5E-6 else 5E-13
      assert(abs(ratio1) <= gramTol)
      assert(abs(ratio2) <= gramTol)

      val orthoTol = if (ifTwice(i)) 5E-13 else 5E-6
      assert(abs(maxU1) <= orthoTol)
      assert(abs(maxV1) <= orthoTol)
      assert(abs(maxU2) <= orthoTol)
      assert(abs(maxV2) <= orthoTol)

      println("Test passed")
      println("--------------------------------" +
        "--------------------------------")
    }
    println("All tests about reconstruction passed")
  }

  def testComb(sc: SparkContext): Unit = {
    val numRows = 400
    val numCols = 300
    val k = 25
    val caseNumS = 1
    val isGram = Seq(true, true, false, false)
    val ifTwice = Seq(true, false, true, false)
    val computeU = true
    val iterPower = 1
    val iterSpectralNorm = 20
    val isRandom = true

    val A = generateMatrix(numRows, numCols, k, caseNumS, sc)
    for (i <- 0 to 3) {
      println("--------------------------------" +
        "--------------------------------")
      // println("Setting: m = " + numRows + ", n = " + numCols +
      //   ", k = " + k + ", caseNumS = " + caseNumS)
      println("Setting: isGram = " + isGram(i))
      println("Setting: ifTwice = " + ifTwice(i))
      println("--------------------------------" +
        "--------------------------------")
      // test partialSVD
      println("Test partialSVD")
      val (ratio1, maxU1, maxV1) = partialSVDSuite(A._1, A._2, k, sc,
        computeU, isGram(i), iterPower, iterSpectralNorm, isRandom)
      // test tallSkinnySVD
      println("--------------------------------" +
        "--------------------------------")
      println("Test tallSkinnySVD")
      val (ratio2, maxU2, maxV2) = tallSkinnySVDSuite(A._1, A._2, k, sc,
        computeU, isGram(i), ifTwice(i), iterSpectralNorm)
      // test computeSVD
      println("--------------------------------" +
        "--------------------------------")
      println("Result: ratio of spectral norm between diff and input")
      println("partialSVD:    " + ratio1)
      println("tallSkinnySVD: " + ratio2)

      println("Result: max entries of non-diagonal entries of the Gram")
      println("matrix of left singular vectors and right singular vectors")
      println("partialSVD:    " + maxU1 + ", " + maxV1)
      println("tallSkinnySVD: " + maxU2 + ", " + maxV2)

      val gramTol = if (isGram(i)) 5E-6 else 5E-11
      assert(abs(ratio1) <= gramTol)
      assert(abs(ratio2) <= gramTol)

      val orthoTol = if (ifTwice(i)) 5E-13 else 5E-4
      assert(abs(maxU1) <= orthoTol)
      assert(abs(maxV1) <= orthoTol)
      assert(abs(maxU2) <= orthoTol)
      assert(abs(maxV2) <= orthoTol)
      println("Test passed")
      println("--------------------------------" +
        "--------------------------------")
    }
    println("All tests about combinations of isGram and ifTwice passed")
  }

  def partialSVDSuite(A: BlockMatrix, sigma: Double, k: Int, sc: SparkContext,
                      computeU: Boolean, isGram: Boolean, iter1: Int, iter2: Int,
                      isRandom: Boolean): (Double, Double, Double) = {
    println("Compute partialSVD")
    val svd = time {A.partialSVD(k, sc, computeU, isGram, iter1, isRandom)}
    val indices = svd.U.toIndexedRowMatrix().rows.map(_.index)

    computeStatistics(A, svd.U.toIndexedRowMatrix().toRowMatrix(), svd.s,
      svd.V, indices, iter2, sc, sigma)
  }

  def tallSkinnySVDSuite(A: BlockMatrix, sigma: Double, k: Int, sc: SparkContext,
                         computeU: Boolean, isGram: Boolean, ifTwice: Boolean,
    iter: Int): (Double, Double, Double) = {
    val indices = A.toIndexedRowMatrix().rows.map(_.index)
    val B = A.toIndexedRowMatrix().toRowMatrix()
    println("Compute tallSkinnySVD")
    val svd = time {B.tallSkinnySVD(k, sc, computeU, isGram, ifTwice)}

    computeStatistics(A, svd.U, svd.s, svd.V, indices, iter, sc, sigma)
  }

  def computeSVDSuite(A: BlockMatrix, sigma: Double, k: Int, sc: SparkContext,
                      computeU: Boolean, iter: Int):
  (Double, Double, Double) = {
    val indices = A.toIndexedRowMatrix().rows.map(_.index)
    val B = A.toIndexedRowMatrix().toRowMatrix()
    println("Compute computeSVD")
    val svd = time {B.computeSVD(k, computeU)}

    computeStatistics(A, svd.U, svd.s, svd.V, indices, iter, sc, sigma)
  }

  def computeStatistics(A: BlockMatrix, U: RowMatrix, S: Vector, V: Matrix,
                        indices: RDD[Long], iter2: Int, sc: SparkContext,
                        sigma: Double): (Double, Double, Double) = {
    val numk = S.toArray.length
    val VDenseMat = new DenseMatrix(A.numCols().toInt, numk, V.toArray)
    val SVT = Matrices.diag(S).multiply(VDenseMat.transpose).toArray
    val SVTMat = Matrices.dense(numk, A.numCols().toInt, SVT)
    val USVT = U.multiply(SVTMat)
    val indexedRows = indices.zip(USVT.rows).map { case (i, v) =>
      IndexedRow(i, v)}
    val USVTIndexedRowMat = new IndexedRowMatrix(indexedRows)
    val diff = A.subtract(USVTIndexedRowMat.toBlockMatrix(rowPerPart, rowPerPart))

    val gramUMat = U.computeGramianMatrix()
    val gramU = new BDM[Double](gramUMat.numRows, gramUMat.numCols,
      gramUMat.toArray)
    val UoffDiagArr = new BDV((gramU - BDM.eye[Double](gramU.rows)).toArray)
    val maxU = math.max(max(UoffDiagArr), math.abs(min(UoffDiagArr)))
    val gramVMat = VDenseMat.transpose.multiply(VDenseMat)
    val gramV = new BDM[Double](gramVMat.numRows, gramVMat.numCols,
      gramVMat.toArray)
    val VoffDiagArr = new BDV((gramV - BDM.eye[Double](gramV.rows)).toArray)
    val maxV = math.max(max(VoffDiagArr), math.abs(min(VoffDiagArr)))

    println("Estimate the spectral norm of input and reconstruction")
    val snormDiff = time {diff.spectralNormEst(iter2, sc)}
    val ratio = if (sigma != 0.0) snormDiff / sigma else 0.0
    (ratio, maxU, maxV)
  }

  def generateDCT(m: Int, k: Int, sc: SparkContext): BlockMatrix = {
    val limit = 65535
    val pi = 4 * math.atan(1)
    if (m < limit) {
      val data = Seq.tabulate(m)(n =>
        (n.toLong, Vectors.dense(Array.tabulate(k) { j =>
          math.sqrt(2.0 / m) * math.cos(pi / m * (n + .5) * (j + .5))
        })))
        .map(x => IndexedRow(x._1, x._2))

      val indexedRows: RDD[IndexedRow] = sc.parallelize(data, numPartitions)
      val mat = new IndexedRowMatrix(indexedRows)
      mat.toBlockMatrix(rowPerPart, colPerPart)
    } else {
      // Generate a m-by-k BlockMatrix: we first generate a sequence of
      // RDD[Vector] where each RDD[Vector] contains either 65535 rows or
      // m mod 65535 rows. The number of elements in the sequence is
      // ceiling(n/65535).
      val num = ceil(m.toFloat/limit).toInt
      val rddsBlock = Seq.tabulate(num)(z =>
        sc.parallelize(Seq.tabulate(if (z < m/limit) limit
        else m.toInt%limit)(n => (z.toLong * limit + n,
          Vectors.dense(Array.tabulate(k) { j =>
            math.sqrt(2.0 / m) * math.cos(pi / m * (z.toLong * limit + n + .5)
              * (j + .5))})))
          .map(x => IndexedRow(x._1, x._2)), numPartitions))
      val rddBlockSeq = sc.union(rddsBlock)
      new IndexedRowMatrix(rddBlockSeq).
        toBlockMatrix(rowPerPart, colPerPart)
    }
  }

  def generateS(k: Int, caseNum: Int): Array[Double] = {
    caseNum match {
      case 1 =>
        Array.tabulate(k) ( i => math.exp(math.log(1e-20) *
          (k - 1 - i) / 3 * 3 / (k - 1)) ).sorted.reverse
      case 2 =>
        Array.tabulate(k) ( i => math.sqrt(i + 1.0)).sorted.reverse
      case 3 =>
        Array.tabulate(k) ( i => math.sqrt(i % 3)).sorted.reverse
      case 4 =>
        Array.tabulate(k) ( i => 1.0e20 * math.exp(math.log(1e-20) *
          (k - 1 - i) / 3 * 3 / (k - 1)) ).sorted.reverse
      case 5 =>
        Array.tabulate(k) ( i => 1.0e20 * math.sqrt(i + 1.0)).sorted.reverse
      case 6 =>
        Array.tabulate(k) ( i => 1.0e20 * math.sqrt(i % 3)).sorted.reverse
      case 7 =>
        Array.tabulate(k) ( i => 1.0e-20 * math.exp(math.log(1e-20) *
          (k - 1 - i) / 3 * 3 / (k - 1)) ).sorted.reverse
      case 8 =>
        Array.tabulate(k) ( i => 1.0e-20 * math.sqrt(i + 1.0)).sorted.reverse
      case 9 =>
        Array.tabulate(k) ( i => 1.0e-20 * math.sqrt(i % 3)).sorted.reverse
    }
  }

  def generateMatrix(m: Int, n: Int, k: Int,
                     caseNumS: Int, sc: SparkContext):
  (BlockMatrix, Double) = {
    val U = generateDCT(m, k, sc)
    val S = new BDV(generateS(k, caseNumS))
    val SVec = Vectors.dense(S.toArray)
    val V = generateDCT(n, k, sc)
    val SVT = V.toIndexedRowMatrix().multiply(Matrices.diag(SVec)).
      toBlockMatrix(rowPerPart, colPerPart).transpose
    (U.multiply(SVT), S.toArray.max)
  }

  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1.0e9 + "s")
    result
  }
}

// scalastyle:on
