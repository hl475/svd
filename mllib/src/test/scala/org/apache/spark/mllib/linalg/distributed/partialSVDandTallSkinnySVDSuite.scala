/*
 * Created by huaminli on 7/21/16.
 */

package org.apache.spark.mllib.linalg.distributed

import breeze.linalg.{max, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.abs

import org.apache.spark.{SparkContext, SparkFunSuite}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.rdd.RDD

// scalastyle:off println
class partialSVDandTallSkinnySVDSuite extends SparkFunSuite with MLlibTestSparkContext{
  val rowPerPart = 20
  val colPerPart = 30
  val numPartitions = 30

  test("Test SVD") {
    val numRows = Seq(400, 400, 200, 1000)
    val numCols = Seq(300, 300, 100, 500)
    val k = Seq(25, 25, 20, 50)
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
      println("--------------------------------")
      println("Generate BlockMatrix")
      val A = generateMatrix(numRows(i),
        numCols(i), k(i), caseNumS(i), sc)
      println("Done")
      // test partialSVD
      println("Test partialSVD")
      val (ratio1, maxU1, maxV1) = partialSVDSuite(A, k(i), sc, computeU,
        isGram(i), iterPower, iterSpectralNorm, isRandom)
      println("Test partialSVD done")
      // test tallSkinnySVD
      println("Test tallSkinnySVD")
      val (ratio2, maxU2, maxV2) = tallSkinnySVDSuite(A, k(i), computeU,
        isGram(i), ifTwice(i), iterSpectralNorm)
      println("Test tallSkinnySVD done")

      println("Result: ratio of spectral norm between diff and input")
      println("partialSVD:    " + ratio1)
      println("tallSkinnySVD: " + ratio2)

      val gramTol = if (isGram(i)) 1E-6 else 1E-13
      assert(ratio1 ~== 0.0 absTol gramTol)
      assert(ratio2 ~== 0.0 absTol gramTol)

      val orthoTol = if (ifTwice(i)) 1E-14 else 1E-3
      assert(maxU1 ~== 0.0 absTol orthoTol)
      assert(maxV1 ~== 0.0 absTol orthoTol)
      assert(maxU2 ~== 0.0 absTol orthoTol)
      assert(maxV2 ~== 0.0 absTol orthoTol)
      println("Test passed")
      println("--------------------------------" +
        "--------------------------------")
    }
    println("All tests passed")
  }

  def partialSVDSuite(A: BlockMatrix, k: Int, sc: SparkContext, computeU: Boolean,
                      isGram: Boolean, iter1: Int, iter2: Int,
                      isRandom: Boolean): (Double, Double, Double) = {
    println("Compute partialSVD")
    val svdResult = time {A.partialSVD(k, sc, computeU, isGram, iter1, isRandom)}
    println("Done")

    val U = svdResult.U.toIndexedRowMatrix()
    val S = svdResult.s
    val V = svdResult.V
    val numk = S.toArray.length
    val VDenseMat = new DenseMatrix(A.numCols().toInt, numk, V.toArray)
    val SVT = Matrices.diag(S).multiply(VDenseMat.transpose).toArray
    val SVTMat = Matrices.dense(numk, A.numCols().toInt, SVT)
    val USVT = U.multiply(SVTMat)
    val diff = A.subtract(USVT.toBlockMatrix(rowPerPart, colPerPart))

    println("Max value of non-diagonal entries of left sigular vectors")
    val gramU = U.computeGramianMatrix().asBreeze.toDenseMatrix
    val maxU = max(abs((gramU - BDM.eye[Double](gramU.rows)).toDenseVector))
    println(maxU)

    println("Max value of non-diagonal entries of right sigular vectors")
    val gramV = VDenseMat.transpose.multiply(VDenseMat).asBreeze.toDenseMatrix
    val maxV = max(abs((gramV - BDM.eye[Double](gramV.rows)).toDenseVector))
    println(maxV)

    println("Estimate the spectral norm of input and reconstruction")
    val snormDiff = time {diff.spectralNormEst(iter2, sc)}
    println("Done")
    println("Estimate the spectral norm of input")
    val snormA = time {A.spectralNormEst(iter2, sc)}
    println("Done")
    val ratio = if (snormA != 0.0) snormDiff / snormA else 0.0
    (ratio, maxU, maxV)
  }

  def tallSkinnySVDSuite(A: BlockMatrix, k: Int, computeU: Boolean,
                         isGram: Boolean, ifTwice: Boolean, iter: Int):
  (Double, Double, Double) = {
    println("Convert BlockMatrix to RowMatrix")
    val indices = A.toIndexedRowMatrix().rows.map(_.index)
    val B = A.toIndexedRowMatrix().toRowMatrix()
    println("Done")
    println("Compute tallSkinnySVD")
    val svd = time {B.tallSkinnySVD(sc, k, computeU, isGram, ifTwice)}
    println("Done")

    val U = svd.U // RowMatrix
    val S = svd.s // Vector
    val V = svd.V // Matrix
    val numk = S.toArray.length
    val VDenseMat = new DenseMatrix(A.numCols().toInt, numk, V.toArray)
    val SVT = Matrices.diag(S).multiply(VDenseMat.transpose).toArray
    val SVTMat = Matrices.dense(numk, A.numCols().toInt, SVT)
    val USVT = U.multiply(SVTMat)
    val indexedRows = indices.zip(USVT.rows).map { case (i, v) =>
      IndexedRow(i, v)}
    val USVTIndexedRowMat = new IndexedRowMatrix(indexedRows)
    val diff = A.subtract(USVTIndexedRowMat.toBlockMatrix(rowPerPart, colPerPart))

    println("Max value of non-diagonal entries of left sigular vectors")
    val gramU = U.computeGramianMatrix().asBreeze.toDenseMatrix
    val maxU = max(abs((gramU - BDM.eye[Double](gramU.rows)).toDenseVector))
    println(maxU)

    println("Max value of non-diagonal entries of right sigular vectors")
    val gramV = VDenseMat.transpose.multiply(VDenseMat).asBreeze.toDenseMatrix
    val maxV = max(abs((gramV - BDM.eye[Double](gramV.rows)).toDenseVector))
    println(maxV)

    println("Estimate the spectral norm of input and reconstruction")
    val snormDiff = time {diff.spectralNormEst(iter, sc)}
    println("Done")
    println("Estimate the spectral norm of input")
    val snormA = time {A.spectralNormEst(iter, sc)}
    println("Done")
    val ratio = if (snormA != 0.0) snormDiff / snormA else 0.0
    (ratio, maxU, maxV)
  }

  def toLocalMatrix(A: RowMatrix): BDM[Double] = {
    val m = A.numRows().toInt
    val n = A.numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    var i = 0
    A.rows.collect().foreach { vector =>
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
      i += 1
    }
    mat
  }

  def generateDCT(m: Int, k: Int, sc: SparkContext): BlockMatrix = {
    val pi = 4 * math.atan(1)
    val data = Seq.tabulate(m)(n =>
      (n.toLong, Vectors.dense(Array.tabulate(k)
      { j => math.sqrt(2.0/m) * math.cos(
        pi/m * (n + .5) * (j + .5)) })))
      .map(x => IndexedRow(x._1, x._2))

    val indexedRows: RDD[IndexedRow] = sc.parallelize(data, numPartitions)
    val mat = new IndexedRowMatrix(indexedRows)
    mat.toBlockMatrix(rowPerPart, colPerPart)
  }

  def generateS(k: Int, caseNum: Int): Array[Double] = {
    caseNum match {
      case 1 =>
        Array.tabulate(k) ( i => math.exp(i * -1.0) )
      case 2 =>
        Array.tabulate(k) ( i => math.sqrt(i + 1.0)).sorted.reverse
      case 3 =>
        Array.tabulate(k) ( i => math.sqrt(i % 3)).sorted.reverse
      case 4 =>
        Array.tabulate(k) ( i => 1.0e20 * math.exp(i * -1.0) )
      case 5 =>
        Array.tabulate(k) ( i => 1.0e20 * math.sqrt(i + 1.0)).sorted.reverse
      case 6 =>
        Array.tabulate(k) ( i => 1.0e20 * math.sqrt(i % 3)).sorted.reverse
      case 7 =>
        Array.tabulate(k) ( i => 1.0e-20 * math.exp(i * -1.0) )
      case 8 =>
        Array.tabulate(k) ( i => 1.0e-20 * math.sqrt(i + 1.0)).sorted.reverse
      case 9 =>
        Array.tabulate(k) ( i => 1.0e-20 * math.sqrt(i % 3)).sorted.reverse
    }
  }

  def generateMatrix(m: Int, n: Int, k: Int,
                     caseNumS: Int, sc: SparkContext):
  BlockMatrix = {
    val U = generateDCT(m, k, sc)
    val S = new BDV(generateS(k, caseNumS))
    val SVec = Vectors.dense(S.toArray)
    val V = generateDCT(n, k, sc)
    val VDenseMat = new DenseMatrix(k, n, V.transpose.toLocalMatrix.toArray)
    val SVArray = Matrices.diag(SVec).multiply(VDenseMat).toArray
    val SV = Matrices.dense(k, n, SVArray)
    val AIndexRowMat = U.toIndexedRowMatrix().multiply(SV)
    AIndexRowMat.toBlockMatrix(rowPerPart, colPerPart)
  }
  def time[R](block: => R): R = {
    val t0 = System.nanoTime()
    val result = block    // call-by-name
    val t1 = System.nanoTime()
    println("Elapsed time: " + (t1 - t0) / 1.0e9 + "s")
    result
  }
}
// scalastyle:on println

