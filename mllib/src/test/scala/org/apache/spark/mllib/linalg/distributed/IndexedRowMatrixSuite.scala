/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg.distributed

import breeze.linalg.{diag => brzDiag, norm => brzNorm, svd => brzSvd,
  DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{Matrices, Vectors}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD

class IndexedRowMatrixSuite extends SparkFunSuite with MLlibTestSparkContext {

  val m = 4
  val n = 3
  val data = Seq(
    (0L, Vectors.dense(0.0, 1.0, 2.0)),
    (1L, Vectors.dense(3.0, 4.0, 5.0)),
    (3L, Vectors.dense(9.0, 0.0, 1.0))
  ).map(x => IndexedRow(x._1, x._2))
  var indexedRows: RDD[IndexedRow] = _

  override def beforeAll() {
    super.beforeAll()
    indexedRows = sc.parallelize(data, 2)
  }

  test("size") {
    val mat1 = new IndexedRowMatrix(indexedRows)
    assert(mat1.numRows() === m)
    assert(mat1.numCols() === n)

    val mat2 = new IndexedRowMatrix(indexedRows, 5, 0)
    assert(mat2.numRows() === 5)
    assert(mat2.numCols() === n)
  }

  test("empty rows") {
    val rows = sc.parallelize(Seq[IndexedRow](), 1)
    val mat = new IndexedRowMatrix(rows)
    intercept[RuntimeException] {
      mat.numRows()
    }
    intercept[RuntimeException] {
      mat.numCols()
    }
  }

  test("toBreeze") {
    val mat = new IndexedRowMatrix(indexedRows)
    val expected = BDM(
      (0.0, 1.0, 2.0),
      (3.0, 4.0, 5.0),
      (0.0, 0.0, 0.0),
      (9.0, 0.0, 1.0))
    assert(mat.toBreeze() === expected)
  }

  test("toRowMatrix") {
    val idxRowMat = new IndexedRowMatrix(indexedRows)
    val rowMat = idxRowMat.toRowMatrix()
    assert(rowMat.numCols() === n)
    assert(rowMat.numRows() === 3, "should drop empty rows")
    assert(rowMat.rows.collect().toSeq === data.map(_.vector).toSeq)
  }

  test("toCoordinateMatrix") {
    val idxRowMat = new IndexedRowMatrix(indexedRows)
    val coordMat = idxRowMat.toCoordinateMatrix()
    assert(coordMat.numRows() === m)
    assert(coordMat.numCols() === n)
    assert(coordMat.toBreeze() === idxRowMat.toBreeze())
  }

  test("toBlockMatrix") {
    val idxRowMat = new IndexedRowMatrix(indexedRows)
    val blockMat = idxRowMat.toBlockMatrix(2, 2)
    assert(blockMat.numRows() === m)
    assert(blockMat.numCols() === n)
    assert(blockMat.toBreeze() === idxRowMat.toBreeze())

    intercept[IllegalArgumentException] {
      idxRowMat.toBlockMatrix(-1, 2)
    }
    intercept[IllegalArgumentException] {
      idxRowMat.toBlockMatrix(2, 0)
    }
  }

  test("multiply a local matrix") {
    val A = new IndexedRowMatrix(indexedRows)
    val B = Matrices.dense(3, 2, Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    val C = A.multiply(B)
    val localA = A.toBreeze()
    val localC = C.toBreeze()
    val expected = localA * B.asBreeze.asInstanceOf[BDM[Double]]
    assert(localC === expected)
  }

  test("QR Decomposition") {
    val A = new IndexedRowMatrix(indexedRows)
    val result = A.tallSkinnyQR(true)
    val expected = breeze.linalg.qr.reduced(A.toBreeze())
    val calcQ = result.Q
    val calcR = result.R
    assert(closeToZero(abs(expected.q) - abs(calcQ.toBreeze())))
    assert(closeToZero(abs(expected.r) - abs(calcR.asBreeze.asInstanceOf[BDM[Double]])))
    assert(closeToZero(calcQ.multiply(calcR).toBreeze - A.toBreeze()))
    // Decomposition without computing Q
    val rOnly = A.tallSkinnyQR(computeQ = false)
    assert(rOnly.Q == null)
    assert(closeToZero(abs(expected.r) - abs(rOnly.R.asBreeze.asInstanceOf[BDM[Double]])))
  }

  test("gram") {
    val A = new IndexedRowMatrix(indexedRows)
    val G = A.computeGramianMatrix()
    val expected = BDM(
      (90.0, 12.0, 24.0),
      (12.0, 17.0, 22.0),
      (24.0, 22.0, 30.0))
    assert(G.asBreeze === expected)
  }

  test("svd") {
    val A = new IndexedRowMatrix(indexedRows)
    val svd = A.computeSVD(n, computeU = true)
    assert(svd.U.isInstanceOf[IndexedRowMatrix])
    val localA = A.toBreeze()
    val U = svd.U.toBreeze()
    val s = svd.s.asBreeze.asInstanceOf[BDV[Double]]
    val V = svd.V.asBreeze.asInstanceOf[BDM[Double]]
    assert(closeToZero(U.t * U - BDM.eye[Double](n)))
    assert(closeToZero(V.t * V - BDM.eye[Double](n)))
    assert(closeToZero(U * brzDiag(s) * V.t - localA))
  }

  test("validate matrix sizes of svd") {
    val k = 2
    val A = new IndexedRowMatrix(indexedRows)
    val svd = A.computeSVD(k, computeU = true)
    assert(svd.U.numRows() === m)
    assert(svd.U.numCols() === k)
    assert(svd.s.size === k)
    assert(svd.V.numRows === n)
    assert(svd.V.numCols === k)
  }

  test("validate k in svd") {
    val A = new IndexedRowMatrix(indexedRows)
    intercept[IllegalArgumentException] {
      A.computeSVD(-1)
    }
  }

  test("tallSkinnySVD") {
    val mat = new IndexedRowMatrix(indexedRows)
    val localMat = mat.toBreeze()
    val brzSvd.SVD(localU, localSigma, localVt) = brzSvd(localMat)
    val localV: BDM[Double] = localVt.t.toDenseMatrix
    val k = 2
    val svd = mat.tallSkinnySVD(k, sc, computeU = true)
    val U = svd.U
    val s = svd.s
    val V = svd.V
    assert(U.numRows() === m)
    assert(U.numCols() === k)
    assert(s.size === k)
    assert(V.numRows === n)
    assert(V.numCols === k)
    assertColumnEqualUpToSign(U.toBreeze(), localU, k)
    assertColumnEqualUpToSign(V.asBreeze.asInstanceOf[BDM[Double]], localV, k)
    assert(closeToZero(s.asBreeze.asInstanceOf[BDV[Double]] -
      localSigma(0 until k)))

    val svdWithoutU = mat.tallSkinnySVD(k, sc, computeU = false)
    assert(svdWithoutU.U === null)

    intercept[IllegalArgumentException] {
      mat.tallSkinnySVD(k = -1, sc)
    }
  }

  test("computeSVDbyGram") {
    val mat = new IndexedRowMatrix(indexedRows)
    val localMat = mat.toBreeze()
    val brzSvd.SVD(localU, localSigma, localVt) = brzSvd(localMat)
    val localV: BDM[Double] = localVt.t.toDenseMatrix
    val svd = mat.computeSVDbyGram(computeU = true)
    val U = svd.U
    val s = svd.s
    val V = svd.V
    assert(U.numRows() === m)
    assert(U.numCols() === n)
    assert(s.size === n)
    assert(V.numRows === n)
    assert(V.numCols === n)
    assertColumnEqualUpToSign(U.toBreeze(), localU, n)
    assertColumnEqualUpToSign(V.asBreeze.asInstanceOf[BDM[Double]], localV, n)
    assert(closeToZero(s.asBreeze.asInstanceOf[BDV[Double]] -
      localSigma(0 until n)))

    val svdWithoutU = mat.computeSVDbyGram(computeU = false)
    assert(svdWithoutU.U === null)
  }

  test("similar columns") {
    val A = new IndexedRowMatrix(indexedRows)
    val gram = A.computeGramianMatrix().asBreeze.toDenseMatrix

    val G = A.columnSimilarities().toBreeze()

    for (i <- 0 until n; j <- i + 1 until n) {
      val trueResult = gram(i, j) / scala.math.sqrt(gram(i, i) * gram(j, j))
      assert(math.abs(G(i, j) - trueResult) < 1e-6)
    }
  }

  def closeToZero(G: BDM[Double]): Boolean = {
    G.valuesIterator.map(math.abs).sum < 1e-6
  }

  def closeToZero(v: BDV[Double]): Boolean = {
    brzNorm(v, 1.0) < 1e-6
  }

  def assertColumnEqualUpToSign(A: BDM[Double], B: BDM[Double], k: Int) {
    assert(A.rows === B.rows)
    for (j <- 0 until k) {
      val aj = A(::, j)
      val bj = B(::, j)
      assert(closeToZero(aj - bj) || closeToZero(aj + bj),
        s"The $j-th columns mismatch: $aj and $bj")
    }
  }
}

