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

import java.util.Arrays

import scala.util.Random

import breeze.linalg.{axpy => brzAxpy, eigSym, shuffle, svd => brzSvd,
  DenseMatrix => BDM, DenseVector => BDV, MatrixSingularException,
  SparseVector => BSV}
import breeze.linalg.eigSym.EigSym
import breeze.math.{i, Complex}
import breeze.numerics.{sqrt => brzSqrt}
import breeze.signal.{fourierTr, iFourierTr}

import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Represents a row of [[org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix]].
 */
@Since("1.0.0")
case class IndexedRow(index: Long, vector: Vector)

/**
 * Represents a row-oriented [[org.apache.spark.mllib.linalg.distributed.DistributedMatrix]] with
 * indexed rows.
 *
 * @param rows indexed rows of this matrix
 * @param nRows number of rows. A non-positive value means unknown, and then the number of rows will
 *              be determined by the max row index plus one.
 * @param nCols number of columns. A non-positive value means unknown, and then the number of
 *              columns will be determined by the size of the first row.
 */
@Since("1.0.0")
class IndexedRowMatrix @Since("1.0.0") (
    @Since("1.0.0") val rows: RDD[IndexedRow],
    private var nRows: Long,
    private var nCols: Int) extends DistributedMatrix with Logging{

  /** Alternative constructor leaving matrix dimensions to be determined automatically. */
  @Since("1.0.0")
  def this(rows: RDD[IndexedRow]) = this(rows, 0L, 0)

  @Since("1.0.0")
  override def numCols(): Long = {
    if (nCols <= 0) {
      // Calling `first` will throw an exception if `rows` is empty.
      nCols = rows.first().vector.size
    }
    nCols
  }

  @Since("1.0.0")
  override def numRows(): Long = {
    if (nRows <= 0L) {
      // Reduce will throw an exception if `rows` is empty.
      nRows = rows.map(_.index).reduce(math.max) + 1L
    }
    nRows
  }


  /**
   * Compute all cosine similarities between columns of this matrix using the brute-force
   * approach of computing normalized dot products.
   *
   * @return An n x n sparse upper-triangular matrix of cosine similarities between
   *         columns of this matrix.
   */
  @Since("1.6.0")
  def columnSimilarities(): CoordinateMatrix = {
    toRowMatrix().columnSimilarities()
  }

  /**
   * Drops row indices and converts this matrix to a
   * [[org.apache.spark.mllib.linalg.distributed.RowMatrix]].
   */
  @Since("1.0.0")
  def toRowMatrix(): RowMatrix = {
    new RowMatrix(rows.map(_.vector), 0L, nCols)
  }

  /**
   * Converts to BlockMatrix. Creates blocks of `SparseMatrix` with size 1024 x 1024.
   */
  @Since("1.3.0")
  def toBlockMatrix(): BlockMatrix = {
    toBlockMatrix(1024, 1024)
  }

  /**
   * Converts to BlockMatrix. Creates blocks of `SparseMatrix`.
   * @param rowsPerBlock The number of rows of each block. The blocks at the bottom edge may have
   *                     a smaller value. Must be an integer value greater than 0.
   * @param colsPerBlock The number of columns of each block. The blocks at the right edge may have
   *                     a smaller value. Must be an integer value greater than 0.
   * @return a [[BlockMatrix]]
   */
  @Since("1.3.0")
  def toBlockMatrix(rowsPerBlock: Int, colsPerBlock: Int): BlockMatrix = {
    // TODO: This implementation may be optimized
    toCoordinateMatrix().toBlockMatrix(rowsPerBlock, colsPerBlock)
  }

  /**
   * Converts this matrix to a
   * [[org.apache.spark.mllib.linalg.distributed.CoordinateMatrix]].
   */
  @Since("1.3.0")
  def toCoordinateMatrix(): CoordinateMatrix = {
    val entries = rows.flatMap { row =>
      val rowIndex = row.index
      row.vector match {
        case SparseVector(size, indices, values) =>
          Iterator.tabulate(indices.length)(i => MatrixEntry(rowIndex, indices(i), values(i)))
        case DenseVector(values) =>
          Iterator.tabulate(values.length)(i => MatrixEntry(rowIndex, i, values(i)))
      }
    }
    new CoordinateMatrix(entries, numRows(), numCols())
  }

  /**
   * Multiplies the Gramian matrix `A^T A` by a dense vector on the right
   * without computing `A^T A`.
   *
   * @param v a dense vector whose length must match the number of columns of
   *          this matrix.
   * @return a dense vector representing the product.
   */
  private[mllib] def multiplyGramianMatrixBy(v: BDV[Double]): BDV[Double] = {
    val n = numCols().toInt
    val vbr = rows.context.broadcast(v)
    rows.treeAggregate(BDV.zeros[Double](n))(
      seqOp = (U, r) => {
        val rBrz = r.vector.asBreeze
        val a = rBrz.dot(vbr.value)
        rBrz match {
          // use specialized axpy for better performance
          case _: BDV[_] => brzAxpy(a, rBrz.asInstanceOf[BDV[Double]], U)
          case _: BSV[_] => brzAxpy(a, rBrz.asInstanceOf[BSV[Double]], U)
          case _ => throw new UnsupportedOperationException(
            s"Do not support vector operation from type" +
              s" ${rBrz.getClass.getName}.")
        }
        U
      }, combOp = (U1, U2) => U1 += U2)
  }

  /**
   * Computes singular value decomposition of this matrix. Denote this matrix
   * by A (m x n). This will compute matrices U, S, V such that
   * A ~ U * S * V', where S contains the leading k singular values, U and V
   * contain the corresponding singular vectors.
   *
   * At most k largest non-zero singular values and associated vectors are
   * returned. If there are k such values, then the dimensions of the return
   * will be:
   *  - U is a IndexedRowMatrix of size m x k that satisfies U' * U = eye(k),
   *  - s is a Vector of size k, holding the singular values in descending
   *    order,
   *  - V is a Matrix of size n x k that satisfies V' * V = eye(k).
   *
   * We assume n is smaller than m, though this is not strictly required.
   * The singular values and the right singular vectors are derived
   * from the eigenvalues and the eigenvectors of the Gramian matrix A' * A. U,
   * the matrix storing the right singular vectors, is computed via matrix
   * multiplication as U = A * (V * S^-1^), if requested by user. The actual
   * method to use is determined automatically based on the cost:
   *  - If n is small (n &lt; 100) or k is large compared with n (k &gt;
   *    n / 2), we compute the Gramian matrix first and then compute its top
   *    eigenvalues and eigenvectors locally on the driver. This requires a
   *    single pass with O(n^2^) storage on each executor and on the driver,
   *    and O(n^2^ k) time on the driver.
   *  - Otherwise, we compute (A' * A) * v in a distributive way and send it to
   *    ARPACK's DSAUPD to compute (A' * A)'s top eigenvalues and eigenvectors
   *    on the driver node. This requires O(k) passes, O(n) storage on each
   *    executor, and O(n k) storage on the driver.
   *
   * Several internal parameters are set to default values. The reciprocal
   * condition number rCond is set to 1e-9. All singular values smaller than
   * rCond * sigma(0) are treated as zeros, where sigma(0) is the largest
   * singular value. The maximum number of Arnoldi update iterations for
   * ARPACK is set to 300 or k * 3, whichever is larger. The numerical
   * tolerance for ARPACK's eigen-decomposition is set to 1e-10.
   *
   * @note The conditions that decide which method to use internally and the
   *       default parameters are subject to change.
   *
   * @param k number of leading singular values to keep (0 &lt; k &lt;= n).
   *          It might return less than k if there are numerically zero
   *          singular values or there are not enough Ritz values converged
   *          before the maximum number of Arnoldi update iterations is reached
   *          (in case that matrix A is ill-conditioned).
   * @param computeU whether to compute U
   * @param rCond the reciprocal condition number. All singular values smaller
   *              than rCond * sigma(0) are treated as zero, where sigma(0) is
   *              the largest singular value.
   * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
   */
  @Since("1.0.0")
  def computeSVD(
                  k: Int,
                  computeU: Boolean = false,
                  rCond: Double = 1e-9):
  SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    // maximum number of Arnoldi update iterations for invoking ARPACK
    val maxIter = math.max(300, k * 3)
    // numerical tolerance for invoking ARPACK
    val tol = 1e-10
    computeSVD(k, computeU, rCond, maxIter, tol, "auto")
  }

  /**
   * The actual SVD implementation, visible for testing.
   *
   * @param k number of leading singular values to keep (0 &lt; k &lt;= n).
   * @param computeU whether to compute U.
   * @param rCond the reciprocal condition number.
   * @param maxIter max number of iterations (if ARPACK is used).
   * @param tol termination tolerance (if ARPACK is used).
   * @param mode computation mode (auto: determine automatically which mode to
   *             use, local-svd: compute gram matrix and computes its full SVD
   *             locally, local-eigs: compute gram matrix and computes its top
   *             eigenvalues locally, dist-eigs: compute the top eigenvalues of
   *             the gram matrix distributively).
   * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
   */
  private[mllib] def computeSVD(
                                 k: Int,
                                 computeU: Boolean,
                                 rCond: Double,
                                 maxIter: Int,
                                 tol: Double,
                                 mode: String):
  SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    val n = numCols().toInt
    require(k > 0 && k <= n, s"Requested k singular values but got k=$k" +
      s" and numCols=$n.")

    object SVDMode extends Enumeration {
      val LocalARPACK, LocalLAPACK, DistARPACK = Value
    }

    val computeMode = mode match {
      case "auto" =>
        if (k > 5000) {
          logWarning(s"computing svd with k=$k and n=$n, please check" +
            " necessity")
        }

        // TODO: The conditions below are not fully tested.
        if (n < 100 || (k > n / 2 && n <= 15000)) {
          // If n is small or k is large compared with n, we better compute the
          // Gramian matrix first and then compute its eigenvalues locally,
          // instead of making multiple passes.
          if (k < n / 3) {
            SVDMode.LocalARPACK
          } else {
            SVDMode.LocalLAPACK
          }
        } else {
          // If k is small compared with n, we use ARPACK with distributed
          // multiplication.
          SVDMode.DistARPACK
        }
      case "local-svd" => SVDMode.LocalLAPACK
      case "local-eigs" => SVDMode.LocalARPACK
      case "dist-eigs" => SVDMode.DistARPACK
      case _ => throw new IllegalArgumentException(s"Do not support" +
        s" mode $mode.")
    }

    // Compute the eigen-decomposition of A' * A.
    val (sigmaSquares: BDV[Double], u: BDM[Double]) = computeMode match {
      case SVDMode.LocalARPACK =>
        require(k < n, s"k must be smaller than n in local-eigs mode but" +
          s" got k=$k and n=$n.")
        val G = computeGramianMatrix().asBreeze.asInstanceOf[BDM[Double]]
        EigenValueDecomposition.symmetricEigs(v => G * v, n, k, tol, maxIter)
      case SVDMode.LocalLAPACK =>
        // breeze (v0.10) svd latent constraint, 7 * n * n + 4 * n <
        // Int.MaxValue.
        require(n < 17515, s"$n exceeds the breeze svd capability")
        val G = computeGramianMatrix().asBreeze.asInstanceOf[BDM[Double]]
        val brzSvd.SVD(uFull: BDM[Double], sigmaSquaresFull: BDV[Double], _) =
          brzSvd(G)
        (sigmaSquaresFull, uFull)
      case SVDMode.DistARPACK =>
        if (rows.getStorageLevel == StorageLevel.NONE) {
          logWarning("The input data is not directly cached, which may hurt" +
            "performance if its parent RDDs are also uncached.")
        }
        require(k < n, s"k must be smaller than n in dist-eigs mode but got" +
          s" k=$k and n=$n.")
        EigenValueDecomposition.symmetricEigs(multiplyGramianMatrixBy, n, k,
          tol, maxIter)
    }

    val sigmas: BDV[Double] = brzSqrt(sigmaSquares)

    // Determine the effective rank.
    val sigma0 = sigmas(0)
    val threshold = rCond * sigma0
    var i = 0
    // sigmas might have a length smaller than k, if some Ritz values do not satisfy the convergence
    // criterion specified by tol after max number of iterations.
    // Thus use i < min(k, sigmas.length) instead of i < k.
    if (sigmas.length < k) {
      logWarning(s"Requested $k singular values but only found ${sigmas.length} converged.")
    }
    while (i < math.min(k, sigmas.length) && sigmas(i) >= threshold) {
      i += 1
    }
    val sk = i

    // Warn at the end of the run as well, for increased visibility.
    if (computeMode == SVDMode.DistARPACK && rows.getStorageLevel ==
      StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt" +
        " performance if its parent RDDs are also uncached.")
    }

    val s = Vectors.dense(Arrays.copyOfRange(sigmas.data, 0, sk))
    val V = Matrices.dense(n, sk, Arrays.copyOfRange(u.data, 0, n * sk))

    if (computeU) {
      // N = Vk * Sk^{-1}
      val N = new BDM[Double](n, sk, Arrays.copyOfRange(u.data, 0, n * sk))
      var i = 0
      var j = 0
      while (j < sk) {
        i = 0
        val sigma = sigmas(j)
        while (i < n) {
          N(i, j) /= sigma
          i += 1
        }
        j += 1
      }
      val U = this.multiply(Matrices.fromBreeze(N))
      SingularValueDecomposition(U, s, V)
    } else {
      SingularValueDecomposition(null, s, V)
    }
  }

  /**
   * Multiply this matrix by a local matrix on the right.
   *
   * @param B a local matrix whose number of rows must match the number of
   *          columns of this matrix.
   * @return an IndexedRowMatrix representing the product, which preserves
   *         partitioning.
   */
  @Since("1.0.0")
  def multiply(B: Matrix): IndexedRowMatrix = {
    val n = numCols().toInt
    val k = B.numCols
    require(n == B.numRows, s"Dimension mismatch: $n vs ${B.numRows}")

    require(B.isInstanceOf[DenseMatrix], s"Only support dense matrix at" +
      s" this time but found ${B.getClass.getName}.")

    val Bb = rows.context.broadcast(B.asBreeze.asInstanceOf[BDM[Double]].
      toDenseVector.toArray)
    val AB = rows.mapPartitions { iter =>
      val Bi = Bb.value
      iter.map { row =>
        val v = BDV.zeros[Double](k)
        var i = 0
        while (i < k) {
          v(i) = row.vector.asBreeze.dot(new BDV(Bi, i * n, 1, n))
          i += 1
        }
        IndexedRow(row.index, Vectors.fromBreeze(v))
      }
    }

    new IndexedRowMatrix(AB, nRows, B.numCols)
  }

  /**
   * Compute QR decomposition for [[IndexedRowMatrix]]. The implementation is
   * designed to optimize the QR decomposition (factorization) for the
   * [[IndexedRowMatrix]] of a tall and skinny shape.
   * Reference:
   *  Paul G. Constantine, David F. Gleich. "Tall and skinny QR factorizations
   *  in MapReduce architectures"
   *  ([[http://dx.doi.org/10.1145/1996092.1996103]])
   *
   * @param computeQ whether to computeQ
   * @return QRDecomposition(Q, R), Q = null if computeQ = false.
   */
  @Since("1.5.0")
  def tallSkinnyQR(computeQ: Boolean = false):
  QRDecomposition[IndexedRowMatrix, Matrix] = {
    /**
     * Solve Q*R = A for Q using forward substitution where A =
     * [[IndexedRowMatrix]] and R is upper-triangular. If the (i,i)th entry of
     * R is close to 0, then we set the ith column of Q to 0, as well.
     *
     * @param R upper-triangular matrix.
     * @return Q [[IndexedRowMatrix]] such that Q*R = A.
     */
    def forwardSolve(R: breeze.linalg.DenseMatrix[Double]):
    IndexedRowMatrix = {
      val m = numRows()
      val n = R.cols
      val dim = math.min(R.rows, n)
      val Bb = rows.context.broadcast(R(0 until dim, 0 until dim).toArray)

      val AB = rows.mapPartitions { iter =>
        val LHS = Bb.value
        val LHSMat = Matrices.dense(dim, dim, LHS).asBreeze
        val FNorm = Vectors.norm(Vectors.dense(LHS), 2.0)
        iter.map { row =>
          val RHS = row.vector.asBreeze.toArray
          val v = BDV.zeros[Double] (dim)
          // We don't use LAPACK here since it will be numerically unstable if
          // R is singular. If R is singular, we set the corresponding
          // column of Q to 0.
          for ( i <- 0 until dim) {
            if (math.abs(LHSMat(i, i)) > 1.0e-15 * FNorm) {
              var sum = 0.0
              for ( j <- 0 until i) {
                sum += LHSMat(j, i) * v(j)
              }
              v(i) = (RHS(i) - sum) / LHSMat(i, i)
            } else {
              v(i) = 0.0
            }
          }
          IndexedRow(row.index, Vectors.fromBreeze(v))
        }
      }
      new IndexedRowMatrix(AB, m, dim)
    }

    val col = numCols().toInt
    // partition into blocks of rows, and compute QR for each of them.
    val blockQRs = rows.retag(classOf[IndexedRow]).glom().
      filter(_.length != 0).map { partRows =>
      val bdm = BDM.zeros[Double](partRows.length, col)
      var i = 0
      partRows.foreach { row =>
        bdm(i, ::) := row.vector.asBreeze.t
        i += 1
      }
      breeze.linalg.qr.reduced(bdm).r
    }

    // combine the R part from previous results vertically into a tall matrix
    val combinedR = blockQRs.treeReduce { (r1, r2) =>
      val stackedR = BDM.vertcat(r1, r2)
      breeze.linalg.qr.reduced(stackedR).r
    }

    val finalR = Matrices.fromBreeze(combinedR.toDenseMatrix)
    val finalQ = if (computeQ) {
      try {
        forwardSolve(combinedR)
      } catch {
        case err: MatrixSingularException =>
          logWarning("R is not invertible and return Q as null")
          null
      }
    } else {
      null
    }
    QRDecomposition(finalQ, finalR)
  }

  /**
   * Compute SVD decomposition for [[IndexedRowMatrix]] A. The implementation
   * is designed to optimize the SVD decomposition (factorization) for the
   * [[IndexedRowMatrix]] of a tall and skinny shape. We either: (1) multiply
   * the matrix being processed by a random orthogonal matrix in order to mix
   * the columns, obviating the need for pivoting; or (2) compute the Gram
   * matrix of A.
   *
   * References:
   *   Parker, Douglass Stott, and Brad Pierce. The randomizing FFT: an
   *   alternative to pivoting in Gaussian elimination. University of
   *   California (Los Angeles). Computer Science Department, 1995.
   *   Le, Dinh, and D. Stott Parker. "Using randomization to make recursive
   *   matrix algorithms practical." Journal of Functional Programming
   *   9.06 (1999): 605-624.
   *   Benson, Austin R., David F. Gleich, and James Demmel. "Direct QR
   *   factorizations for tall-and-skinny matrices in MapReduce architectures."
   *   Big Data, 2013 IEEE International Conference on. IEEE, 2013.
   *   Mary, Theo, et al. "Performance of random sampling for computing
   *   low-rank approximations of a dense matrix on GPUs." Proceedings of the
   *   International Conference for High Performance Computing, Networking,
   *   Storage and Analysis. ACM, 2015.
   *
   * @param k number of singular values to keep. We might return less than k
   *          if there are numerically zero singular values. See rCond.
   * @param sc SparkContext used in an intermediate step which converts an
   *           upper triangular matrix to RDD[IndexedRow] if isGram = false.
   * @param computeU whether to compute U.
   * @param isGram whether to compute the Gram matrix for matrix
   *               orthonormalization.
   * @param ifTwice whether to compute orthonormalization twice to make
   *                the columns of the matrix be orthonormal to nearly the
   *                machine precision.
   * @param iteration number of times to run multiplyDFS if isGram = false.
   * @param rCond the reciprocal condition number. All singular values smaller
   *              than rCond * sigma(0) are treated as zero, where sigma(0) is
   *              the largest singular value.
   * @return SingularValueDecomposition[U, s, V], U = null if computeU = false.
   * @note it will lose half or more of the precision of the arithmetic
   *       but could accelerate the computation if isGram = true.
   */
  @Since("2.0.0")
  def tallSkinnySVD(k: Int, sc: SparkContext = null, computeU: Boolean = false,
                    isGram: Boolean = false, ifTwice: Boolean = true,
                    iteration: Int = 2, rCond: Option[Double] = None):
  SingularValueDecomposition[IndexedRowMatrix, Matrix] = {

    /**
     * Convert [[Matrix]] to [[RDD[IndexedRow]]].
     * @param mat an [[Matrix]].
     * @param sc SparkContext used to create RDDs.
     * @return RDD[IndexedRow].
     */
    def toRDD(mat: Matrix, sc: SparkContext): RDD[IndexedRow] = {
      val columns = mat.transpose.toArray.grouped(mat.numCols).zipWithIndex
      val rows = columns.toSeq
      val vectors = rows.map( row => IndexedRow(row._2.toLong,
        new DenseVector(row._1)))

      // Create RDD[IndexedRow]
      sc.parallelize(vectors)
    }

    require(k > 0 && k <= numCols().toInt,
      s"Requested k singular values but got k=$k and" +
        s" numCols=$numCols().toInt.")

    // Compute Q and R such that A = Q * R where Q has orthonormal columns.
    // When isGram = true, the columns of Q are the left singular vectors of A
    // and R is not necessary upper triangular. When isGram = false, R is upper
    // triangular.
    val (qMat, rMat) = if (isGram) {
      if (ifTwice) {
        // Apply computeSVDbyGram twice to A in order to produce the
        // factorization A = U1 * S1 * V1' = U2 * (S2 * V2' * S1 * V1')
        // = U2 * R. Orthonormalizing twice makes the columns of U2 be
        // orthonormal to nearly the machine precision.
        val svdResult1 = computeSVDbyGram(computeU = true)
        val svdResult2 = svdResult1.U.computeSVDbyGram(computeU = true)
        val V1 = svdResult1.V.asBreeze.toDenseMatrix
        val V2 = svdResult2.V.asBreeze.toDenseMatrix
        // Compute R1 = S1 * V1'.
        val R1 = new BDM[Double](V1.cols, V1.rows)
        for (i <- 0 until V1.cols) R1(i, ::) :=
          (V1(::, i) * svdResult1.s(i)).t
        // Compute R2 = S2 * V2'.
        val R2 = new BDM[Double](V2.cols, V2.rows)
        for (i <- 0 until V2.cols) R2(i, ::) :=
          (V2(::, i) * svdResult2.s(i)).t

        // Return U2 and R = R2 * R1.
        (svdResult2.U, R2 * R1)
      } else {
        // Apply computeSVDbyGram to A and directly return the result.
        return computeSVDbyGram(computeU)
      }
    } else {
      // Convert the input IndexedRowMatrix A to another IndexedRowMatrix B by
      // multiplying with a random matrix, discrete fourier transform, and
      // random shuffle, i.e. A * Q = B where Q = D * F * S. Repeat several
      // times, according to the number of iterations specified by iteration
      // (default 2).
      val (aq, randUnit, randIndex) = multiplyDFS(iteration,
        isForward = true, null, null)

      val (qMat, rMat) = if (ifTwice) {
        // Apply tallSkinnyQR twice to B in order to produce the
        // factorization B = Q1 * R1 = Q2 * R2 * R1 = Q2 * (R2 * R1) = Q * R.
        // Orthonormalizing twice makes the columns of the matrix be
        // orthonormal to nearly the machine precision. Later parts of the code
        // assume that the columns are numerically orthonormal in order to
        // simplify the computations.
        val qrResult1 = aq.tallSkinnyQR(computeQ = true)
        val qrResult2 = qrResult1.Q.tallSkinnyQR(computeQ = true)
        // Return Q and R = R2 * R1.
        (qrResult2.Q, qrResult2.R.asBreeze.toDenseMatrix *
          qrResult1.R.asBreeze.toDenseMatrix)
      } else {
        // Apply tallSkinnyQR to B such that B = Q * R.
        val qrResult = aq.tallSkinnyQR(computeQ = true)
        (qrResult.Q, qrResult.R.asBreeze.toDenseMatrix)
      }
      // Convert R to IndexedRowMatrix.
      val RIndexRowMat = new IndexedRowMatrix(toRDD(Matrices.fromBreeze(rMat),
        sc))
      // Convert RIndexRowMat back by reverse shuffle, inverse fourier
      // transform, and dividing random matrix,
      // i.e. R * Q^T = R * S^{-1} * F^{-1} * D^{-1}. Repeat several times,
      // according to the number of iterations specified by iteration.
      val (rq, _, _) = RIndexRowMat.multiplyDFS(iteration,
        isForward = false, randUnit, randIndex)
      (qMat, rq.toBreeze())
    }
    // Apply SVD on R * Q^T.
    val brzSvd.SVD(w, s, vt) = brzSvd.reduced.apply(rMat)

    // Determine the effective rank.
    val rConD = if (rCond.isDefined) rCond.get
    else if (!isGram) 1e-11 else 1e-6

    val threshold = rConD * s(0)
    var rank = 0
    while (rank < math.min(k, s.length) && s(rank) >= threshold) {
      rank += 1
    }

    // Truncate S, V.
    val sk = Vectors.fromBreeze(s(0 until rank))
    val VMat = Matrices.dense(rank, numCols().toInt, vt(0 until rank,
      0 until numCols().toInt).toArray).transpose

    if (computeU) {
      // Truncate W.
      val WMat = Matrices.dense(rMat.rows, rank,
        Arrays.copyOfRange(w.toArray, 0, rMat.rows * rank))
      // U = Q * W.
      val U = qMat.multiply(WMat)
      SingularValueDecomposition(U, sk, VMat)
    } else {
      SingularValueDecomposition(null, sk, VMat)
    }
  }

  /**
   * Given a m-by-2n or m-by-(2n+1) real [[IndexedRowMatrix]], convert it to
    * m-by-n complex [[IndexedRowMatrix]]. Multiply this m-by-n complex
    * [[IndexedRowMatrix]] by a random diagonal n-by-n [[BDM[Complex]] D,
    * discrete fourier transform F, and random shuffle n-by-n [[BDM[Int]]] S
    * with a given [[Int]] k number of times, and convert it back to m-by-2n
    * real [[IndexedRowMatrix]]; or backwards, i.e., convert it from real to
    * complex, apply reverse random shuffle n-by-n [[BDM[Int]]] S^{-1}, inverse
    * fourier transform F^{-1}, dividing the given random diagonal n-by-n
    * [[BDM[Complex]] D with a given [[Int]] k number of times, and convert it
    * from complex to real.
    *
    * References:
    *   Parker, Douglass Stott, and Brad Pierce. The randomizing FFT: an
    *   alternative to pivoting in Gaussian elimination. University of
    *   California (Los Angeles). Computer Science Department, 1995.
    *   Le, Dinh, and D. Stott Parker. "Using randomization to make recursive
    *   matrix algorithms practical." Journal of Functional Programming
    *   9.06 (1999): 605-624.
    *   Ailon, Nir, and Edo Liberty. "An almost optimal unrestricted fast
    *   Johnson-Lindenstrauss transform." ACM Transactions on Algorithms (TALG)
    *   9.3 (2013): 21.
    *
    * @note The entries with the same column index of input
    *       [[IndexedRowMatrix]] are multiplied by the same random number, and
    *       shuffle to the same place.
    *
    * @param iteration k number of times applying D, F, and S.
    * @param isForward whether to apply D, F, S forwards or backwards.
    *                  If backwards, then needs to specify rUnit and rIndex.
    * @param rUnit a complex k-by-n matrix such that each entry is a complex
    *              number with absolute value 1.
    * @param rIndex an integer k-by-n matrix such that each row is a random
    *               permutation of the integers 1, 2, ..., n.
    * @return transformed m-by-2n or m-by-(2n+1) IndexedRowMatrix, a complex
    *         k-by-n matrix, and an int k-by-n matrix.
    */
  @Since("2.0.0")
  def multiplyDFS(iteration: Int = 2, isForward: Boolean, rUnit: BDM[Complex],
                  rIndex: BDM[Int]):
  (IndexedRowMatrix, BDM[Complex], BDM[Int]) = {

    /**
     * Given a 1-by-2n [[BDV[Double]] arr, either do D, F, S forwards with
     * [[Int]] iteration k times if [[Boolean]] isForward is true;
     * or S, F, D backwards with [[Int]] iteration k times if [[Boolean]]
     * isForward is false.
     *
     * @param iteration k number of times applying D, F, and S.
     * @param isForward whether to apply D, F, S forwards or backwards.
     *                  If backwards, then needs to specify rUnit and rIndex.
     * @param randUnit a complex k-by-n such that each entry is a complex number
     *              with absolute value 1.
     * @param randIndex an integer k-by-n matrix such that each row is a random
     *                  permutation of the integers 1, 2, ..., n.
     * @param arr a 1-by-2n [[BDV[Double]]].
     * @param index the row index of arr.
     * @return a 1-by-2n [[IndexedRow]].
     */
    def dfs(iteration: Int, isForward: Boolean, randUnit: BDM[Complex],
            randIndex: BDM[Int], arr: BDV[Double], index: Long): IndexedRow = {

      // Either keep arr or add an extra entry with value 0 so that
      // number of indices is even.
      val input = {
        if (arr.length % 2 == 1) BDV.vertcat(arr, BDV.zeros[Double](1))
        else arr
      }

      // convert input from real to complex.
      var inputComplex = realToComplex(input)
      if (isForward) {
        // Apply D, F, S to the input iteration times.
        for (i <- 0 until iteration) {
          // Element-wise multiplication with randUnit.
          val inputMul = inputComplex :* randUnit(i, ::).t
          // Discrete Fourier transform.
          val inputFFT = fourierTr(inputMul).toArray
          // Random shuffle.
          val shuffleIndex = randIndex(i, ::).t.toArray
          inputComplex = BDV(shuffle(inputFFT, shuffleIndex, false))
        }
      } else {
        // Apply S^{-1}, F^{-1}, D^{-1} to the input iteration times.
        for (i <- iteration - 1 to 0 by -1) {
          // Reverse shuffle.
          val shuffleIndex = randIndex(i, ::).t.toArray
          val shuffleBackArr = shuffle(inputComplex.toArray, shuffleIndex, true)
          // Inverse Fourier transform.
          val inputIFFT = iFourierTr(BDV(shuffleBackArr))
          // Element-wise divide with randUnit.
          inputComplex = inputIFFT :/ randUnit(i, ::).t
        }
      }
      IndexedRow(index, Vectors.fromBreeze(complexToReal(inputComplex)))
    }

    /**
     * Given [[Int]] k and [[Int]] n, generate a k-by-n [[BDM[Complex]]] such
     * that each entry has absolute value 1 and a k-by-n [[BDM[Int]]] such
     * that each row is a random permutation of integers from 1 to n.
     *
     * @param iteration k number of rows for D and S.
     * @param nCols the number of columns n.
     * @return a k-by-n complex matrix and a k-by-n int matrix.
     */
    def generateDS(iteration: Int = 2, nCols: Int): (BDM[Complex], BDM[Int]) = {
      val shuffleIndex = new BDM[Int](iteration, nCols)
      val randUnit = new BDM[Complex](iteration, nCols)

      for (i <- 0 until iteration) {
        // Random permuatation of integers from 1 to n.
        shuffleIndex(i, ::) := shuffle(BDV.tabulate(nCols)(i => i)).t
        // Generate random complex number with absolute value 1. These random
        // complex numbers are uniformly distributed over the unit circle.
        for (j <- 0 until nCols) {
          Random.setSeed((851342769 + j).toLong)
          val randComplex = Complex(Random.nextGaussian(),
            Random.nextGaussian())
          randUnit(i, j) = randComplex / randComplex.abs
        }
      }
      (randUnit, shuffleIndex)
    }

    /**
     * Convert a 2n-by-1 [[BDV[Double]]] u to n-by-1 [[BDV[Complex]]] v. The
     * odd index entry of u changes to the real part of each entry in v. The
     * even index entry of u changes to the imaginary part of each entry in v.
     * Please note that "index" here refers to "1-based indexing" rather than
     * "0-based indexing."
     *
     * @param arr 2n-by-1 real vector.
     * @return n-by-1 complex vector.
     */
    def realToComplex(arr: BDV[Double]): BDV[Complex] = {
      // Odd entries transfer to real part.
      val odd = arr(0 until arr.length by 2).map(v => v + i * 0)
      // Even entries transfer to imaginary part.
      val even = arr(1 until arr.length by 2).map(v => i * v)
      // Combine real part and imaginary part.
      odd + even
    }

    /**
     * Convert a n-by-1 [[BDV[Complex]]] v to 2n-by-1 [[BDV[Double]]] u. The
     * the real part of each entry in v changes to the odd index entry of u.
     * The imaginary part of each entry in v changes to the even index entry
     * uf u. Please note that "index" here refers to "1-based indexing" rather
     * than "0-based indexing."
     *
     * @param arr n-by-1 complex vector.
     * @return 2n-by-1 real vector.
     */
    def complexToReal(arr: BDV[Complex]): BDV[Double] = {
      // Filter out the real part.
      val reconReal = arr.map(v => v.real)
      // Filter out the imaginary part.
      val reconImag = arr.map(v => v.imag)
      // Concatenate the real and imaginary part.
      BDV.horzcat(reconReal, reconImag).t.toDenseVector
    }

    // Either generate D and S or take them from the input.
    val (randUnit, randIndex) = {
      if (isForward) generateDS(iteration, (numCols().toInt + 1) / 2)
      else (rUnit, rIndex)
    }

    // Apply DFS forwards or backwards to the input IndexedRowMatrix.
    val AB = rows.mapPartitions( iter =>
      if (iter.nonEmpty) {
        val temp = iter.toArray
        val tempAfter = new Array[IndexedRow](temp.length)
        for (i <- temp.indices) {
          tempAfter(i) = dfs(iteration, isForward, randUnit, randIndex,
            BDV(temp(i).vector.toArray), temp(i).index)
        }
        Iterator.tabulate(temp.length)(tempAfter(_))
      } else {
        Iterator.empty
      }
    )

    // Generate the output IndexedRowMatrix with even number of columns.
    val n = if (nCols % 2 == 0) nCols else nCols + 1
    (new IndexedRowMatrix(AB, nRows, n), randUnit, randIndex)
  }

  /**
   * Compute the singular value decomposition of the [[IndexedRowMatrix]] A
   * such that A ~ U * S * V' via computing the Gram matrix of A. We (1)
   * compute the Gram matrix G = A' * A, (2) apply the eigenvalue decomposition
   * on G = V * D * V', (3) compute W = A * V, then the Euclidean norms of the
   * columns of W are the singular values of A, and (4) normalizing the columns
   * of W yields U such that A = U * S * V', where S is the diagonal matrix of
   * singular values.
   *
   * @return SingularValueDecomposition[U, s, V].
   * @note it will lose half or more of the precision of the arithmetic
   *       but could accelerate the computation compared to tallSkinnyQR.
   */
  @Since("2.0.0")
  def computeSVDbyGram(computeU: Boolean = false):
  SingularValueDecomposition[IndexedRowMatrix, Matrix] = {
    // Compute Gram matrix G of A such that G = A' * A.
    val G = computeGramianMatrix().asBreeze.toDenseMatrix

    // Compute the eigenvalue decomposition of G such that G = V * D * V'.
    val EigSym(d, vMat) = eigSym(G)

    // Find the effective rank of G.
    val eigenRank = {
      var i = d.length - 1
      while (i >= 0 && d(i) > 1e-14 * d(d.length - 1)) i = i - 1
      i + 1
    }

    // Calculate W such that W = A * V.
    val vMatTruncated = vMat(::, vMat.cols - 1 to eigenRank by -1)
    val V = Matrices.dense(vMatTruncated.rows, vMatTruncated.cols,
      vMat(::, vMat.cols - 1 to eigenRank by -1).toArray)
    val W = multiply(V)
    val normW = Statistics.colStats(W.rows.map(_.vector)).normL2.asBreeze.
      toDenseVector

    if (computeU) {
      // Normalize W to U such that each column of U has norm 1.
      val U = W.multiply(Matrices.diag(Vectors.fromBreeze(1.0 / normW)))
      SingularValueDecomposition(U, Vectors.fromBreeze(normW), V)
    } else {
      SingularValueDecomposition(null, Vectors.fromBreeze(normW), V)
    }
  }

  /**
   * Computes the Gramian matrix `A^T A`.
   *
   * @note This cannot be computed on matrices with more than 65535 columns.
   */
  @Since("1.0.0")
  def computeGramianMatrix(): Matrix = {
    toRowMatrix().computeGramianMatrix()
  }

  /**
   * Convert distributed storage of IndexedRowMatrix into locally stored BDM, whereas
   * asBreeze works on matrices stored locally and requires no memcopy.
   */
  private[mllib] override def toBreeze(): BDM[Double] = {
    val m = numRows().toInt
    val n = numCols().toInt
    val mat = BDM.zeros[Double](m, n)
    rows.collect().foreach { case IndexedRow(rowIndex, vector) =>
      val i = rowIndex.toInt
      vector.foreachActive { case (j, v) =>
        mat(i, j) = v
      }
    }
    mat
  }
}
