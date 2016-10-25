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

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import breeze.linalg.{eigSym, svd => brzSvd, DenseMatrix => BDM,
  DenseVector => BDV, Matrix => BM, SparseVector => BSV, Vector => BV}
import breeze.linalg.eigSym.EigSym
import breeze.numerics.ceil
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW

import org.apache.spark.{Partitioner, SparkContext, SparkException}
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * A grid partitioner, which uses a regular grid to partition coordinates.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @param rowsPerPart Number of rows per partition, which may be less at the bottom edge.
 * @param colsPerPart Number of columns per partition, which may be less at the right edge.
 */
private[mllib] class GridPartitioner(
    val rows: Int,
    val cols: Int,
    val rowsPerPart: Int,
    val colsPerPart: Int) extends Partitioner {

  require(rows > 0)
  require(cols > 0)
  require(rowsPerPart > 0)
  require(colsPerPart > 0)

  private val rowPartitions = math.ceil(rows * 1.0 / rowsPerPart).toInt
  private val colPartitions = math.ceil(cols * 1.0 / colsPerPart).toInt

  override val numPartitions: Int = rowPartitions * colPartitions

  /**
   * Returns the index of the partition the input coordinate belongs to.
   *
   * @param key The partition id i (calculated through this method for coordinate (i, j) in
   *            `simulateMultiply`, the coordinate (i, j) or a tuple (i, j, k), where k is
   *            the inner index used in multiplication. k is ignored in computing partitions.
   * @return The index of the partition, which the coordinate belongs to.
   */
  override def getPartition(key: Any): Int = {
    key match {
      case i: Int => i
      case (i: Int, j: Int) =>
        getPartitionId(i, j)
      case (i: Int, j: Int, _: Int) =>
        getPartitionId(i, j)
      case _ =>
        throw new IllegalArgumentException(s"Unrecognized key: $key.")
    }
  }

  /** Partitions sub-matrices as blocks with neighboring sub-matrices. */
  private def getPartitionId(i: Int, j: Int): Int = {
    require(0 <= i && i < rows, s"Row index $i out of range [0, $rows).")
    require(0 <= j && j < cols, s"Column index $j out of range [0, $cols).")
    i / rowsPerPart + j / colsPerPart * rowPartitions
  }

  override def equals(obj: Any): Boolean = {
    obj match {
      case r: GridPartitioner =>
        (this.rows == r.rows) && (this.cols == r.cols) &&
          (this.rowsPerPart == r.rowsPerPart) && (this.colsPerPart == r.colsPerPart)
      case _ =>
        false
    }
  }

  override def hashCode: Int = {
    com.google.common.base.Objects.hashCode(
      rows: java.lang.Integer,
      cols: java.lang.Integer,
      rowsPerPart: java.lang.Integer,
      colsPerPart: java.lang.Integer)
  }
}

private[mllib] object GridPartitioner {

  /** Creates a new [[GridPartitioner]] instance. */
  def apply(rows: Int, cols: Int, rowsPerPart: Int, colsPerPart: Int): GridPartitioner = {
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }

  /** Creates a new [[GridPartitioner]] instance with the input suggested number of partitions. */
  def apply(rows: Int, cols: Int, suggestedNumPartitions: Int): GridPartitioner = {
    require(suggestedNumPartitions > 0)
    val scale = 1.0 / math.sqrt(suggestedNumPartitions)
    val rowsPerPart = math.round(math.max(scale * rows, 1.0)).toInt
    val colsPerPart = math.round(math.max(scale * cols, 1.0)).toInt
    new GridPartitioner(rows, cols, rowsPerPart, colsPerPart)
  }
}

/**
 * Represents a distributed matrix in blocks of local matrices.
 *
 * @param blocks The RDD of sub-matrix blocks ((blockRowIndex, blockColIndex), sub-matrix) that
 *               form this distributed matrix. If multiple blocks with the same index exist, the
 *               results for operations like add and multiply will be unpredictable.
 * @param rowsPerBlock Number of rows that make up each block. The blocks forming the final
 *                     rows are not required to have the given number of rows
 * @param colsPerBlock Number of columns that make up each block. The blocks forming the final
 *                     columns are not required to have the given number of columns
 * @param nRows Number of rows of this matrix. If the supplied value is less than or equal to zero,
 *              the number of rows will be calculated when `numRows` is invoked.
 * @param nCols Number of columns of this matrix. If the supplied value is less than or equal to
 *              zero, the number of columns will be calculated when `numCols` is invoked.
 */
@Since("1.3.0")
class BlockMatrix @Since("1.3.0") (
    @Since("1.3.0") val blocks: RDD[((Int, Int), Matrix)],
    @Since("1.3.0") val rowsPerBlock: Int,
    @Since("1.3.0") val colsPerBlock: Int,
    private var nRows: Long,
    private var nCols: Long) extends DistributedMatrix with Logging {

  private type MatrixBlock = ((Int, Int), Matrix) // ((blockRowIndex, blockColIndex), sub-matrix)

  /**
   * Alternate constructor for BlockMatrix without the input of the number of rows and columns.
   *
   * @param blocks The RDD of sub-matrix blocks ((blockRowIndex, blockColIndex), sub-matrix) that
   *               form this distributed matrix. If multiple blocks with the same index exist, the
   *               results for operations like add and multiply will be unpredictable.
   * @param rowsPerBlock Number of rows that make up each block. The blocks forming the final
   *                     rows are not required to have the given number of rows
   * @param colsPerBlock Number of columns that make up each block. The blocks forming the final
   *                     columns are not required to have the given number of columns
   */
  @Since("1.3.0")
  def this(
      blocks: RDD[((Int, Int), Matrix)],
      rowsPerBlock: Int,
      colsPerBlock: Int) = {
    this(blocks, rowsPerBlock, colsPerBlock, 0L, 0L)
  }

  @Since("1.3.0")
  override def numRows(): Long = {
    if (nRows <= 0L) estimateDim()
    nRows
  }

  @Since("1.3.0")
  override def numCols(): Long = {
    if (nCols <= 0L) estimateDim()
    nCols
  }

  @Since("1.3.0")
  val numRowBlocks = math.ceil(numRows() * 1.0 / rowsPerBlock).toInt
  @Since("1.3.0")
  val numColBlocks = math.ceil(numCols() * 1.0 / colsPerBlock).toInt

  private[mllib] def createPartitioner(): GridPartitioner =
    GridPartitioner(numRowBlocks, numColBlocks, suggestedNumPartitions = blocks.partitions.length)

  private lazy val blockInfo = blocks.mapValues(block => (block.numRows, block.numCols)).cache()

  /** Estimates the dimensions of the matrix. */
  private def estimateDim(): Unit = {
    val (rows, cols) = blockInfo.map { case ((blockRowIndex, blockColIndex), (m, n)) =>
      (blockRowIndex.toLong * rowsPerBlock + m,
        blockColIndex.toLong * colsPerBlock + n)
    }.reduce { (x0, x1) =>
      (math.max(x0._1, x1._1), math.max(x0._2, x1._2))
    }
    if (nRows <= 0L) nRows = rows
    assert(rows <= nRows, s"The number of rows $rows is more than claimed $nRows.")
    if (nCols <= 0L) nCols = cols
    assert(cols <= nCols, s"The number of columns $cols is more than claimed $nCols.")
  }

  /**
   * Validates the block matrix info against the matrix data (`blocks`) and throws an exception if
   * any error is found.
   */
  @Since("1.3.0")
  def validate(): Unit = {
    logDebug("Validating BlockMatrix...")
    // check if the matrix is larger than the claimed dimensions
    estimateDim()
    logDebug("BlockMatrix dimensions are okay...")

    // Check if there are multiple MatrixBlocks with the same index.
    blockInfo.countByKey().foreach { case (key, cnt) =>
      if (cnt > 1) {
        throw new SparkException(s"Found multiple MatrixBlocks with the indices $key. Please " +
          "remove blocks with duplicate indices.")
      }
    }
    logDebug("MatrixBlock indices are okay...")
    // Check if each MatrixBlock (except edges) has the dimensions rowsPerBlock x colsPerBlock
    // The first tuple is the index and the second tuple is the dimensions of the MatrixBlock
    val dimensionMsg = s"dimensions different than rowsPerBlock: $rowsPerBlock, and " +
      s"colsPerBlock: $colsPerBlock. Blocks on the right and bottom edges can have smaller " +
      s"dimensions. You may use the repartition method to fix this issue."
    blockInfo.foreach { case ((blockRowIndex, blockColIndex), (m, n)) =>
      if ((blockRowIndex < numRowBlocks - 1 && m != rowsPerBlock) ||
          (blockRowIndex == numRowBlocks - 1 && (m <= 0 || m > rowsPerBlock))) {
        throw new SparkException(s"The MatrixBlock at ($blockRowIndex, $blockColIndex) has " +
          dimensionMsg)
      }
      if ((blockColIndex < numColBlocks - 1 && n != colsPerBlock) ||
        (blockColIndex == numColBlocks - 1 && (n <= 0 || n > colsPerBlock))) {
        throw new SparkException(s"The MatrixBlock at ($blockRowIndex, $blockColIndex) has " +
          dimensionMsg)
      }
    }
    logDebug("MatrixBlock dimensions are okay...")
    logDebug("BlockMatrix is valid!")
  }

  /** Caches the underlying RDD. */
  @Since("1.3.0")
  def cache(): this.type = {
    blocks.cache()
    this
  }

  /** Persists the underlying RDD with the specified storage level. */
  @Since("1.3.0")
  def persist(storageLevel: StorageLevel): this.type = {
    blocks.persist(storageLevel)
    this
  }

  /** Converts to CoordinateMatrix. */
  @Since("1.3.0")
  def toCoordinateMatrix(): CoordinateMatrix = {
    val entryRDD = blocks.flatMap { case ((blockRowIndex, blockColIndex), mat) =>
      val rowStart = blockRowIndex.toLong * rowsPerBlock
      val colStart = blockColIndex.toLong * colsPerBlock
      val entryValues = new ArrayBuffer[MatrixEntry]()
      mat.foreachActive { (i, j, v) =>
        if (v != 0.0) entryValues += new MatrixEntry(rowStart + i, colStart + j, v)
      }
      entryValues
    }
    new CoordinateMatrix(entryRDD, numRows(), numCols())
  }


  /** Converts to IndexedRowMatrix. The number of columns must be within the integer range. */
  @Since("1.3.0")
  def toIndexedRowMatrix(): IndexedRowMatrix = {
    val cols = numCols().toInt

    require(cols < Int.MaxValue, s"The number of columns should be less than Int.MaxValue ($cols).")

    val rows = blocks.flatMap { case ((blockRowIdx, blockColIdx), mat) =>
      mat.rowIter.zipWithIndex.map {
        case (vector, rowIdx) =>
          blockRowIdx * rowsPerBlock + rowIdx -> (blockColIdx, vector.asBreeze)
      }
    }.groupByKey().map { case (rowIdx, vectors) =>
      val numberNonZeroPerRow = vectors.map(_._2.activeSize).sum.toDouble / cols.toDouble

      val wholeVector = if (numberNonZeroPerRow <= 0.1) { // Sparse at 1/10th nnz
        BSV.zeros[Double](cols)
      } else {
        BDV.zeros[Double](cols)
      }

      vectors.foreach { case (blockColIdx: Int, vec: BV[Double]) =>
        val offset = colsPerBlock * blockColIdx
        wholeVector(offset until Math.min(cols, offset + colsPerBlock)) := vec
      }
      new IndexedRow(rowIdx, Vectors.fromBreeze(wholeVector))
    }
    new IndexedRowMatrix(rows)
  }

  /** Collect the distributed matrix on the driver as a `DenseMatrix`. */
  @Since("1.3.0")
  def toLocalMatrix(): Matrix = {
    require(numRows() < Int.MaxValue, "The number of rows of this matrix should be less than " +
      s"Int.MaxValue. Currently numRows: ${numRows()}")
    require(numCols() < Int.MaxValue, "The number of columns of this matrix should be less than " +
      s"Int.MaxValue. Currently numCols: ${numCols()}")
    require(numRows() * numCols() < Int.MaxValue, "The length of the values array must be " +
      s"less than Int.MaxValue. Currently numRows * numCols: ${numRows() * numCols()}")
    val m = numRows().toInt
    val n = numCols().toInt
    val mem = m * n / 125000
    if (mem > 500) logWarning(s"Storing this matrix will require $mem MB of memory!")
    val localBlocks = blocks.collect()
    val values = new Array[Double](m * n)
    localBlocks.foreach { case ((blockRowIndex, blockColIndex), submat) =>
      val rowOffset = blockRowIndex * rowsPerBlock
      val colOffset = blockColIndex * colsPerBlock
      submat.foreachActive { (i, j, v) =>
        val indexOffset = (j + colOffset) * m + rowOffset + i
        values(indexOffset) = v
      }
    }
    new DenseMatrix(m, n, values)
  }

  /**
   * Transpose this `BlockMatrix`. Returns a new `BlockMatrix` instance sharing the
   * same underlying data. Is a lazy operation.
   */
  @Since("1.3.0")
  def transpose: BlockMatrix = {
    val transposedBlocks = blocks.map { case ((blockRowIndex, blockColIndex), mat) =>
      ((blockColIndex, blockRowIndex), mat.transpose)
    }
    new BlockMatrix(transposedBlocks, colsPerBlock, rowsPerBlock, nCols, nRows)
  }

  /** Collects data and assembles a local dense breeze matrix (for test only). */
  private[mllib] def toBreeze(): BDM[Double] = {
    val localMat = toLocalMatrix()
    new BDM[Double](localMat.numRows, localMat.numCols, localMat.toArray)
  }

  /**
   * For given matrices `this` and `other` of compatible dimensions and compatible block dimensions,
   * it applies a binary function on their corresponding blocks.
   *
   * @param other The second BlockMatrix argument for the operator specified by `binMap`
   * @param binMap A function taking two breeze matrices and returning a breeze matrix
   * @return A [[BlockMatrix]] whose blocks are the results of a specified binary map on blocks
   *         of `this` and `other`.
   * Note: `blockMap` ONLY works for `add` and `subtract` methods and it does not support
   * operators such as (a, b) => -a + b
   * TODO: Make the use of zero matrices more storage efficient.
   */
  private[mllib] def blockMap(
      other: BlockMatrix,
      binMap: (BM[Double], BM[Double]) => BM[Double]): BlockMatrix = {
    require(numRows() == other.numRows(), "Both matrices must have the same number of rows. " +
      s"A.numRows: ${numRows()}, B.numRows: ${other.numRows()}")
    require(numCols() == other.numCols(), "Both matrices must have the same number of columns. " +
      s"A.numCols: ${numCols()}, B.numCols: ${other.numCols()}")
    if (rowsPerBlock == other.rowsPerBlock && colsPerBlock == other.colsPerBlock) {
      val newBlocks = blocks.cogroup(other.blocks, createPartitioner())
        .map { case ((blockRowIndex, blockColIndex), (a, b)) =>
          if (a.size > 1 || b.size > 1) {
            throw new SparkException("There are multiple MatrixBlocks with indices: " +
              s"($blockRowIndex, $blockColIndex). Please remove them.")
          }
          if (a.isEmpty) {
            val zeroBlock = BM.zeros[Double](b.head.numRows, b.head.numCols)
            val result = binMap(zeroBlock, b.head.asBreeze)
            new MatrixBlock((blockRowIndex, blockColIndex), Matrices.fromBreeze(result))
          } else if (b.isEmpty) {
            new MatrixBlock((blockRowIndex, blockColIndex), a.head)
          } else {
            val result = binMap(a.head.asBreeze, b.head.asBreeze)
            new MatrixBlock((blockRowIndex, blockColIndex), Matrices.fromBreeze(result))
          }
      }
      new BlockMatrix(newBlocks, rowsPerBlock, colsPerBlock, numRows(), numCols())
    } else {
      throw new SparkException("Cannot perform on matrices with different block dimensions")
    }
  }

  /**
   * Adds the given block matrix `other` to `this` block matrix: `this + other`.
   * The matrices must have the same size and matching `rowsPerBlock` and `colsPerBlock`
   * values. If one of the blocks that are being added are instances of [[SparseMatrix]],
   * the resulting sub matrix will also be a [[SparseMatrix]], even if it is being added
   * to a [[DenseMatrix]]. If two dense matrices are added, the output will also be a
   * [[DenseMatrix]].
   */
  @Since("1.3.0")
  def add(other: BlockMatrix): BlockMatrix =
    blockMap(other, (x: BM[Double], y: BM[Double]) => x + y)

  /**
   * Subtracts the given block matrix `other` from `this` block matrix: `this - other`.
   * The matrices must have the same size and matching `rowsPerBlock` and `colsPerBlock`
   * values. If one of the blocks that are being subtracted are instances of [[SparseMatrix]],
   * the resulting sub matrix will also be a [[SparseMatrix]], even if it is being subtracted
   * from a [[DenseMatrix]]. If two dense matrices are subtracted, the output will also be a
   * [[DenseMatrix]].
   */
  @Since("2.0.0")
  def subtract(other: BlockMatrix): BlockMatrix =
    blockMap(other, (x: BM[Double], y: BM[Double]) => x - y)

  /** Block (i,j) --> Set of destination partitions */
  private type BlockDestinations = Map[(Int, Int), Set[Int]]

  /**
   * Simulate the multiplication with just block indices in order to cut costs on communication,
   * when we are actually shuffling the matrices.
   * The `colsPerBlock` of this matrix must equal the `rowsPerBlock` of `other`.
   * Exposed for tests.
   *
   * @param other The BlockMatrix to multiply
   * @param partitioner The partitioner that will be used for the resulting matrix `C = A * B`
   * @return A tuple of [[BlockDestinations]]. The first element is the Map of the set of partitions
   *         that we need to shuffle each blocks of `this`, and the second element is the Map for
   *         `other`.
   */
  private[distributed] def simulateMultiply(
      other: BlockMatrix,
      partitioner: GridPartitioner): (BlockDestinations, BlockDestinations) = {
    val leftMatrix = blockInfo.keys.collect() // blockInfo should already be cached
    val rightMatrix = other.blocks.keys.collect()

    val rightCounterpartsHelper = rightMatrix.groupBy(_._1).mapValues(_.map(_._2))
    val leftDestinations = leftMatrix.map { case (rowIndex, colIndex) =>
      val rightCounterparts = rightCounterpartsHelper.getOrElse(colIndex, Array.empty[Int])
      val partitions = rightCounterparts.map(b => partitioner.getPartition((rowIndex, b)))
      ((rowIndex, colIndex), partitions.toSet)
    }.toMap

    val leftCounterpartsHelper = leftMatrix.groupBy(_._2).mapValues(_.map(_._1))
    val rightDestinations = rightMatrix.map { case (rowIndex, colIndex) =>
      val leftCounterparts = leftCounterpartsHelper.getOrElse(rowIndex, Array.empty[Int])
      val partitions = leftCounterparts.map(b => partitioner.getPartition((b, colIndex)))
      ((rowIndex, colIndex), partitions.toSet)
    }.toMap

    (leftDestinations, rightDestinations)
  }

  /**
   * Left multiplies this [[BlockMatrix]] to `other`, another [[BlockMatrix]]. The `colsPerBlock`
   * of this matrix must equal the `rowsPerBlock` of `other`. If `other` contains
   * [[SparseMatrix]], they will have to be converted to a [[DenseMatrix]]. The output
   * [[BlockMatrix]] will only consist of blocks of [[DenseMatrix]]. This may cause
   * some performance issues until support for multiplying two sparse matrices is added.
   *
   * Note: The behavior of multiply has changed in 1.6.0. `multiply` used to throw an error when
   * there were blocks with duplicate indices. Now, the blocks with duplicate indices will be added
   * with each other.
   */
  @Since("1.3.0")
  def multiply(other: BlockMatrix): BlockMatrix = {
    require(numCols() == other.numRows(), "The number of columns of A and the number of rows " +
      s"of B must be equal. A.numCols: ${numCols()}, B.numRows: ${other.numRows()}. If you " +
      "think they should be equal, try setting the dimensions of A and B explicitly while " +
      "initializing them.")
    if (colsPerBlock == other.rowsPerBlock) {
      val resultPartitioner = GridPartitioner(numRowBlocks, other.numColBlocks,
        math.max(blocks.partitions.length, other.blocks.partitions.length))
      val (leftDestinations, rightDestinations) = simulateMultiply(other, resultPartitioner)
      // Each block of A must be multiplied with the corresponding blocks in the columns of B.
      val flatA = blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = leftDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
      }
      // Each block of B must be multiplied with the corresponding blocks in each row of A.
      val flatB = other.blocks.flatMap { case ((blockRowIndex, blockColIndex), block) =>
        val destinations = rightDestinations.getOrElse((blockRowIndex, blockColIndex), Set.empty)
        destinations.map(j => (j, (blockRowIndex, blockColIndex, block)))
      }
      val newBlocks = flatA.cogroup(flatB, resultPartitioner).flatMap { case (pId, (a, b)) =>
        a.flatMap { case (leftRowIndex, leftColIndex, leftBlock) =>
          b.filter(_._1 == leftColIndex).map { case (rightRowIndex, rightColIndex, rightBlock) =>
            val C = rightBlock match {
              case dense: DenseMatrix => leftBlock.multiply(dense)
              case sparse: SparseMatrix => leftBlock.multiply(sparse.toDense)
              case _ =>
                throw new SparkException(s"Unrecognized matrix type ${rightBlock.getClass}.")
            }
            ((leftRowIndex, rightColIndex), C.asBreeze)
          }
        }
      }.reduceByKey(resultPartitioner, (a, b) => a + b).mapValues(Matrices.fromBreeze)
      // TODO: Try to use aggregateByKey instead of reduceByKey to get rid of intermediate matrices
      new BlockMatrix(newBlocks, rowsPerBlock, other.colsPerBlock, numRows(), other.numCols())
    } else {
      throw new SparkException("colsPerBlock of A doesn't match rowsPerBlock of B. " +
        s"A.colsPerBlock: $colsPerBlock, B.rowsPerBlock: ${other.rowsPerBlock}")
    }
  }

  def iterativeRefinemet(B: BlockMatrix): BlockMatrix = {
    // Compute Gram matrix of B.
    val gramMat = B.toIndexedRowMatrix().toRowMatrix().
      computeGramianMatrix().asBreeze.toDenseMatrix
    // compute the eigenvalue decomposition of gramMat
    // such that gramMat = U * D * U'.
    val EigSym(eigenValues, eigenVectors) = eigSym(gramMat)
    // println("EigenValues")
    // println(eigenValues)

    val eigenRank = {
      var i = eigenValues.length - 1
      while (i >= 0 && eigenValues(i) > 1.0e-15 *
        eigenValues(eigenValues.length - 1)) i = i - 1
      i + 1
    }

    // val eigenRank = 0
    // println("EigenRank")
    // println(eigenValues.length - eigenRank)
    val EigenVectors = eigenVectors(::, eigenVectors.cols - 1 to eigenRank by -1)
    // Calculate Q such that Q = B * U and normalize Q such that each column of
    // Q has norm 1.

    val U = Matrices.dense(EigenVectors.rows, EigenVectors.cols,
      eigenVectors(::, eigenVectors.cols - 1 to eigenRank by -1).toArray)

    val Q = B.toIndexedRowMatrix().multiply(U)
    val normQ = Vectors.fromBreeze(1.0 / Statistics.colStats(Q.toRowMatrix().rows).
      normL2.asBreeze.toDenseVector)
    Q.multiply(Matrices.diag(normQ)).toBlockMatrix(B.rowsPerBlock, B.colsPerBlock)
  }

  /**
    * Computes the randomized singular value decomposition of this BlockMatrix.
    * Denote this matrix by A (m x n), this will compute matrices U, S, V such
    * that A = U * S * V', where the columns of U are orthonormal, S is a
    * diagonal matrix with non-negative real numbers on the diagonal, and the
    * columns of V are orthonormal.
    *
    * At most k largest non-zero singular values and associated vectors are
    * returned. If there are k such values, then the dimensions of the return
    * will be:
    *  - U is a RowMatrix of size m x k that satisfies U' * U = eye(k),
    *  - s is a Vector of size k, holding the singular values in
    *  descending order,
    *  - V is a Matrix of size n x k that satisfies V' * V = eye(k).
    *
    * @param k number of singular values to keep. We might return less than k
    *          if there are numerically zero singular values.
    * @param sc SparkContext, use to generate the random gaussian matrix.
    * @param computeU whether to compute U.
    * @param isQR whether to use tallSkinnyQR or Cholesky decomposition and
    *             a forward solve for matrix orthonormalization.
    * @param iteration number of normalized power iterations to conduct.
    * @param isRandom whether or not fix seed to generate random matrix.
    * @return SingularValueDecomposition(U, s, V).
    */
  @Since("2.0.0")
  def partialSVD(k: Int, sc: SparkContext, computeU: Boolean = false,
                 isQR: Boolean = true, iteration: Int = 2,
                 isRandom: Boolean = true):
  SingularValueDecomposition[BlockMatrix, Matrix] = {

    /**
      * Generate a random [[BlockMatrix]] to compute the singular
      * value decomposition.
      *
      * @param k number of columns in this random [[BlockMatrix]].
      * @param sc a [[SparkContext]] to generate the random [[BlockMatrix]].
      * @return a random [[BlockMatrix]].
      *
      * @note the generated random [[BlockMatrix]] V has colsPerBlock for the
      *       number of rows in each block, and rowsPerBlock for the number
      *       of columns in each block. We will perform matrix multiplication
      *       with A, i.e., A * V. We want the number of rows in each block
      *       of V to be same as the number of columns in each block A.
      */
    def generateRandomMatrices(k: Int, sc: SparkContext): BlockMatrix = {
      val limit = 65535
      if (numCols().toInt < limit) {
        val data = Seq.tabulate(numCols().toInt)(n =>
          (n.toLong, Vectors.fromBreeze(BDV.rand(k))))
          .map(x => IndexedRow(x._1, x._2))
        val indexedRows: RDD[IndexedRow] = sc.parallelize(data,
          createPartitioner().numPartitions)
        new IndexedRowMatrix(indexedRows).
          toBlockMatrix(colsPerBlock, rowsPerBlock)
      } else {
        // Generate a n-by-k BlockMatrix: we first generate a sequence of
        // RDD[Vector] where each RDD[Vector] contains either 65535 rows or
        // n mod 65535 rows. The number of elements in the sequence is
        // ceiling(n/65535).
        val num = ceil(numCols().toFloat/limit).toInt
        val rddsBlock = Seq.tabulate(num)(m =>
          sc.parallelize(Seq.tabulate(if (m < numCols/limit) limit
          else numCols().toInt%limit)(n => (m.toLong * limit + n,
            Vectors.fromBreeze(BDV.rand(k))))
            .map(x => IndexedRow(x._1, x._2)),
            createPartitioner().numPartitions))
        val rddBlockSeq = sc.union(rddsBlock)
        new IndexedRowMatrix(rddBlockSeq).
          toBlockMatrix(colsPerBlock, rowsPerBlock)
      }
    }



    /**
      * Computes the partial singular value decomposition of the[[BlockMatrix]]
      * A given an [[BlockMatrix]] Q such that A' is close to A' * Q * Q'.
      * The columns of Q are orthonormal.
      *
      * @param Q a [[BlockMatrix]] with orthonormal columns.
      * @param k number of singular values to compute.
      * @param computeU whether to compute U.
      * @param isQR whether to use tallSkinnySVD.
      * @return SingularValueDecomposition[U, s, V], U = null
      *         if computeU = false.
      *
      * @note if isQR is false, then we find SVD of B = X * S * V' via the
      *       following steps: Compute gram matrix G of B; find eigenvalue
      *       decomposition of G such that G = U * D * U'; calculate
      *       R = U * sqrt(D); find singular value decomposition of R such that
      *       R = Y * S * V'; calculate Q such that B = Q * R = Q * Y * S * V'
      *       = X * S * V' where X = Q * Y.
      */
    def lastStep(Q: BlockMatrix, k: Int, computeU: Boolean,
                 isQR: Boolean, sc: SparkContext):
    SingularValueDecomposition[BlockMatrix, Matrix] = {
      // B = A' * Q
      // val Q1 = iterativeRefinemet(Q)
      val Q1 = Q
      val B = transpose.multiply(Q1)
      // Find SVD such that B = X * S * V'.
      val (svdResult, indices) = if (isQR) {
        (B.toIndexedRowMatrix().toRowMatrix().tallSkinnySVD(sc, k,
          computeU = true, rCond = 1.0e-15), B.toIndexedRowMatrix().rows.map(_.index))
      } else {
        // val QofB = iterativeRefinemet(iterativeRefinemet(B))
        val QofB = iterativeRefinemet(B)
        val R = B.transpose.multiply(QofB).transpose
        // Compute svd of R such that R = W * S * V'.
        val brzSvd.SVD(w, s, vt) = brzSvd.reduced.apply(R.toBreeze())
        val WMat = Matrices.fromBreeze(w)
        val sk = Vectors.fromBreeze(s)
        val VMat = Matrices.fromBreeze(vt).transpose
        // Return svd of B.
        (SingularValueDecomposition(QofB.toIndexedRowMatrix().toRowMatrix().
          multiply(WMat), sk, VMat), QofB.toIndexedRowMatrix().rows.map(_.index))
      }

      val indexedRows = indices.zip(svdResult.U.rows).map { case (i, v) =>
        IndexedRow(i, v) }
      // U = Q * X and V = W and convert type
      val VMat = new IndexedRowMatrix(indexedRows, B.numRows().toInt,
        svdResult.s.size)
      val V = Matrices.fromBreeze(VMat.toBreeze())

      if (computeU) {
        val XMat = svdResult.V
        val U = Q1.toIndexedRowMatrix().multiply(XMat).toBlockMatrix()
        SingularValueDecomposition(U, svdResult.s, V)
      } else {
        SingularValueDecomposition(null, svdResult.s, V)
      }
    }

    // Whether to set the random seed or not. Set the seed would help debug.
    if (!isRandom) {
      Random.setSeed(513427689.toLong)
    }
    val V = generateRandomMatrices(k, sc)
    // V = A * V, with the V on the left now known as x.
    val x = multiply(V)
    // Orthonormalize V (now known as x).
    var y = x.orthonormal(isQR, isTranspose = true, sc)

    for (i <- 0 until iteration) {
      // V = A' * V,  with the V on the left now known as a, and the V on
      // the right known as y.
      val a = transpose.multiply(y)
      // Orthonormalize V (now known as a).
      val b = a.orthonormal(isQR, isTranspose = false, sc)
      // V = A * V, with the V on the left now known as c, and the V on
      // the right known as b.
      val c = multiply(b)
      // Orthonormalize V (now known as c).
      y = c.orthonormal(isQR, isTranspose = true, sc)
    }
    // Find SVD of A using V (now known as y).
    lastStep(y, k, computeU, isQR, sc)
  }


  /**
    * Orthonormalize the columns of the [[BlockMatrix]] V by using either QR
    * decomposition or Cholesky decomposition and forward solve. We convert V
    * to [[RowMatrix]] first, then either (1) directly apply tallSkinnyQR or
    * (2) compute the Gram matrix G of V, then apply Cholesky decomposition
    * on G to get the upper triangular matrix R, and apply forward solve to
    * find Q such that Q * R = V where the columns of Q are orthonormal (so
    * that Q' * Q is the identity matrix).
    *
    * @param isQR whether to use tallSkinnyQR or Cholesky decomposition and
    *             a forward solve for matrix orthonormalization.
    * @param isTranspose whether to switch colsPerBlock and rowsPerBlock.
    * @return a [[BlockMatrix]] whose columns are orthonormal vectors.
    *
    * @note if isQR is false, it will lose half or more of the precision
    * of the arithmetic but could accelerate the computation.
    */
  @Since("2.0.0")
  def orthonormal(isQR: Boolean = true, isTranspose: Boolean = false,
                  sc: SparkContext): BlockMatrix = {
    /**
      * Convert [[Matrix]] to [[RDD[Vector]]]
      * @param mat an [[Matrix]].
      * @param sc SparkContext used to create RDDs.
      * @return RDD[Vector].
      */
    def toRDD(mat: Matrix, sc: SparkContext, numPar: Int): RDD[Vector] = {
      // Convert mat to Sequence of DenseVector
      val columns = mat.toArray.grouped(mat.numRows)
      val rows = columns.toSeq.transpose
      val vectors = rows.map(row => new DenseVector(row.toArray))
      // Create RDD[Vector]
      sc.parallelize(vectors, numPar)
    }

    /**
      * Solve Q*R = A for Q using forward substitution where
      * A = [[IndexedRowMatrix]] and R is upper-triangular.
      *
      * @param A [[IndexedRowMatrix]].
      * @param R upper-triangular matrix.
      * @return Q [[IndexedRowMatrix]] such that Q*R = A.
      */
    def forwardSolve(A: IndexedRowMatrix, R: BDM[Double]):
    IndexedRowMatrix = {
      // Convert A to RowMatrix.
      val rowMat = A.toRowMatrix()
      val Bb = rowMat.rows.context.broadcast(R.toArray)
      // Solving Q by using forward substitution.
      val AB = rowMat.rows.mapPartitions { iter =>
        val Bi = Bb.value
        iter.map { row =>
          val info = new intW(0)
          val B = row.asBreeze.toArray
          lapack.dtrtrs("L", "N", "N", R.cols, 1, Bi, R.cols, B, R.cols, info)
          Vectors.dense(B)
        }
      }
      // Form Q as RowMatrix and convert it to IndexedRowMatrix.
      val QRow = new RowMatrix(AB, numRows().toInt, R.cols)
      val indexedRows = A.rows.map(_.index).
        zip(QRow.rows).map { case (i, v) => IndexedRow(i, v) }
      new IndexedRowMatrix(indexedRows, numRows().toInt, R.cols)
    }

    // Orthonormalize the columns of the input BlockMatrix.
    val Q = if (isQR) {
      // Convert to RowMatrix and apply tallSkinnyQR.
      val indices = toIndexedRowMatrix().rows.map(_.index)
      val qrResult = toIndexedRowMatrix().toRowMatrix().tallSkinnyQR(true)
      // Convert Q to IndexedRowMatrix
      val indexedRows = indices.zip(qrResult.Q.rows).map { case (i, v) =>
        IndexedRow(i, v)}
      new IndexedRowMatrix(indexedRows, numRows().toInt, numCols().toInt)
    } else {
      /*
      println("Matrix need to be orthogonal")
      for (i <- 0 until numRows().toInt) {
        val lala = toBreeze()
        println(lala(i, ::).t.toArray.mkString(" "))
      }

      println("Singular Values")
      val brzSvd.SVD(_, s, _) = brzSvd.reduced.apply(this.toBreeze())
      println(s)
      */
      val GramQR = toIndexedRowMatrix().toRowMatrix().tallSkinnyQR(true).Q.computeGramianMatrix()
      // val Q = iterativeRefinemet(iterativeRefinemet(this)).toIndexedRowMatrix()
      val Q = iterativeRefinemet(this).toIndexedRowMatrix()
      /*
      println("Gram Matrix sum of diagonal entries using tallSkinnyQR")
      println(trace(GramQR.asBreeze.toDenseMatrix))
      println("Gram Matrix sum of diagonal entries")
      println(trace(Q.computeGramianMatrix().asBreeze.toDenseMatrix))

      println("Gram Matrix sum of non-diagonal entries using tallSkinnyQR")
      println(sum(GramQR.asBreeze.toDenseMatrix) -
        trace(GramQR.asBreeze.toDenseMatrix))
      println("Gram Matrix sum of non-diagonal entries")
      println(sum(Q.computeGramianMatrix().asBreeze.toDenseMatrix) -
        trace(Q.computeGramianMatrix().asBreeze.toDenseMatrix))
      */
      // Convert Q to IndexedRowMatrix
      val indices = Q.rows.map(_.index)
      val indexedRows = indices.zip(Q.toRowMatrix().rows).map { case (i, v) =>
        IndexedRow(i, v)}
      new IndexedRowMatrix(indexedRows, numRows().toInt, numCols().toInt)

      /*
      // Compute the Gram matrix of A
      val gramMat = toIndexedRowMatrix().toRowMatrix().computeGramianMatrix()
      val CholeskyRegularization = 1.0e-8 * Vectors.norm(
        Vectors.dense(gramMat.toArray), 2.0)
      // Add some small values to the diagonal of Gram matrix such
      // that it is strictly positive-definite.
      val B = gramMat.asBreeze.toDenseMatrix + CholeskyRegularization *
        BDM.eye[Double](gramMat.numCols)
      // Cholesky decomposition
      val R = cholesky((B + B.t) * .5)
      // Forward solve
      forwardSolve(toIndexedRowMatrix(), R)
      */
    }
    // Convert Q to BlockMatrix.
    if (isTranspose) {
      Q.toBlockMatrix(colsPerBlock, rowsPerBlock)
    } else {
      Q.toBlockMatrix(rowsPerBlock, colsPerBlock)
    }
  }

  /**
    * Estimate the largest singular value of [[BlockMatrix]] A using
    * power method.
    *
    * @param iteration number of iterations for power method.
    * @param sc a [[SparkContext]] generates the normalized
    *           vector in each iteration.
    * @return a [[Double]] estimate of the largest singular value.
    */
  @Since("2.0.0")
  def spectralNormEst(iteration: Int = 20, sc: SparkContext): Double = {
    /**
      * Normalize the [[BlockMatrix]] v which has one column such that it has
      * unit norm.
      *
      * @param v the [[BlockMatrix]] which has one column.
      * @param sc SparkContext, use to generate the normalized [[BlockMatrix]].
      * @param isTranspose determines whether to partition v commensurate with
      *                    multiplication by A or with A'.
      * @return a [[BlockMatrix]] such that it has unit norm.
      */
    def unit(v : BlockMatrix, sc: SparkContext, isTranspose: Boolean = false):
    BlockMatrix = {
      // v = v / norm(v).
      val temp = Vectors.dense(v.toBreeze().toArray)
      val vUnit = temp.asBreeze * (1.0 / Vectors.norm(temp, 2))
      // Convert v back to BlockMatrix.
      val indexedRow = Seq.tabulate(v.numRows().toInt)(n => (n.toLong,
        Vectors.dense(vUnit(n)))).map(x => IndexedRow(x._1, x._2))
      val data = new IndexedRowMatrix(sc.parallelize(indexedRow))
      // isTranspose determines whether to partition v commensurate with
      // multiplication by A or with A'. We will perform matrix multiplication
      // with A or A', i.e., A * v or A' * v. We want the number of rows in
      // each block of v to be same as the number of columns in each block A.
      if (isTranspose) {
        data.toBlockMatrix(colsPerBlock, 1)
      } else {
        data.toBlockMatrix(rowsPerBlock, 1)
      }
    }
    // Generate a random vector v.
    var v = {
      val data = Seq.tabulate(numCols().toInt)(n =>
        (n.toLong, Vectors.fromBreeze(BDV.rand(1))))
        .map(x => IndexedRow(x._1, x._2))
      val indexedRows: RDD[IndexedRow] = sc.parallelize(data,
        createPartitioner().numPartitions)
      // v has colsPerBlock for the number of rows in each block, and 1
      // as the number of columns in each block. We will
      // perform matrix multiplication with A, i.e., A * v. We want the number
      // of rows in each block of v to be same as the number of columns in
      // each block A.
      new IndexedRowMatrix(indexedRows).toBlockMatrix(colsPerBlock, 1)
    }
    // Find the largest singular value of A using power method.
    for (i <- 0 until iteration) {
      // normalize v.
      v = unit(v, sc, isTranspose = true)
      // v = A * v.
      val Av = multiply(v)
      // normalize v.
      v = unit(Av, sc, isTranspose = false)
      // v = A' * v.
      v = transpose.multiply(v)
    }
    // Calculate the 2-norm of final v.
    Vectors.norm(Vectors.dense(v.toBreeze().toArray), 2)
  }
}
