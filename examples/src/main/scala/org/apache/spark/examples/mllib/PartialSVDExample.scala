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

// scalastyle:off println
package org.apache.spark.examples.mllib

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.DenseMatrix
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
// $example on$
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
// $example off$

object PartialSVDExample {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("PartialSVDExample")
    val sc = new SparkContext(conf)

    // $example on$
    val blocks: Seq[((Int, Int), Matrix)] = Seq(
      ((0, 0), new DenseMatrix(2, 2, Array(1.0, 0.0, 0.0, 2.0))),
      ((0, 1), new DenseMatrix(2, 2, Array(0.0, 1.0, 0.0, 0.0))),
      ((1, 0), new DenseMatrix(2, 2, Array(3.0, 0.0, 1.0, 1.0))),
      ((1, 1), new DenseMatrix(2, 2, Array(1.0, 2.0, 0.0, 0.0))),
      ((2, 1), new DenseMatrix(1, 2, Array(1.0, 1.0))))

    val mat = new BlockMatrix(sc.parallelize(blocks, 2), 2, 2)

    // Compute the top 4 singular values and corresponding singular vectors.
    val svd: SingularValueDecomposition[BlockMatrix, Matrix] =
      mat.partialSVD(4, sc, computeU = true)
    val U: BlockMatrix = svd.U  // The U factor is a RowMatrix.
    val s: Vector = svd.s  // The singular values are stored in a local dense vector.
    val V: Matrix = svd.V  // The V factor is a local dense matrix.
    // $example off$
    val collect = U.toIndexedRowMatrix().toRowMatrix().rows.collect()
    println("U factor is:")
    collect.foreach { vector => println(vector) }
    println(s"Singular values are: $s")
    println(s"V factor is:\n$V")
  }
}
// scalastyle:on println
