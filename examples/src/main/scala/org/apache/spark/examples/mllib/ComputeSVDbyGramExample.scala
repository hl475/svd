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
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
// $example on$
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.linalg.Vectors
// $example off$

object ComputeSVDbyGramExample {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAppName("TallSkinnySVDExample")
    val sc = new SparkContext(conf)

    // $example on$
    val data = Seq(
      (0L, Vectors.dense(0.0, 1.0, 2.0)),
      (1L, Vectors.dense(3.0, 4.0, 5.0)),
      (3L, Vectors.dense(9.0, 0.0, 1.0))
    ).map(x => IndexedRow(x._1, x._2))

    val indexedRows = sc.parallelize(data, 2)

    val mat = new IndexedRowMatrix(indexedRows)

    // Compute the singular value decompositions of mat.
    val svd: SingularValueDecomposition[IndexedRowMatrix, Matrix] =
      mat.computeSVDbyGram(computeU = true)
    val U: IndexedRowMatrix = svd.U  // The U factor is a RowMatrix.
    val s: Vector = svd.s  // The singular values are stored in a local dense vector.
    val V: Matrix = svd.V  // The V factor is a local dense matrix.
    // $example off$
    val collect = U.rows.collect()
    println("U factor is:")
    collect.foreach { vector => println(vector) }
    println(s"Singular values are: $s")
    println(s"V factor is:\n$V")
  }
}
// scalastyle:on println
