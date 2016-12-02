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

package org.apache.spark.sql.execution.streaming

import org.apache.spark.TaskContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, ForeachWriter}
import org.apache.spark.sql.catalyst.plans.logical.CatalystSerde

/**
 * A [[Sink]] that forwards all data into [[ForeachWriter]] according to the contract defined by
 * [[ForeachWriter]].
 *
 * @param writer The [[ForeachWriter]] to process all data.
 * @tparam T The expected type of the sink.
 */
class ForeachSink[T : Encoder](writer: ForeachWriter[T]) extends Sink with Serializable {

  override def addBatch(batchId: Long, data: DataFrame): Unit = {
    // TODO: Refine this method when SPARK-16264 is resolved; see comments below.

    // This logic should've been as simple as:
    // ```
    //   data.as[T].foreachPartition { iter => ... }
    // ```
    //
    // Unfortunately, doing that would just break the incremental planing. The reason is,
    // `Dataset.foreachPartition()` would further call `Dataset.rdd()`, but `Dataset.rdd()` just
    // does not support `IncrementalExecution`.
    //
    // So as a provisional fix, below we've made a special version of `Dataset` with its `rdd()`
    // method supporting incremental planning. But in the long run, we should generally make newly
    // created Datasets use `IncrementalExecution` where necessary (which is SPARK-16264 tries to
    // resolve).
    val incrementalExecution = data.queryExecution.asInstanceOf[IncrementalExecution]
    val datasetWithIncrementalExecution =
      new Dataset(data.sparkSession, incrementalExecution, implicitly[Encoder[T]]) {
        override lazy val rdd: RDD[T] = {
          val objectType = exprEnc.deserializer.dataType
          val deserialized = CatalystSerde.deserialize[T](logicalPlan)

          // was originally: sparkSession.sessionState.executePlan(deserialized) ...
          val newIncrementalExecution = new IncrementalExecution(
            this.sparkSession,
            deserialized,
            incrementalExecution.outputMode,
            incrementalExecution.checkpointLocation,
            incrementalExecution.currentBatchId,
            incrementalExecution.currentEventTimeWatermark)
          newIncrementalExecution.toRdd.mapPartitions { rows =>
            rows.map(_.get(0, objectType))
          }.asInstanceOf[RDD[T]]
        }
      }
    datasetWithIncrementalExecution.foreachPartition { iter =>
      if (writer.open(TaskContext.getPartitionId(), batchId)) {
        try {
          while (iter.hasNext) {
            writer.process(iter.next())
          }
        } catch {
          case e: Throwable =>
            writer.close(e)
            throw e
        }
        writer.close(null)
      } else {
        writer.close(null)
      }
    }
  }
}
