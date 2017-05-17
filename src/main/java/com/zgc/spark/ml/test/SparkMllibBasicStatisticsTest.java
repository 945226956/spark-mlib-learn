package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.*;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import java.util.Arrays;
import java.util.List;

/**
 * spark mlib 基础统计工具类学习
 *
 * @author guocheng.zhao
 * @date 2017/5/16 11:56
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class SparkMllibBasicStatisticsTest {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("logistic")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        SparkContext sc = new SparkContext(sparkConf);
        JavaSparkContext jsc = new JavaSparkContext("local", "First Spark App", sparkConf);
        SummaryStatistics(jsc);
    }

    /**
     * 汇总统计
     *
     * @param jsc
     */
    private static void SummaryStatistics(JavaSparkContext jsc) {
        JavaRDD<Vector> mat = jsc.parallelize(
                Arrays.asList(
                        Vectors.dense(1.0, 10.0, 100.0),
                        Vectors.dense(2.0, 20.0, 200.0),
                        Vectors.dense(3.0, 30.0, 300.0)
                )
        );
        MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
        System.out.println("summary.count():"+summary.count());
        System.out.println("summary.max():"+summary.max());
        System.out.println("summary.min():"+summary.min());
        System.out.println(summary.mean());  // a dense vector containing the mean value for each column
        //
        System.out.println(summary.variance());  // column-wise variance
        System.out.println(summary.numNonzeros());  // number of nonzeros in each column

    }
}
