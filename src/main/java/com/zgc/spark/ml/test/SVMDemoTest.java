package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

/**
 * 支持向量机 预测胃癌转移
 *
 * @author guocheng.zhao
 * @date 2017/4/6 13:33
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */

public class SVMDemoTest {
    private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(SVMDemoTest.class);

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("LogisticRegressionWithGSDTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        List<LabeledPoint> dataTraining = Arrays.asList(
                new LabeledPoint(0, Vectors.dense(59, 2, 43.4, 2, 1)),
                new LabeledPoint(0, Vectors.dense(36, 1, 57.2, 1, 1)),
                new LabeledPoint(0, Vectors.dense(61, 2, 190, 2, 1)),
                new LabeledPoint(1, Vectors.dense(58, 3, 128, 4, 3)),
                new LabeledPoint(1, Vectors.dense(55, 3, 128, 4, 3)),
                new LabeledPoint(0, Vectors.dense(61, 1, 94, 4, 2)),
                new LabeledPoint(0, Vectors.dense(38, 1, 76, 1, 1)),
                new LabeledPoint(0, Vectors.dense(42, 1, 240, 3, 2)),
                new LabeledPoint(0, Vectors.dense(50, 1, 74, 1, 1)),
                new LabeledPoint(0, Vectors.dense(58, 3, 68.6, 2, 2)),
                new LabeledPoint(0, Vectors.dense(68, 3, 132.8, 4, 2)),
                new LabeledPoint(1, Vectors.dense(25, 2, 94.6, 4, 3)),
                new LabeledPoint(0, Vectors.dense(52, 1, 56, 1, 1)),
                new LabeledPoint(0, Vectors.dense(31, 1, 47.8, 2, 1)),
                new LabeledPoint(1, Vectors.dense(36, 3, 31.6, 3, 1)),
                new LabeledPoint(0, Vectors.dense(42, 1, 66.2, 2, 1)),
                new LabeledPoint(1, Vectors.dense(14, 3, 138.6, 3, 3)),
                new LabeledPoint(0, Vectors.dense(32, 1, 114, 2, 3)),
                new LabeledPoint(0, Vectors.dense(35, 1, 40.2, 2, 1)),
                new LabeledPoint(1, Vectors.dense(70, 3, 177.2, 4, 3)),
                new LabeledPoint(1, Vectors.dense(65, 2, 51.6, 4, 4)),
                new LabeledPoint(0, Vectors.dense(45, 2, 124, 2, 4)),
                new LabeledPoint(1, Vectors.dense(68, 3, 127.2, 3, 3)),
                new LabeledPoint(0, Vectors.dense(31, 2, 124.8, 2, 3))

        );
        SparkContext sc = new SparkContext(conf);
        JavaSparkContext jsc = new JavaSparkContext(sc);
        RDD<LabeledPoint> labeledPointRDD = jsc.parallelize(dataTraining).rdd();
        //支持向量机
        SVMModel svmModel = SVMWithSGD.train(labeledPointRDD, 10);
        System.out.println("svmModel.predict(Vectors.dense(70, 3, 180.0, 4, 3)):"+svmModel.predict(Vectors.dense(70, 3, 180.0, 4, 3)));
        System.out.println("权重svmModel.weights():" + svmModel.weights());
        System.out.println("截距svmModel.intercept():" + svmModel.intercept());
    }
}
