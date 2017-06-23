package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.rdd.RDD;

import java.util.Arrays;
import java.util.List;

/**
 * Linear Regression
 *
 * @author guocheng.zhao
 * @date 2017/4/6 13:33
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */

public class LinearRegressionTest {
    private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(LinearRegressionTest.class);

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("LinearRegressionTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        SparkContext sc = new SparkContext(conf);
        // Prepare training data.
        List<LabeledPoint> dataTraining = Arrays.asList(
                new LabeledPoint(65, Vectors.dense(7,400)),
                new LabeledPoint(90, Vectors.dense(5,1300)),
                new LabeledPoint(100, Vectors.dense(4,1100)),
                new LabeledPoint(110, Vectors.dense(3,1300)),
                new LabeledPoint(60, Vectors.dense(9,300)),
                new LabeledPoint(100, Vectors.dense(5,1000)),
                new LabeledPoint(75, Vectors.dense(7,600)),
                new LabeledPoint(80, Vectors.dense(6,1200)),
                new LabeledPoint(70, Vectors.dense(6,500)),
                new LabeledPoint(50, Vectors.dense(8,30))
        );
        JavaSparkContext jsc = new JavaSparkContext(sc);
        RDD<LabeledPoint> labeledPointRDD = jsc.parallelize(dataTraining).rdd();
        LinearRegressionModel model = LinearRegressionWithSGD.train(labeledPointRDD,2,0.1);
        System.out.println(model.predict(Vectors.dense(2,1)));

    }
}
