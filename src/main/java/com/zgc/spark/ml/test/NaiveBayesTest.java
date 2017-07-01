package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.rdd.RDD;

import java.util.Arrays;
import java.util.List;

/**
 * 朴素贝叶斯分类
 * ref :http://blog.csdn.net/illbehere/article/details/53230916
 * @author guocheng.zhao
 * @date 2017/4/6 13:33
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */

public class NaiveBayesTest {
    private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(NaiveBayesTest.class);

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("Bayes")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        SparkContext sc = new SparkContext(conf);
        // Prepare training data.
        List<LabeledPoint> dataTraining = Arrays.asList(
                new LabeledPoint(0, Vectors.dense(1,0,0,4)),
                new LabeledPoint(0, Vectors.dense(2,0,0,4)),
                new LabeledPoint(1, Vectors.dense(0,1,0,4)),
                new LabeledPoint(1, Vectors.dense(1,2,0,4)),
                new LabeledPoint(2, Vectors.dense(1,0,1,4)),
                new LabeledPoint(2, Vectors.dense(1,0,2,4))
//                ,
//                new LabeledPoint(3, Vectors.dense(4,5,6)),
//                new LabeledPoint(4, Vectors.dense(1,7,7))
        );
        JavaSparkContext jsc = new JavaSparkContext(sc);
        RDD<LabeledPoint> labeledPointRDD = jsc.parallelize(dataTraining).rdd();
        //第二个参数 是平滑参数
        NaiveBayesModel naiveBayesModel = NaiveBayes.train(labeledPointRDD,1);
        //标签类别
        printlnArr(naiveBayesModel.labels());
        //各个标签类别的先验概率
        printlnArr(naiveBayesModel.pi());
        System.out.println("thetas:");
        double[][] thetas = naiveBayesModel.theta();
        for(double[] theta:thetas){
            for(double t: theta){
                System.out.print(t + "\t");
            }
            System.out.println("");
        }
        System.out.println(naiveBayesModel.predict(Vectors.dense(7,7,7)));

    }

    public static void printlnArr(double[] objects) {
        System.out.println("start---println");
        for (Object o:objects){
            System.out.println(o);
        }
        System.out.println("end---println");
    }
}
