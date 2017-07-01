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
 * logistic predict
 *
 * @author guocheng.zhao
 * @date 2017/4/6 13:33
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */

public class LogisticRegressionWithGSDTest {
    private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(LogisticRegressionWithGSDTest.class);

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("LogisticRegressionWithGSDTest")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        List<LabeledPoint> dataTraining = Arrays.asList(
                new LabeledPoint(1, Vectors.dense(2)),
                new LabeledPoint(1, Vectors.dense(3)),
                new LabeledPoint(1, Vectors.dense(4)),
                new LabeledPoint(1, Vectors.dense(5)),
                new LabeledPoint(1, Vectors.dense(6)),
                new LabeledPoint(0, Vectors.dense(7)),
                new LabeledPoint(0, Vectors.dense(8)),
                new LabeledPoint(0, Vectors.dense(9)),
                new LabeledPoint(0, Vectors.dense(10)),
                new LabeledPoint(0, Vectors.dense(11))
        );
        SparkContext sc = new SparkContext(conf);
        JavaSparkContext jsc = new JavaSparkContext(sc);
        RDD<LabeledPoint> labeledPointRDD = jsc.parallelize(dataTraining).rdd();
        LogisticRegressionModel model = LogisticRegressionWithSGD.train(labeledPointRDD,50);
        System.out.println(model.predict(Vectors.dense(12)));
        //多远逻辑回归示例
        RDD<LabeledPoint> rdd2 = MLUtils.loadLibSVMFile(sc, "data\\mllib\\sample_libsvm_data.txt");
        LogisticRegressionModel model2 = LogisticRegressionWithSGD.train(rdd2, 50);
        System.out.println("model2.weights().size():"+model2.weights().size());
        System.out.println("model2.weights():"+model2.weights());
        RDD<LabeledPoint>[] rdds = rdd2.randomSplit(new double[]{0.6, 0.4}, 11l);
        //训练数据集
        RDD<LabeledPoint> parsedData = rdds[0];
        //测试数据集
        RDD<LabeledPoint> parsedTTes = rdds[1];
        LogisticRegressionModel model3 = LogisticRegressionWithSGD.train(parsedData, 50);
        System.out.println("model3.weights():"+model3.weights());
        JavaRDD<Tuple2<Object,Object>> predict_lable = parsedTTes.toJavaRDD().map(new Function<LabeledPoint, Tuple2<Object,Object>>() {
            @Override
            public Tuple2 call(LabeledPoint labeledPoint) throws Exception {
                Double predict = model3.predict(labeledPoint.features());
                return new Tuple2(predict, labeledPoint.label());
            }
        });
        MulticlassMetrics multiclassMetrics = new MulticlassMetrics(predict_lable.rdd());
        //训练模型和预测值的差异
        System.out.println("multiclassMetrics.precision():"+multiclassMetrics.precision());
        //支持向量机
        SVMModel svmModel = SVMWithSGD.train(parsedData, 10);
        System.out.println("权重svmModel.weights():"+svmModel.weights());
        System.out.println("截距svmModel.intercept():"+svmModel.intercept());
    }
}
