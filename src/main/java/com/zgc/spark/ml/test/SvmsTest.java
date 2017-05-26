package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import scala.Tuple2;

import java.io.File;

/**
 * project name is cms
 *
 * @author guocheng.zhao
 * @date 2017/5/25 15:59
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class SvmsTest {
    public static void main(String[] args) {
        //hadoop home xia ������winutils.exe�ļ�
        // E:\HADOOP\hadoop-2.7.3\bin\winutils.exe
        System.setProperty("hadoop.home.dir", "E:/HADOOP/hadoop-2.7.3/");
        //System.setProperty("hadoop.home.dir", "winutils/");
        SparkConf sparkConf = new SparkConf().setAppName("logistic")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local[2]");
        SparkContext sc = new SparkContext(sparkConf);
        String path = "data/mllib/sample_libsvm_data.txt";
        //��LIBSVM��ʽ����ѵ�����ݡ�
        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(sc, path).toJavaRDD();
        // Split initial RDD into two... [60% training data, 40% testing data].
        //ref http://www.myexception.cn/other/1961405.html rdd����api���
        //ref http://www.cnblogs.com/MOBIN/p/5373256.html rdd����api���
        //�����ݷֽ�Ϊѵ����60�����Ͳ��ԣ�40������
        //true�� ��ʾ�зŻصĳ���,0.6��ʾ�������ʣ�1l seed �������
        JavaRDD<LabeledPoint> training = data.sample(false, 0.01, 1L);
        //training.cache();
        JavaRDD<LabeledPoint> test = data.subtract(training);
        // Run training algorithm to build the model.
        System.out.println("data count:"+data.count());
        System.out.println("training count:"+training.count());
        System.out.println("test count:"+test.count());
        int numIterations = 100;
        final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);

        // Clear the default threshold.
        //���Ĭ����ֵ��
        model.clearThreshold();

        // Compute raw scores on the test set.
        //������Լ��ϵ�ԭʼ����
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double score = model.predict(p.features());
                        return new Tuple2<Object, Object>(score, p.label());
                    }
                }
        );

        scoreAndLabels.foreach(new VoidFunction<Tuple2<Object, Object>>() {
            @Override
            public void call(Tuple2<Object, Object> objectObjectTuple2) throws Exception {
                System.out.println(objectObjectTuple2);
            }
        });
        System.out.println("------------------");
        // Get evaluation metrics.
        //��ȡ����ָ�ꡣ
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();
        System.out.println("Area under ROC = " + auROC);
        // Save and load model
        //���沢����ģ��
        String saveOutPath = "target/tmp/javaSVMWithSGDModel";
        File  f = new File(saveOutPath);
        if(f.exists()) {
            f.delete();
        }
        model.save(sc, saveOutPath);
        SVMModel sameModel = SVMModel.load(sc, saveOutPath);
    }
}
