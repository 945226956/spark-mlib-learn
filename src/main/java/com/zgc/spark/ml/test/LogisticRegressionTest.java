package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

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

public class LogisticRegressionTest {
    private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(LogisticRegressionTest.class);

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("logistic")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        SparkContext sc = new SparkContext(conf);
        // Every record of this DataFrame contains the label and
        // features represented by a vector.
        StructType schema = new StructType(new StructField[]{
                new StructField("label", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty()),
        });
        SQLContext sqlContext = new SQLContext(sc);
        // Prepare training data.
        List<Row> dataTraining = Arrays.asList(
                RowFactory.create(0.0, Vectors.dense(5)),
                RowFactory.create(1.0, Vectors.dense(6)),
                RowFactory.create(1.0, Vectors.dense(7)),
                RowFactory.create(1.0, Vectors.dense(8)),
                RowFactory.create(1.0, Vectors.dense(9)),
                RowFactory.create(1.0, Vectors.dense(10))

        );
        DataFrame df = sqlContext.createDataFrame(dataTraining, schema);

        //DataFrame df = sqlContext.createDataFrame(textFile, schema);

        // Set parameters for the algorithm.
        // Here, we limit the number of iterations to 10.
        LogisticRegression lr = new LogisticRegression().setMaxIter(10);
        // Print out the parameters, documentation, and any default values.
        System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");

        // Fit the model to the data.
        LogisticRegressionModel model1 = lr.fit(df);

        // Given a dataset, predict each point's label, and show the results.
        model1.transform(df).toJavaRDD().foreach(new VoidFunction<Row>() {
            @Override
            public void call(Row row) throws Exception {
                System.out.print(row.get(0) + "---");
                System.out.print(row.get(1) + "---");
                System.out.print(row.get(2) + "---");
                System.out.print(row.get(3) + "---");
                System.out.print(row.get(4) + "---");
                System.out.println("-----------");
            }
        });
        model1.transform(df).show();
        Vector featrues1 = Vectors.dense(11);
        System.out.println("预测---" + model1.predict(featrues1));
        System.out.println(model1.getWeightCol());
        System.out.println(model1.numClasses());
        System.out.println(model1.numFeatures());

        //查看使用的参数设置
        /*System.out.println("Model 1 was fit using parameters: " + model1.parent().extractParamMap());
        // 使用 ParamMap 来设置算法执行参数
        ParamMap paramMap = new ParamMap()
                .put(lr.maxIter().w(20))  // 指定一个参数
                .put(lr.maxIter(), 30)  // 覆盖上面指定的参数
                .put(lr.regParam().w(0.1), lr.threshold().w(0.55));  // 也可以指定多个参数


        ParamMap paramMap2 = new ParamMap()
                //.put(lr.probabilityCol().w("myProbability"))// 改变输出预测的列名
                ;

        // 组合参数设置
        ParamMap paramMapCombined = paramMap.$plus$plus(paramMap2);
        LogisticRegressionModel model2 = lr.fit(df, paramMapCombined);
        System.out.println("Model 2 was fit using parameters: " + model2.parent().extractParamMap());
        DataFrame rs = model2.transform(df);
        rs.show();*/
        //DataFrame rows = rs.select("features", "label", "myProbability", "prediction");
        /*for (Row r: rows.collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + ") -> prob=" + r.get(2)
                    + ", prediction=" + r.get(3));
        }*/

    }
}
