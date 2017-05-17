package com.zgc.spark.ml.test;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.mllib.linalg.*;
import org.apache.spark.mllib.linalg.distributed.*;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SQLContext;
import scala.reflect.ClassTag;

import java.util.Arrays;
import java.util.List;

/**
 * spark mlib 基础数据类型练习
 *
 * @author guocheng.zhao
 * @date 2017/5/16 11:56
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class SparkMllibDataTypeTest {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("logistic")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        SparkContext sc = new SparkContext(sparkConf);
        //VectorsTest(sc);
        MatricesTest(sparkConf);
    }

    /**
     * 矩阵测试
     *
     * @param sparkConf
     */
    private static void MatricesTest(SparkConf sparkConf) {
        System.out.println("本地矩阵Matrices------------");
        // Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        Matrix dm = Matrices.dense(3, 2, new double[]{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});
        Matrix dm2 = Matrices.dense(2, 3, new double[]{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});
        // Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
        Matrix sm = Matrices.sparse(3, 2, new int[]{0, 1, 3}, new int[]{0, 2, 1}, new double[]{9, 6, 8});
        /* 3行2列
        1.0  2.0
        3.0  4.0
        5.0  6.0
        */
        System.out.println("dm:\n" + dm);
        /* 2行3列
        1.0  5.0  4.0
        3.0  2.0  6.0
        */
        System.out.println("dm2:\n" + dm2);
        /*
        3 x 2 CSCMatrix
        (0,0) 9.0
        (2,1) 6.0
        (1,1) 8.0
        */
        System.out.println("sm:\n" + sm);

        System.out.println("分布式矩阵 行矩阵RowMatrix------------");
        List<Vector> localVector = Arrays.asList(
                Vectors.sparse(2, new int[]{0, 1}, new double[]{2, 3}),
                Vectors.sparse(2, new int[]{0, 1}, new double[]{4, 5})
        );
        JavaSparkContext jsc = new JavaSparkContext("local", "First Spark App", sparkConf);
        //将本地list转分布式rdd ！！
        JavaRDD<Vector> rows = jsc.parallelize(localVector);
        RowMatrix mat = new RowMatrix(rows.rdd());

        // Get its size.
        System.out.println("mat numRows:" + mat.numRows());
        System.out.println("mat numCols:" + mat.numCols());
        System.out.println("mat:\n" + mat.toString());

        // QR 分解矩阵 ,报错，暂时未找到原因
        //QRDecomposition<RowMatrix, Matrix> result = mat.tallSkinnyQR(true);

        System.out.println("分布式矩阵 索引行矩阵 IndexedRowMatrix------------");
        List<IndexedRow> localIndexedRowVector = Arrays.asList(
                ////IndexeRow就是对Vector的一层封装。
                new IndexedRow(1, Vectors.sparse(2, new int[]{0, 1}, new double[]{2, 3})),
                new IndexedRow(2, Vectors.sparse(2, new int[]{0, 1}, new double[]{4, 5}))
        );
        IndexedRowMatrix mat2 = new IndexedRowMatrix(jsc.parallelize(localIndexedRowVector).rdd());
        System.out.println("mat2 numRows:" + mat2.numRows());
        System.out.println("mat2 numCols:" + mat2.numCols());
        System.out.println("mat2.toRowMatrix()转成RowMatrix:"+mat2.toRowMatrix());


        System.out.println("分布式矩阵 坐标矩阵CoordinateMatrix------------");
        List<MatrixEntry> matrixEntries =  Arrays.asList(
                new MatrixEntry(0,0,1),
                new MatrixEntry(0,1,2),
                new MatrixEntry(1,0,3),
                new MatrixEntry(1,1,4)
        );
        JavaRDD<MatrixEntry> matrixEntryJavaRDD = jsc.parallelize(matrixEntries);
        // Create a CoordinateMatrix from a JavaRDD<MatrixEntry>.
        CoordinateMatrix mat3 = new CoordinateMatrix(matrixEntryJavaRDD.rdd());
        // Get its size.
        System.out.println("mat3 numRows:" + mat3.numRows());
        System.out.println("mat3 numCols:" + mat3.numCols());
        // Convert it to an IndexRowMatrix whose rows are sparse vectors.
        IndexedRowMatrix indexedRowMatrix = mat3.toIndexedRowMatrix();
        System.out.println("mat3.toIndexedRowMatrix()转成 IndexedRowMatrix " +indexedRowMatrix);

        System.out.println("分布式矩阵 分块矩阵BlockMatrix------------");
        //可由坐标矩阵转成
        BlockMatrix blockMatrix = mat3.toBlockMatrix();
        //验证
        blockMatrix.validate();
        // Calculate A^T A. 矩阵乘法
        BlockMatrix multiplyBlockMatrix =  blockMatrix.transpose().multiply(blockMatrix);
        System.out.println(""+multiplyBlockMatrix);
    }

    /**
     * 本地向量，标示点测试
     *
     * @param sc
     */
    private static void VectorsTest(SparkContext sc) {
        //本地向量
        //本地向量具有整数类型和基于0的索引和双类型值，存储在单个机器上。MLlib支持两种类型的局部向量：密集和稀疏。密集向量由表示其条目值的双数组支持，而稀疏向量由两个并行数组支持：索引和值。
        //例如，向量(1.0, 0.0, 3.0)可以密集格式表示为[1.0, 0.0, 3.0]或以稀疏格式表示(3, [0, 2], [1.0, 3.0])，其中3向量的大小在哪里。
        Vector dv = Vectors.dense(1, 2, 3, 4);
        System.out.println(dv.toJson());
        Vector sv = Vectors.sparse(3, new int[]{1, 2, 3}, new double[]{1.2, 1.3, 1.5});
        System.out.println(sv.toJson());
        //标记点
        //标记点是与标签/响应相关联的局部矢量，密集或稀疏。在MLlib中，标注点用于监督学习算法。
        // 我们使用双重存储标签，所以我们可以在回归和分类中使用标记点。对于二进制分类，标签应为0（负）或1（正）。
        // 对于多类分类，标签应该是从零开始的类索引：0, 1, 2, ...。
        //创建带有正标签和密集特征向量的标记点
        LabeledPoint pos1 = new LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0));
        System.out.println("pos1:" + pos1);

        LabeledPoint pos2 = new LabeledPoint(1.0, Vectors.sparse(3, new int[]{0, 1}, new double[]{1, 2}));
        //pos2:(1.0,(3,[0,1],[1.0,2.0]))
        System.out.println("pos2:" + pos2);

        //创建带有负标签和稀疏特征向量的标记点。
        LabeledPoint neg = new LabeledPoint(0.0, Vectors.sparse(3, new int[]{0, 2}, new double[]{1.0, 3.0}));

        //读取libsvm格式数据转换为LabledPoint
        /*
        0 1:1 2:2
        0 1:2 2:2
        0 1:3 2:2
        0 1:4 2:2
        0 1:5 2:2
        1 1:6 2:2
        1 1:7 2:2
        1 1:8 2:2
        1 1:9 2:2
        1 1:10 2:1
        */
        JavaRDD<LabeledPoint> examples =
                MLUtils.loadLibSVMFile(sc, "E:/机器学习/data/mllib/sample_libsvm_data.txt").toJavaRDD();
        examples.foreach(new VoidFunction<LabeledPoint>() {
            @Override
            public void call(LabeledPoint labeledPoint) throws Exception {
                //(0.0,(2,[0,1],[1.0,2.0]))
                System.out.println(labeledPoint);
            }
        });
        SQLContext sqlContext = new SQLContext(sc);
        LogisticRegression lr = new LogisticRegression().setMaxIter(10);
        DataFrame df = sqlContext.createDataFrame(examples, LabeledPoint.class);
        LogisticRegressionModel model1 = lr.fit(df);
        model1.transform(df).show();
        Vector f = Vectors.dense(10, 1);
        System.out.println("预测" + model1.predict(f));
        ;
    }
}
