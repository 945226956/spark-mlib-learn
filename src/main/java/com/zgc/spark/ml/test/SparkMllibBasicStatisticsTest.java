package com.zgc.spark.ml.test;

import com.google.common.collect.ImmutableMap;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.api.java.function.VoidFunction2;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.random.RandomRDDs;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.KernelDensity;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.mllib.stat.test.*;
import org.apache.spark.streaming.*;
import org.apache.spark.streaming.api.java.JavaDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.io.File;
import java.io.IOException;
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
    public static void main(String[] args) throws Exception {
        SparkConf sparkConf = new SparkConf().setAppName("logistic")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        SparkContext sc = new SparkContext(sparkConf);
        JavaSparkContext jsc = new JavaSparkContext("local", "First Spark App", sparkConf);
        //基础统计
        //summaryStatistics(jsc);
        //相关性
        //correlations(jsc);
        //分层抽样
        //stratifiedSampling(jsc);
        //假设验证
        //hypothesisTesting(jsc);
        ;
        //流式意义测试
        //streamingSignificanceTesting(sparkConf);
        //随机数据生成
        //randomDataGeneration(jsc);
        //核密度估计
        kernelDensityEstimation(jsc);
    }

    /**
     * 汇总统计
     *
     * @param jsc
     */
    private static void summaryStatistics(JavaSparkContext jsc) {
        JavaRDD<Vector> mat = jsc.parallelize(
                Arrays.asList(
                        Vectors.dense(1.0, 10.0, 100.0),
                        Vectors.dense(2.0, 20.0, 200.0),
                        Vectors.dense(3.0, 30.0, 300.0)
                )
        );
        MultivariateStatisticalSummary summary = Statistics.colStats(mat.rdd());
        System.out.println("行数:summary.count():" + summary.count());
        System.out.println("最大值:summary.max():" + summary.max());
        System.out.println("最小值:summary.min():" + summary.min());
        //每一列的平均值
        System.out.println("每一列平均值summary.mean():" + summary.mean());
        //方差：集中程度。除以n-1而不是n，这样我们能以较小的样本更好地逼近总体的方差，即统计上所谓的“无偏估计”。方差是标准差平方。
        //variance(DimA)=∑nk=1(ak−mean)2/(n−1)
        //平均值:[2.0,20.0,200.0] 方差: [1.0,100.0,10000.0]
        //1.0 = {(1.0-2.0)*(1.0-2.0)+(2.0-2.0)*(2.0-2.0)+(3.0-2.0)*(1.0-2.0)}/3-1
        System.out.println("每一列方差 summary.variance():" + summary.variance());  // column-wise variance
        //每一列的非零向量数量
        System.out.println("每一列的非零向量数量summary.numNonzeros():" + summary.numNonzeros());  // number of nonzeros in each column

    }

    /**
     * 相关性
     * 相关系数 0.8-1.0 极强相关
     * 0.6-0.8 强相关
     * 0.4-0.6 中等程度相关
     * 0.2-0.4 弱相关
     * 0.0-0.2 极弱相关或无相关
     * 需要指出的是，相关系数有一个明显的缺点，即它接近于1的程度与数据组数n相关，这容易给人一种假象。
     * 因为，当n较小时，相关系数的波动较大，对有些样本相关系数的绝对值易接近于1；当n较大时，相关系数的绝对值容易偏小。
     * 特别是当n=2时，相关系数的绝对值总为1。
     * 因此在样本容量n较小时，我们仅凭相关系数较大就判定变量x与y之间有密切的线性关系是不妥当的。
     * 例如，就我国深沪两股市资产负债率与每股收益之间的相关关系做研究。
     * 发现1999年资产负债率前40名的上市公司，二者的相关系数为r=–0.6139；资产负债率后20名的上市公司，二者的相关系数r=0.1072；
     * 而对于沪、深全部上市公司（基金除外）结果却是，r沪=–0.5509，r深=–0.4361，根据三级划分方法，两变量为显著性相关。
     * 这也说明仅凭r的计算值大小判断相关程度有一定的缺陷。
     *
     * @param jsc
     */
    public static void correlations(JavaSparkContext jsc) {
        JavaDoubleRDD seriesX = jsc.parallelizeDoubles(
                Arrays.asList(1.0, 2.0, 3.0));  // a series

        // must have the same number of partitions and cardinality as seriesX
        JavaDoubleRDD seriesY = jsc.parallelizeDoubles(
                Arrays.asList(11.0, 22.0, 3.0));

        // compute the correlation using Pearson's method. Enter "spearman" for Spearman's method.
        // If a method is not specified, Pearson's method will be used by default.
        //参考公式  http://www.baike.com/wiki/%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0
        Double correlation = Statistics.corr(seriesX.srdd(), seriesY.srdd(), "pearson");
        System.out.println("Correlation is: " + correlation);

        // note that each Vector is a row and not a column
        JavaRDD<Vector> data = jsc.parallelize(
                Arrays.asList(
                        Vectors.dense(1.0, 10.0, 100.0),
                        Vectors.dense(2.0, 20.0, 200.0),
                        Vectors.dense(5.0, 33.0, 366.0)
                )
        );
        Matrix matrix = Statistics.corr(data.rdd(), "pearson");
        System.out.println("matrix Correlation is: \n" + matrix);

    }

    /**
     * 分层抽样
     * http://baike.baidu.com/link?url=2ffJL6psKZ8q5mGdP8w22hYcrEokiJlnyS_0kH9mmJM3dk3wIIfGb91md20atAydbqwMtb4AHwe4ZTRfbfvMjHQQ0ZNgQh4KmPioTM8INfIdU8U-VB3May77Z-7pHT9a5_q69WnnTJCZP6GpKyBzDlcAqdvm5PACAbDbRDUGetgsdQYnVaBibgbS1sekOdPwvRtzRluu7NBBwLpIJ1h_D3CXGSO6pM8Hg_IODzMfv9O
     *
     * @param jsc
     */
    public static void stratifiedSampling(JavaSparkContext jsc) {
        List<Tuple2<Integer, Character>> list = Arrays.asList(
                new Tuple2<>(1, 'a'),
                new Tuple2<>(1, 'b'),
                new Tuple2<>(2, 'c'),
                new Tuple2<>(2, 'd'),
                new Tuple2<>(2, 'e'),
                new Tuple2<>(3, 'f')
        );

        JavaPairRDD<Integer, Character> data = jsc.parallelizePairs(list);

        // 指定不同key抽取的百分比
        // 1:10%,2:60%,3:30%
        ImmutableMap fractions = ImmutableMap.of(1, 0.9, 2, 0.1, 3, 0.1);

        // 近似样本
        System.out.println("近似样本:");
        JavaPairRDD<Integer, Character> approxSample = data.sampleByKey(false, fractions);
        approxSample.foreach(new VoidFunction<Tuple2<Integer, Character>>() {
            @Override
            public void call(Tuple2<Integer, Character> integerCharacterTuple2) throws Exception {
                System.out.println(integerCharacterTuple2._1 + "--" + integerCharacterTuple2._2);
            }
        });
        //  精确样本
        System.out.println("精确样本:");
        JavaPairRDD<Integer, Character> exactSample = data.sampleByKeyExact(false, fractions);
        exactSample.foreach(new VoidFunction<Tuple2<Integer, Character>>() {
            @Override
            public void call(Tuple2<Integer, Character> integerCharacterTuple2) throws Exception {
                System.out.println(integerCharacterTuple2._1 + "--" + integerCharacterTuple2._2);
            }
        });
    }

    /**
     * 假设校验
     *
     * @param jsc
     */
    public static void hypothesisTesting(JavaSparkContext jsc) {
        //事件发生的概率向量
        Vector vec = Vectors.dense(0.1, 0.15, 0.2, 0.3, 0.25);
        //计算适合度。如果没有提供第二个测试向量,作为参数，测试针对均匀分布运行。
        ChiSqTestResult goodnessOfFitTestResult = Statistics.chiSqTest(vec);

        //卡方检验总结：
        // 方法：Pearson
        // 自由度= 4 （不同事件数量-1）
        // 统计量= 0.12499999999999999
        // pValue = 0.998126379239318
        //无假设为零假设：遵循与预期相同的分布。
        //ref http://wiki.mbalib.com/wiki/%E5%8D%A1%E6%96%B9%E6%A3%80%E9%AA%8C
        /*Chi squared test summary:
        method: pearson
        degrees of freedom = 4
        statistic = 0.12499999999999999
        pValue = 0.998126379239318
        No presumption against null hypothesis: observed follows the same distribution as expected..*/
        System.out.println(goodnessOfFitTestResult + "\n");

        // Create a contingency matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        Matrix mat = Matrices.dense(3, 2, new double[]{1.0, 3.0, 5.0, 2.0, 4.0, 6.0});

        // 对输入的应急矩阵进行Pearson独立性测试
        ChiSqTestResult independenceTestResult = Statistics.chiSqTest(mat);
        // ummary of the test including the p-value, degrees of freedom...
        //测试总结包括p值，自由度...

        System.out.println(independenceTestResult + "\n");

        // an RDD of labeled points
        JavaRDD<LabeledPoint> obs = jsc.parallelize(
                Arrays.asList(
                        new LabeledPoint(1.0, Vectors.dense(1.0, 0.0, 3.0)),
                        new LabeledPoint(1.0, Vectors.dense(1.0, 2.0, 0.0)),
                        new LabeledPoint(-1.0, Vectors.dense(-1.0, 0.0, -0.5))
                )
        );

        // The contingency table is constructed from the raw (feature, label) pairs and used to conduct
        // the independence test. Returns an array containing the ChiSquaredTestResult for every feature
        // against the label.
        //应急表由原始（特征，标签）对构成，用于进行独立测试。返回一个包含每个功能的ChiSquaredTestResult的数组 反对标签。
        ChiSqTestResult[] featureTestResults = Statistics.chiSqTest(obs.rdd());
        int i = 1;
        for (ChiSqTestResult result : featureTestResults) {
            System.out.println("Column " + i + ":");
            System.out.println(result + "\n");  // summary of the test
            i++;
        }

        JavaDoubleRDD data = jsc.parallelizeDoubles(Arrays.asList(0.1, 0.15, 0.2, 0.3, 0.25));
        KolmogorovSmirnovTestResult testResult =
                Statistics.kolmogorovSmirnovTest(data, "norm", 0.0, 1.0);
        // summary of the test including the p-value, test statistic, and null hypothesis
        //测试总结包括p值，检验统计量和零假设
        // if our p-value indicates significance, we can reject the null hypothesis
        //如果我们的p值表示重要性，我们可以拒绝零假设
        System.out.println(testResult);
    }

    /**
     * 流式意义测试
     * 不是很懂。。。
     *
     * @param sparkConf
     */
    public static void streamingSignificanceTesting(SparkConf sparkConf) throws Exception {
        JavaSparkContext jsc = new JavaSparkContext("local", "First Spark App", sparkConf);
        JavaSparkContext javaSparkContext = new JavaSparkContext("local", "First Spark App", sparkConf);
        File f = new File("a.txt");
        JavaStreamingContext javaStreamingContext = new JavaStreamingContext(javaSparkContext, Durations.seconds(2000));
        String dataDir = f.getAbsolutePath();
        JavaDStream<BinarySample> data = javaStreamingContext.textFileStream(dataDir).map(
                new Function<String, BinarySample>() {
                    @Override
                    public BinarySample call(String line) {
                        String[] ts = line.split(",");
                        boolean label = Boolean.parseBoolean(ts[0]);
                        double value = Double.parseDouble(ts[1]);
                        return new BinarySample(label, value);
                    }
                });
        StreamingTest streamingTest = new StreamingTest()
                .setPeacePeriod(10)
                .setWindowSize(0)
                .setTestMethod("welch");
        JavaDStream<StreamingTestResult> out = streamingTest.registerStream(data);
        out.print();
        out.foreachRDD(new VoidFunction2<JavaRDD<StreamingTestResult>, Time>() {
            @Override
            public void call(JavaRDD<StreamingTestResult> streamingTestResultJavaRDD, Time time) throws Exception {
                System.out.println(time);
                streamingTestResultJavaRDD.foreach(new VoidFunction<StreamingTestResult>() {
                    @Override
                    public void call(StreamingTestResult streamingTestResult) throws Exception {
                        System.out.println("----" + streamingTestResult);
                    }
                });
            }
        });
    }

    /**
     * 随机数据生成
     *
     * @param javaSparkContext
     */
    public static void randomDataGeneration(JavaSparkContext javaSparkContext) {
        //生成一个包含10个随机double的 RDD。
        // 从标准正态分布“N（0，1）”，均匀分布在5个分区中。
        JavaDoubleRDD u = RandomRDDs.normalJavaRDD(javaSparkContext, 10L, 5);
        u.foreach(new VoidFunction<Double>() {
            @Override
            public void call(Double aDouble) throws Exception {
                System.out.println("----" + aDouble);
            }
        });
        System.out.println("!!!!!!!!!!!!!!!!!!!!");
        //在“N（1，4）”之后，进行变换以获得随机double RDD。
        JavaRDD<Double> v = u.map(
                new Function<Double, Double>() {
                    public Double call(Double x) {
                        return 1.0 + 2.0 * x;
                    }
                });
        System.out.println(v.count());
        v.foreach(new VoidFunction<Double>() {
            @Override
            public void call(Double aDouble) throws Exception {
                System.out.println("----" + aDouble);
            }
        });
    }

    /**
     * 核密度估计
     * 并不是很懂什么意思
     * @param javaSparkContext
     */
    public static void kernelDensityEstimation(JavaSparkContext javaSparkContext){
        // an RDD of sample data
        JavaRDD<Double> data = javaSparkContext.parallelize(
                Arrays.asList(1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0));
        // Construct the density estimator with the sample data
        // and a standard deviation for the Gaussian kernels
        KernelDensity kd = new KernelDensity().setSample(data).setBandwidth(3.0);
        // Find density estimates for the given values
        double[] densities = kd.estimate(new double[]{-1.0, 2.0, 5.0});
        System.out.println(Arrays.toString(densities));
    }
}
