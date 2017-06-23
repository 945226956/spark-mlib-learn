package com.zgc.spark.ml.test;

import com.zgc.spark.Utils.MlibUtls;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import scala.Tuple2;

import java.io.File;

/**
 * 协同过滤
 *
 * @author guocheng.zhao
 * @date 2017/6/7 9:27
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class CollaborativeFilteringTest {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("Java Collaborative Filtering Example")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local[2]");
        JavaSparkContext jsc = new JavaSparkContext(sparkConf);
        String path = "data/mllib/als/test.data";
        JavaRDD<String> data = jsc.textFile(path);
        JavaRDD<Rating> ratings = data.map(new Function<String, Rating>() {
            @Override
            public Rating call(String s) throws Exception {
                String[] arr = s.split(",");
                int user = Integer.parseInt(arr[0]);
                int product = Integer.parseInt(arr[1]);
                double rating = Double.parseDouble(arr[2]);
                return new Rating(user, product, rating);
            }
        });
        String path2 = "data/mllib/als/sample_movielens_ratings.txt";
        JavaRDD<String> data2 = jsc.textFile(path2);
        JavaRDD<Rating> ratings2 = data2.map(new Function<String, Rating>() {
            @Override
            public Rating call(String s) throws Exception {
                String[] arr = s.split("::");
                int user = Integer.parseInt(arr[0]);
                int product = Integer.parseInt(arr[1]);
                double rating = Double.parseDouble(arr[2]);
                return new Rating(user, product, rating);
            }
        });



        CollaborativeFilter1(jsc, ratings2);

    }

    /**
     * 协同过滤
     * @param jsc
     * @param ratings
     */
    private static void CollaborativeFilter1(JavaSparkContext jsc, JavaRDD<Rating> ratings) {
        // Build the recommendation model using ALS
        int rank = 10;
        int numIterations = 10;
        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numIterations, 0.01);

        //给所有人推荐一个商品
        System.out.println("给所有用户推荐2个商品");
        model.recommendProductsForUsers(2).toJavaRDD().foreach(new VoidFunction<Tuple2<Object, Rating[]>>() {
            @Override
            public void call(Tuple2<Object, Rating[]> objectTuple2) throws Exception {
                System.out.println(objectTuple2._1+"--");
                CollaborativeFilteringTest.showRating(objectTuple2._2);
            }
        });
        //给用户1 推荐2个商品
        Rating[] recommendProducts = model.recommendProducts(1, 2);
        System.out.println("给用户1推荐2个商品");
        CollaborativeFilteringTest.showRating(recommendProducts);
        Rating[] recommendUsers = model.recommendUsers(61, 2);
        System.out.println("将商品61推荐给2个人");
        CollaborativeFilteringTest.showRating(recommendUsers);
        System.out.println("将所有商品推荐给2个人");
        model.recommendUsersForProducts(2).toJavaRDD().foreach(new VoidFunction<Tuple2<Object, Rating[]>>() {
            @Override
            public void call(Tuple2<Object, Rating[]> objectTuple2) throws Exception {
                System.out.println("商品:"+objectTuple2._1+"--");
                CollaborativeFilteringTest.showRating(objectTuple2._2);
            }
        });
        System.out.println("用户特征:");
        model.userFeatures().toJavaRDD().foreach(new VoidFunction<Tuple2<Object, double[]>>() {
            @Override
            public void call(Tuple2<Object, double[]> objectTuple2) throws Exception {
                System.out.println("用户:"+objectTuple2._1+"---"+objectTuple2._2.toString());
            }
        });

        System.out.println("用户1给商品20的预测评分:"+model.predict(1, 20));
        // 构建用户产品关系rdd
        JavaRDD<Tuple2<Object, Object>> userProducts = ratings.map(
                new Function<Rating, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(Rating r) {
                        return new Tuple2<Object, Object>(r.user(), r.product());
                    }
                }
        );
        //根据训练模型预测真实数据，得到用户产品的预测评分
        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
                model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD().map(
                        new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                            public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                                //System.out.println("用户："+r.user()+"--产品："+r.product()+"--预测评分："+r.rating());
                                return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                            }
                        }
                ));
        System.out.println("实际数据和预测数据join操作 得到预测评分和真实评分");

        JavaPairRDD<Tuple2<Integer, Integer>, Tuple2<Double, Double>> joinRdd = JavaPairRDD.fromJavaRDD(ratings.map(
                new Function<Rating, Tuple2<Tuple2<Integer, Integer>, Double>>() {
                    public Tuple2<Tuple2<Integer, Integer>, Double> call(Rating r) {
                        return new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating());
                    }
                }
        )).join(predictions);
        joinRdd.foreach(new VoidFunction<Tuple2<Tuple2<Integer, Integer>, Tuple2<Double, Double>>>() {
            @Override
            public void call(Tuple2<Tuple2<Integer, Integer>, Tuple2<Double, Double>> tuple2Tuple2Tuple2) throws Exception {
                System.out.println("用户："+tuple2Tuple2Tuple2._1._1+"--产品："+tuple2Tuple2Tuple2._1._2+"--真实评分："+tuple2Tuple2Tuple2._2._1+"--预测评分："+tuple2Tuple2Tuple2._2._2());
            }
        });

        JavaRDD<Tuple2<Double, Double>> ratesAndPreds = joinRdd.values();
        System.out.println("预测评分和真实评分求出方差 均方误差");
        double MSE = JavaDoubleRDD.fromRDD(ratesAndPreds.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        Double err = pair._1() - pair._2();
                        return err * err;
                    }
                }
        ).rdd()).mean();
        System.out.println("Mean Squared Error = " + MSE);
        // 保存训练好的模型
        String path = "target/tmp/myCollaborativeFilter";
        MlibUtls.deleteAllFilesOfDir(new File(path));
        model.save(jsc.sc(), path);
        MatrixFactorizationModel sameModel = MatrixFactorizationModel.load(jsc.sc(),
                "target/tmp/myCollaborativeFilter");
        System.out.println(sameModel);
    }

    public static void showRating(Rating[] ratings) {
        System.out.println("----------BEGIN-----------");
        for (Rating r: ratings){
            System.out.println("用户："+r.user()+"--商品："+r.product()+"--评分："+r.rating());
        }
        System.out.println("----------END-----------");
    }
}
