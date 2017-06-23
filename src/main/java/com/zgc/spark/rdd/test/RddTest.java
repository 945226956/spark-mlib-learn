package com.zgc.spark.rdd.test;

import org.apache.spark.Partition;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.VoidFunction;
import scala.Tuple2;

import java.util.*;

/**
 * project name is spark
 *
 * @author guocheng.zhao
 * @date 2017/5/26 16:54
 * @tel 13524779402
 * @email guocheng.zhao@hand-china.com
 */
public class RddTest {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("logistic")
                .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .set("spark.driver.allowMultipleContexts", "true")
                .setMaster("local");
        JavaSparkContext jsc = new JavaSparkContext("local", "First Spark App", sparkConf);
        //aggregate 方法测试
        //rddAggregateTest(jsc);
        //笛卡尔计算
        rddCartesain(jsc);
    }

    /**
     * aggregate 方法测试
     *
     * @param jsc
     */
    private static void rddAggregateTest(JavaSparkContext jsc) {

        //这里把数据分成2个节点存储
        // (1,2,3),(4,5,6)
        //第一个函数执行过程---
        // 第一个节点： 0,1 1,2 2,3 得到3
        // 第二个节点： 0,4 4,5 5,6 得到6.
        //第二个函数执行过程--
        //0+3+6 =9
        JavaRDD<Integer> rdd = jsc.parallelize(Arrays.asList(1, 2, 3, 4, 5, 6), 2);
        //重新分配节点，参数为true表示保持每个节点数据平均 (1,6),(2,5),(3,4)
        //rdd = rdd.coalesce(3,true);
        //rdd = rdd.repartition(3);// 等于 rdd.coalesce(3,true)
        //aggregate(U,seqOp,combOp)
        //U 是数据类型，可以是任意，seqOp是给定计算的方法输出也是U，combOP是合并方法，将第一个计算的结果与U进行合并
        //注意 合并的次数和rdd的数据节点数一致。
        Integer rs = rdd.aggregate(0, new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer integer, Integer integer2) throws Exception {
                System.out.println("function seqOp");
                System.out.println("integer:" + integer + "---integer2:" + integer2);
                return integer > integer2 ? integer : integer2;
            }
        }, new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer integer, Integer integer2) throws Exception {
                //return integer > integer2?integer:integer2;
                System.out.println("function combOp");
                System.out.println("integer:" + integer + "---integer2:" + integer2);
                return integer + integer2;
            }
        });
        System.out.println("rs:" + rs);
        JavaRDD<String> rdd2 = jsc.parallelize(Arrays.asList("abc", "b", "c", "d", "e", "f"));
        String rs2 = rdd2.aggregate("", new Function2<String, String, String>() {
            @Override
            public String call(String s, String s2) throws Exception {
                return s + s2;
            }
        }, new Function2<String, String, String>() {
            @Override
            public String call(String s, String s2) throws Exception {
                return s + s2;
            }
        });
        System.out.println("rs2:" + rs2);
        System.out.println("rdd2:" + rdd2);
    }

    /**
     * 笛卡尔操作
     *
     * @param jsc
     */
    private static void rddCartesain(JavaSparkContext jsc) {
        JavaRDD<Integer> rdd1 = jsc.parallelize(Arrays.asList(1, 2, 3));
        Map<Integer, Long> valueCount = rdd1.countByValue();
        System.out.println("遍历rdd1 countByValue 得到的Map");
        for (Integer k : valueCount.keySet()) {
            System.out.println(k + "----" + valueCount.get(k));
        }
        JavaRDD<Integer> rdd2 = jsc.parallelize(Arrays.asList(5, 6));
        JavaPairRDD<Integer, Integer> dkrRdd = rdd1.cartesian(rdd2);
        System.out.println("笛卡尔操作:");
        dkrRdd.foreach(new VoidFunction<Tuple2<Integer, Integer>>() {
            @Override
            public void call(Tuple2<Integer, Integer> integerIntegerTuple2) throws Exception {
                System.out.println(integerIntegerTuple2._1 + "--" + integerIntegerTuple2._2);
            }
        });
        System.out.println("dkrRdd countBykey得到的结果");
        Map countByKey = dkrRdd.countByKey();
        for (Object k : countByKey.keySet()) {
            System.out.println(k + "----" + countByKey.get(k));
        }
        System.out.println("dkrRdd countByValue得到的结果");
        Map m = dkrRdd.countByValue();
        for (Object k : m.keySet()) {
            System.out.println(k + "---" + m.get(k));
        }
        System.out.println("rdd1.flatMap 得到的结果");
        //1,2,3
        rdd1.flatMap(new FlatMapFunction<Integer, Integer>() {
            @Override
            public Iterable<Integer> call(Integer integer) throws Exception {
                Set<Integer> s = new HashSet<Integer>();
                s.add(integer * integer);
                //1,4,9
                s.add(integer * integer * integer);
                //1,4,8,9,27
                return s;
            }
        }).foreach(new VoidFunction<Integer>() {
            @Override
            public void call(Integer integer) throws Exception {
                System.out.println(integer);
            }
        });
        System.out.println("rdd1 groupBy分组");
        rdd1.groupBy(new Function<Integer, String>() {
            @Override
            public String call(Integer integer) throws Exception {
                return integer > 2 ? ">2" : "<=2";
            }
        }, 3/*分片执行数*/).foreach(new VoidFunction<Tuple2<String, Iterable<Integer>>>() {
            @Override
            public void call(Tuple2<String, Iterable<Integer>> stringIterableTuple2) throws Exception {
                System.out.println(stringIterableTuple2._1 + "---" + stringIterableTuple2._2);
            }
        });
        System.out.println("rdd1 keyBy方法");
        rdd1.keyBy(new Function<Integer, String>() {
            @Override
            public String call(Integer integer) throws Exception {
                return "key";
            }
        }).foreach(new VoidFunction<Tuple2<String, Integer>>() {
            @Override
            public void call(Tuple2<String, Integer> stringIntegerTuple2) throws Exception {
                System.out.println(stringIntegerTuple2);
            }
        });

        System.out.println("rdd1 reduce方法");
        Integer reduceRs = rdd1.reduce(new Function2<Integer, Integer, Integer>() {
            @Override
            public Integer call(Integer integer, Integer integer2) throws Exception {
                //1+2+3
                return integer + integer2;
            }
        });
        System.out.println(reduceRs);
        System.out.println("使用reduce 寻找(\"a\",\"aa\",\"aaaaaa\",\"aaa\")最大长度字符串");
        String maxStr = jsc.parallelize(Arrays.asList("a", "aa", "aaaaaa", "aaa")).reduce(new Function2<String, String, String>() {
            @Override
            public String call(String s, String s2) throws Exception {
                return s.length() > s2.length() ? s : s2;
            }
        });
        System.out.println(maxStr);
        System.out.println("sortBy得到的结果");
        //true(默认) :从小到大，false从大到小
        jsc.parallelize(Arrays.asList(
                new Tuple2<Integer, String>(1, "a"),
                new Tuple2<Integer, String>(2, "b"),
                new Tuple2<Integer, String>(3, "c"),
                new Tuple2<Integer, String>(4, "d"),
                new Tuple2<Integer, String>(5, "e")
        )).sortBy(new Function<Tuple2<Integer, String>, String>() {
            @Override
            public String call(Tuple2<Integer, String> integerStringTuple2) throws Exception {
                return integerStringTuple2._2;
            }
        },false,1).foreach(new VoidFunction<Tuple2<Integer, String>>() {
            @Override
            public void call(Tuple2<Integer, String> integerStringTuple2) throws Exception {
                System.out.println(integerStringTuple2);
            }
        });
        System.out.println("zip方法");
        //前后rdd数量必须一致
        JavaPairRDD<String,String> zipRdd = jsc.parallelize(Arrays.asList("1", "2", "3", "4")).zip(jsc.parallelize(Arrays.asList("a","b","c","d")));
        zipRdd.foreach(new VoidFunction<Tuple2<String, String>>() {
            @Override
            public void call(Tuple2<String, String> stringStringTuple2) throws Exception {
                System.out.println(stringStringTuple2);
            }
        });
    }
}
